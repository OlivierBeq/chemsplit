"""The ``baseline`` splitter family: no structural control.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import numpy as np

from chemsplit.base import (
    BaseSplitter,
    SplitResult,
    Strictness,
    _Context,
)
from chemsplit.determinism import seed_for
from chemsplit.exceptions import ConfigurationError, EmptyPartitionError, LabelError, ParameterError

__all__ = [
    "RandomSplitter",
    "StratifiedRandomSplitter",
    "KFoldSplitter",
    "MonteCarloSplitter",
    "PredefinedSplitter",
]

_PARTITIONS = ("train", "valid", "test", "discard")


def _realised_sizes(train: np.ndarray, valid: np.ndarray, test: np.ndarray) -> dict[str, int]:
    return {"train": int(train.size), "valid": int(valid.size), "test": int(test.size)}


def _resolved_random_state_params(splitter: BaseSplitter, ctx: _Context) -> dict[str, Any]:
    """``params`` must be JSON-serialisable and "fully resolved". ``random_state`` is
    overwritten with the concrete resolved integer seed (never a raw ``Generator``, which is not
    JSON-safe) — see :meth:`BaseSplitter._sanitize_params`."""
    params = splitter._sanitize_params(splitter.get_params())
    params["random_state"] = ctx.extra["resolved_seed"]
    return params


class RandomSplitter(BaseSplitter):
    """Split records by a uniform random (or fixed-order) permutation.

    The cheapest possible split and the correct control for "is my pipeline wired correctly?" —
    see Pitfalls below for why it is the wrong tool for "will this generalise?".

    :param shuffle: If ``False``, the split is a contiguous prefix/suffix cut in input order and
        ``random_state`` is ignored (an *index split*).
    :param base: See :class:`chemsplit.base.BaseSplitter`.

    Notes
    -----
    Algorithm: draw ``perm = rng.permutation(n)`` (or ``arange(n)`` when
    ``shuffle=False``), then cut ``perm`` at ``n_train``, ``n_train+n_valid``,
    ``n_train+n_valid+n_test``. ``purpose="random.permutation"``, ``k=`` the fold index. Complexity
    ``O(n)`` time and memory.

    Advantages
    ----------
    - Linear-time and dependency-free: no featurization, no chemistry, nothing to compute.
    - The only split that leaves train and test distributionally identical by construction, so it is the right control for "is my pipeline wired correctly?" rather than "will this generalise?".
    - An unbiased estimate of interpolation error within the dataset's own chemical space — the correct answer when the deployment library is drawn from that same space (e.g. re-scoring a collection you already own).
    - Lower variance across seeds than any structural split, making small model differences easier to detect.

    Pitfalls
    --------
    - Scatters congeneric series across train and test, so near-duplicates land on both sides and reported metrics overstate prospective performance, often by 0.2-0.4 in R² or ROC-AUC.
    - On datasets built from a handful of papers, a random split can be close to a memorisation test, where a nearest-neighbour baseline matches a deep model.
    - Says nothing about scaffold generalisation, temporal drift, or assay-protocol shift.
    - Exact duplicates (salts, tautomers, unspecified stereocentres) straddle the boundary unless removed first — see `chemsplit.preprocess.find_duplicates` and `aggregate_replicates`.
    - Reporting only a random-split number is the most common cause of irreproducible QSAR results.

    """

    splitter_id: ClassVar[str] = "random"
    family: ClassVar[str] = "baseline"
    strictness: ClassVar[Strictness] = Strictness.OPTIMISTIC
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features", "interactions", "sequences")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(self, *, shuffle: bool = True, **base: Any) -> None:
        super().__init__(**base)
        self.shuffle = shuffle
        if not isinstance(shuffle, bool):
            raise ParameterError(f"shuffle must be bool, got {shuffle!r}")
        # Instance-level override of the ClassVar default: deterministic without a seed only
        # when shuffle=False (an index split). Not a @property: registry-style introspection
        # elsewhere may read this off the class or an unconstructed default instance.
        self.deterministic_without_seed = not shuffle

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        n_folds = self.get_n_splits()
        results = []
        for k in range(n_folds):
            if self.shuffle:
                rng = seed_for(ctx.rng_seeds, "random.permutation", k)
                perm = rng.permutation(n)
            else:
                perm = np.arange(n)
            a = ctx.sizes.n_train
            b = a + ctx.sizes.n_valid
            c = b + ctx.sizes.n_test
            train = np.sort(perm[:a]).astype(np.int64)
            valid = np.sort(perm[a:b]).astype(np.int64)
            test = np.sort(perm[b:c]).astype(np.int64)
            discard = np.sort(perm[c:]).astype(np.int64)
            params = _resolved_random_state_params(self, ctx)
            results.append(
                SplitResult(
                    train=train,
                    test=test,
                    valid=valid,
                    discard=discard,
                    groups=None,
                    splitter_id=self.splitter_id,
                    params=params,
                    n_records=n,
                    metadata={
                        "realised_sizes": _realised_sizes(train, valid, test),
                        "resolved_seed": ctx.extra["resolved_seed"],
                        "fold_index": k,
                    },
                )
            )
        return results


def _largest_remainder(m: int, targets: tuple[int, int, int], total: int) -> tuple[int, int, int]:
    """Hare-quota apportionment: raw_j = m*targets_j/total, floor each, remaining seats
    go to the largest fractional remainders, ties -> bucket order (train, valid, test)."""
    if total <= 0:
        return (0, 0, 0)
    raw = [m * t / total for t in targets]
    base = [math.floor(r) for r in raw]
    remainder = m - sum(base)
    fracs = sorted(range(3), key=lambda i: (-(raw[i] - base[i]), i))
    out = list(base)
    for i in fracs[:remainder]:
        out[i] += 1
    return (out[0], out[1], out[2])


def _compute_strata(
    y: np.ndarray,
    *,
    task: str,
    n_bins: int,
    binning: str,
    multitask: str,
) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        if multitask == "error":
            raise LabelError(
                "StratifiedRandomSplitter: y is 2-D (multi-task); pass multitask="
                '"first"/"sum_labels"/"iterative" or reduce y to 1-D'
            )
        if multitask == "first":
            y = y[:, 0]
        elif multitask == "sum_labels":
            y = np.nan_to_num(y, nan=0.0).sum(axis=1)
        elif multitask == "iterative":
            # Best-effort simplification of Sechidis et al.'s iterative stratification: stratify
            # on the row-sum of non-NaN positive labels, which at least balances gross label mass
            # per stratum even though it does not implement the full per-label greedy algorithm.
            y = np.nan_to_num(y, nan=0.0).sum(axis=1)
        else:
            raise ParameterError(f"invalid multitask={multitask!r}")

    if y.dtype.kind in "iuf" and np.any(np.isinf(y.astype(float))):
        raise LabelError("StratifiedRandomSplitter: y contains +/-inf")

    resolved_task = task
    if task == "auto":
        is_intlike = np.issubdtype(y.dtype, np.integer) or y.dtype == bool or y.dtype.kind in "US"
        if is_intlike:
            resolved_task = "classification"
        else:
            n_distinct = len(np.unique(y[~np.isnan(y.astype(float))])) if y.dtype.kind == "f" else len(np.unique(y))
            resolved_task = "classification" if n_distinct <= 20 else "regression"

    if resolved_task == "classification":
        _, strata = np.unique(y, return_inverse=True)
        return strata.astype(np.int64), resolved_task

    # regression: bin
    yf = y.astype(np.float64)
    if binning == "quantile":
        edges = np.quantile(yf, np.linspace(0, 1, n_bins + 1), method="linear")
        edges = np.unique(edges)
    elif binning == "uniform":
        edges = np.linspace(yf.min(), yf.max(), n_bins + 1)
        edges = np.unique(edges)
    elif binning == "kmeans":
        from sklearn.cluster import KMeans

        k = min(n_bins, len(np.unique(yf)))
        km = KMeans(n_clusters=max(1, k), n_init=10, random_state=0)
        labels = km.fit_predict(yf.reshape(-1, 1))
        # relabel by ascending cluster-centre so strata are ordered like bins
        order = np.argsort(km.cluster_centers_.ravel())
        remap = {int(old): int(new) for new, old in enumerate(order)}
        return np.asarray([remap[int(v)] for v in labels], dtype=np.int64), resolved_task
    else:
        raise ParameterError(f"invalid binning={binning!r}")

    if len(edges) < 2:
        return np.zeros(len(yf), dtype=np.int64), resolved_task
    strata = np.clip(np.searchsorted(edges, yf, side="right") - 1, 0, len(edges) - 2)
    return strata.astype(np.int64), resolved_task


class StratifiedRandomSplitter(BaseSplitter):
    """Random split stratified on the label (class balance for classification, quantile/uniform/
    k-means bins for regression), sized per-stratum by largest-remainder apportionment.

    :param task: ``"auto"``, ``"classification"``, or ``"regression"``.
    :param n_bins: Number of quantile/uniform/k-means bins for regression stratification.
    :param binning: ``"quantile"``, ``"uniform"``, or ``"kmeans"``.
    :param min_per_stratum: Minimum stratum size before ``on_small_stratum`` kicks in.
    :param on_small_stratum: ``"merge"``, ``"raise"``, or ``"ignore"``.
    :param multitask: ``"error"``, ``"first"``, ``"sum_labels"``, or ``"iterative"``.
    :param base: See :class:`chemsplit.base.BaseSplitter`.

    Notes
    -----
    Per-stratum quotas use Hare-quota (largest-remainder) apportionment,
    which guarantees exact totals and off-target deviation of at most one record per stratum.
    ``purpose="stratified.permutation"``. The ``multitask="iterative"`` path is a documented
    best-effort simplification (row-sum-weighted stratification), not the full Sechidis et al.
    greedy per-label algorithm.

    Advantages
    ----------
    - Guarantees every class, or every label decile, appears in every partition — essential for imbalanced data where a naive random split can leave a fold with zero actives.
    - Cuts the variance of ROC-AUC/PR-AUC/R² estimates on small datasets, often more than any change to the model.
    - Largest-remainder apportionment keeps realised sizes exactly reproducible and off by at most one record per stratum.
    - The same mechanism handles regression through quantile binning, so one splitter serves both task types.

    Pitfalls
    --------
    - Only controls the **label** distribution; says nothing about chemical similarity, and the word "stratified" tempts people to treat it as a rigorous split.
    - Quantile binning on a heavily tied label (e.g. a censored `pIC50 = 5.0` for every inactive) produces degenerate bins — check `metadata["bin_edges"]`.
    - Stratifying on the label leaks the label distribution into the split design; on very small `n` this mildly biases the test set toward looking like train.
    - Multi-task stratification is genuinely hard: `multitask="sum_labels"` is a crude heuristic and `"iterative"` only approximate. Prefer `BalancedMultiTaskSplitter` for sparse multi-task matrices.
    - Merging small strata changes the effective `n_bins`; read it back from `metadata["n_strata"]` instead of assuming the requested value held.

    """

    splitter_id: ClassVar[str] = "stratified_random"
    family: ClassVar[str] = "baseline"
    strictness: ClassVar[Strictness] = Strictness.OPTIMISTIC
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features", "interactions")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True

    def __init__(
        self,
        *,
        task: Literal["auto", "classification", "regression"] = "auto",
        n_bins: int = 10,
        binning: Literal["quantile", "uniform", "kmeans"] = "quantile",
        min_per_stratum: int = 2,
        on_small_stratum: Literal["merge", "raise", "ignore"] = "merge",
        multitask: Literal["error", "first", "sum_labels", "iterative"] = "error",
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.task = task
        self.n_bins = n_bins
        self.binning = binning
        self.min_per_stratum = min_per_stratum
        self.on_small_stratum = on_small_stratum
        self.multitask = multitask
        if task not in ("auto", "classification", "regression"):
            raise ParameterError(f"invalid task={task!r}")
        if not (2 <= n_bins):
            raise ParameterError(f"n_bins must be >= 2, got {n_bins!r}")
        if binning not in ("quantile", "uniform", "kmeans"):
            raise ParameterError(f"invalid binning={binning!r}")
        if min_per_stratum < 1:
            raise ParameterError(f"min_per_stratum must be >= 1, got {min_per_stratum!r}")
        if on_small_stratum not in ("merge", "raise", "ignore"):
            raise ParameterError(f"invalid on_small_stratum={on_small_stratum!r}")
        if multitask not in ("error", "first", "sum_labels", "iterative"):
            raise ParameterError(f"invalid multitask={multitask!r}")

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        strata, resolved_task = _compute_strata(
            ctx.y, task=self.task, n_bins=self.n_bins, binning=self.binning, multitask=self.multitask
        )
        strata = self._handle_small_strata(strata, resolved_task)

        n_folds = self.get_n_splits()
        results = []
        for k in range(n_folds):
            rng = seed_for(ctx.rng_seeds, "stratified.permutation", k)
            train_idx: list[int] = []
            valid_idx: list[int] = []
            test_idx: list[int] = []
            discard_idx: list[int] = []
            for s in sorted(np.unique(strata).tolist()):
                members = np.nonzero(strata == s)[0]
                perm = members[rng.permutation(len(members))]
                m = len(members)
                q_tr, q_va, q_te = _largest_remainder(
                    m, (ctx.sizes.n_train, ctx.sizes.n_valid, ctx.sizes.n_test), ctx.n
                )
                c1, c2, c3 = q_tr, q_tr + q_va, q_tr + q_va + q_te
                train_idx.extend(perm[:c1].tolist())
                valid_idx.extend(perm[c1:c2].tolist())
                test_idx.extend(perm[c2:c3].tolist())
                discard_idx.extend(perm[c3:].tolist())

            train = np.sort(np.asarray(train_idx, dtype=np.int64))
            valid = np.sort(np.asarray(valid_idx, dtype=np.int64))
            test = np.sort(np.asarray(test_idx, dtype=np.int64))
            discard = np.sort(np.asarray(discard_idx, dtype=np.int64))
            params = _resolved_random_state_params(self, ctx)
            results.append(
                SplitResult(
                    train=train,
                    test=test,
                    valid=valid,
                    discard=discard,
                    groups=None,
                    splitter_id=self.splitter_id,
                    params=params,
                    n_records=ctx.n,
                    metadata={
                        "n_strata": int(len(np.unique(strata))),
                        "stratum_sizes": [
                            int(np.sum(strata == s)) for s in sorted(np.unique(strata).tolist())
                        ],
                        "task_resolved": self.task,
                        "realised_sizes": _realised_sizes(train, valid, test),
                        "resolved_seed": ctx.extra["resolved_seed"],
                        "fold_index": k,
                    },
                )
            )
        return results

    def _handle_small_strata(self, strata: np.ndarray, resolved_task: str) -> np.ndarray:
        counts = {int(s): int(np.sum(strata == s)) for s in np.unique(strata)}
        small = [s for s, c in counts.items() if c < self.min_per_stratum]
        if not small:
            return strata
        if self.on_small_stratum == "raise":
            raise LabelError(
                f"StratifiedRandomSplitter: stratum/strata {small} have fewer than "
                f"min_per_stratum={self.min_per_stratum} members"
            )
        if self.on_small_stratum == "ignore":
            return strata
        # Regression bins are ordered (merge by index); classification codes aren't (merge by
        # frequency, ties -> lowest index).
        strata = strata.copy()
        all_strata = sorted(counts)
        for s in small:
            others = [o for o in all_strata if o != s and counts[o] >= self.min_per_stratum]
            if not others:
                continue
            if resolved_task == "classification":
                target = max(others, key=lambda o: (counts[o], -o))
            else:
                target = min(others, key=lambda o: (abs(o - s), o))
            strata[strata == s] = target
        # relabel densely in ascending order
        remap = {old: new for new, old in enumerate(sorted(np.unique(strata).tolist()))}
        return np.asarray([remap[int(v)] for v in strata], dtype=np.int64)


class KFoldSplitter(BaseSplitter):
    """Standard (optionally stratified, optionally leave-one-out) k-fold cross-validation.

    :param n_splits: Number of folds, or ``"loo"`` for leave-one-out.
    :param shuffle: Whether to shuffle before folding.
    :param stratify: Whether to stratify folds on the label.
    :param stratify_kwargs: Extra kwargs forwarded to the stratification helper, or ``None``.
    :param base: See :class:`chemsplit.base.BaseSplitter`. ``train_size``/``valid_size``/
        ``test_size`` MUST be left ``None`` here (fold sizes are determined by ``n_splits``);
        passing any raises :class:`chemsplit.exceptions.ConfigurationError`.

    Notes
    -----
    ``"loo"`` sets ``n_splits = n`` (leave-one-out; guarded above ``n=10_000`` unless
    ``allow_large_loo=True`` is passed to :meth:`split`). Unshuffled folds are index-contiguous and
    match ``sklearn.model_selection.KFold(shuffle=False)`` fold *contents* exactly.
    ``purpose="kfold.permutation"``, ``k=0`` — one permutation shared by every fold, not one per
    fold.

    Advantages
    ----------
    - Every record serves for both training and evaluation, cutting estimator variance on the small assay-sized datasets (n < 2000) this is typically used on.
    - Yields a *distribution* of scores instead of a single number, so model comparisons can be tested statistically rather than eyeballed.
    - Composes with any grouping — feed group labels from any group-forming splitter into `GroupKFoldSplitter` for the structural analogue.

    Pitfalls
    --------
    - K-fold is a **resampling protocol, not a split criterion**: plain `KFoldSplitter` inherits every weakness of `random`, and calling a model "cross-validated" says nothing about chemical generalisation.
    - Fold scores aren't independent — training sets overlap by `(k-2)/(k-1)` — so a naive standard error across folds understates uncertainty. Use repeated k-fold (`repeated`) and report the spread between repeats.
    - Selecting hyperparameters on the same folds used for reporting inflates the score; use `NestedCVSplitter` instead.
    - Leave-one-out has very high variance for classification metrics and ROC-AUC is undefined per fold, so per-fold ranking metrics are left to the caller rather than computed here.

    """

    splitter_id: ClassVar[str] = "k_fold"
    family: ClassVar[str] = "baseline"
    strictness: ClassVar[Strictness] = Strictness.OPTIMISTIC
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = (
        "smiles",
        "mol",
        "features",
        "interactions",
        "sequences",
    )
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_method: ClassVar[bool] = True

    def __init__(
        self,
        *,
        n_splits: int | Literal["loo"] = 5,
        shuffle: bool = True,
        stratify: bool = False,
        stratify_kwargs: dict[str, Any] | None = None,
        **base: Any,
    ) -> None:
        super().__init__(n_splits=n_splits, **base)
        self.shuffle = shuffle
        self.stratify = stratify
        self.stratify_kwargs = stratify_kwargs
        if not isinstance(n_splits, (int, np.integer)) and n_splits != "loo":
            raise ParameterError(f"n_splits must be an int or 'loo', got {n_splits!r}")
        if isinstance(n_splits, (int, np.integer)) and n_splits < 2:
            raise ParameterError(f"n_splits must be >= 2, got {n_splits!r}")
        if not isinstance(shuffle, bool):
            raise ParameterError(f"shuffle must be bool, got {shuffle!r}")
        if not isinstance(stratify, bool):
            raise ParameterError(f"stratify must be bool, got {stratify!r}")
        if not stratify and stratify_kwargs:
            raise ParameterError("stratify_kwargs given but stratify=False")
        for name in ("train_size", "valid_size", "test_size"):
            if getattr(self, name) is not None:
                raise ConfigurationError(
                    f"KFoldSplitter: {name} must be None -- fold sizes are determined by n_splits"
                )

    def _resolve_k(self, n: int) -> int:
        return n if self.n_splits == "loo" else int(self.n_splits)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        if self.n_splits == "loo":
            if X is None:
                return 1
            from chemsplit import preprocess

            x_kind = preprocess.detect_input_kind(X, None)
            return preprocess.input_length(X, x_kind)
        return int(self.n_splits)

    def _check_preconditions(self, ctx: _Context) -> None:
        k = self._resolve_k(ctx.n)
        if ctx.n < k:
            raise ParameterError(f"n_splits={k} exceeds n={ctx.n}")
        if self.n_splits == "loo" and ctx.n > 10_000 and not ctx.extra.get("allow_large_loo"):
            from chemsplit.exceptions import ScalabilityError

            raise ScalabilityError(
                f"KFoldSplitter(n_splits='loo') on n={ctx.n} > 10,000 is almost certainly a "
                "mistake (n folds, each near-O(n) train set). Pass allow_large_loo=True to "
                "split() to proceed anyway."
            )

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        k = self._resolve_k(n)
        if self.shuffle:
            perm = seed_for(ctx.rng_seeds, "kfold.permutation", 0).permutation(n)
        else:
            perm = np.arange(n)

        if self.stratify:
            strata, _ = _compute_strata(ctx.y, task="auto", n_bins=10, binning="quantile", multitask="error")
            folds: list[list[int]] = [[] for _ in range(k)]
            for s in sorted(np.unique(strata).tolist()):
                members = [int(i) for i in perm if strata[i] == s]
                for pos, idx in enumerate(members):
                    folds[pos % k].append(idx)
        else:
            fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
            folds = []
            start = 0
            for size in fold_sizes:
                folds.append([int(v) for v in perm[start: start + size]])
                start += size

        results = []
        for i in range(k):
            test = np.sort(np.asarray(folds[i], dtype=np.int64))
            train = np.sort(np.asarray([idx for j in range(k) if j != i for idx in folds[j]], dtype=np.int64))
            params = _resolved_random_state_params(self, ctx)
            results.append(
                SplitResult(
                    train=train,
                    test=test,
                    valid=np.array([], dtype=np.int64),
                    discard=np.array([], dtype=np.int64),
                    groups=None,
                    splitter_id=self.splitter_id,
                    params=params,
                    n_records=n,
                    metadata={
                        "fold_index": i,
                        "n_splits": k,
                        "fold_sizes": [len(f) for f in folds],
                    },
                )
            )
        return results


class MonteCarloSplitter(BaseSplitter):
    """Repeated independent random splits (``ShuffleSplit``); unlike :class:`KFoldSplitter`, test
    sets across repeats are NOT disjoint.

    :param n_splits: Number of repeats.
    :param stratify: Whether to stratify each repeat on the label.
    :param stratify_kwargs: Extra kwargs forwarded to the stratification helper, or ``None``.
    :param base: See :class:`chemsplit.base.BaseSplitter`.

    Notes
    -----
    Delegates each repeat to :class:`RandomSplitter`/:class:`StratifiedRandomSplitter` logic with
    ``purpose="montecarlo.permutation"``, ``k=`` the repeat index.

    Advantages
    ----------
    - Decouples test-set size from the repeat count, so a 10% test set can be evaluated 50 times — not possible with k-fold.
    - The spread across repeats directly estimates split-induced variance, which for chemical data often exceeds the gap between the models being compared.

    Pitfalls
    --------
    - Test sets overlap across repeats, so results are correlated and the naive standard error is optimistic.
    - Some records may never land in any test set (probability `(1-p)^n_splits`); the last fold's `metadata` reports coverage, and callers needing full coverage should use `k_fold`.
    - Inherits every chemical-leakage weakness of `random`.

    """

    splitter_id: ClassVar[str] = "monte_carlo"
    family: ClassVar[str] = "baseline"
    strictness: ClassVar[Strictness] = Strictness.OPTIMISTIC
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = (
        "smiles",
        "mol",
        "features",
        "interactions",
        "sequences",
    )
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_method: ClassVar[bool] = True

    def __init__(
        self,
        *,
        n_splits: int = 10,
        stratify: bool = False,
        stratify_kwargs: dict[str, Any] | None = None,
        **base: Any,
    ) -> None:
        super().__init__(n_splits=n_splits, **base)
        self.stratify = stratify
        self.stratify_kwargs = stratify_kwargs
        if n_splits < 1:
            raise ParameterError(f"n_splits must be >= 1, got {n_splits!r}")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return int(self.n_splits)

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        results = []
        for i in range(int(self.n_splits)):
            if self.stratify:
                strata, _ = _compute_strata(
                    ctx.y, task="auto", n_bins=10, binning="quantile", multitask="error"
                )
                rng = seed_for(ctx.rng_seeds, "montecarlo.permutation", i)
                train_idx: list[int] = []
                valid_idx: list[int] = []
                test_idx: list[int] = []
                discard_idx: list[int] = []
                for s in sorted(np.unique(strata).tolist()):
                    members = np.nonzero(strata == s)[0]
                    perm = members[rng.permutation(len(members))]
                    m = len(members)
                    q_tr, q_va, q_te = _largest_remainder(
                        m, (ctx.sizes.n_train, ctx.sizes.n_valid, ctx.sizes.n_test), n
                    )
                    c1, c2, c3 = q_tr, q_tr + q_va, q_tr + q_va + q_te
                    train_idx.extend(perm[:c1].tolist())
                    valid_idx.extend(perm[c1:c2].tolist())
                    test_idx.extend(perm[c2:c3].tolist())
                    discard_idx.extend(perm[c3:].tolist())
                train = np.sort(np.asarray(train_idx, dtype=np.int64))
                valid = np.sort(np.asarray(valid_idx, dtype=np.int64))
                test = np.sort(np.asarray(test_idx, dtype=np.int64))
                discard = np.sort(np.asarray(discard_idx, dtype=np.int64))
            else:
                rng = seed_for(ctx.rng_seeds, "montecarlo.permutation", i)
                perm = rng.permutation(n)
                a = ctx.sizes.n_train
                b = a + ctx.sizes.n_valid
                c = b + ctx.sizes.n_test
                train = np.sort(perm[:a]).astype(np.int64)
                valid = np.sort(perm[a:b]).astype(np.int64)
                test = np.sort(perm[b:c]).astype(np.int64)
                discard = np.sort(perm[c:]).astype(np.int64)

            params = _resolved_random_state_params(self, ctx)
            results.append(
                SplitResult(
                    train=train,
                    test=test,
                    valid=valid,
                    discard=discard,
                    groups=None,
                    splitter_id=self.splitter_id,
                    params=params,
                    n_records=n,
                    metadata={
                        "repeat_index": i,
                        "realised_sizes": _realised_sizes(train, valid, test),
                        "resolved_seed": ctx.extra["resolved_seed"],
                    },
                )
            )
        return results


class PredefinedSplitter(BaseSplitter):
    """Wrap an externally supplied train/valid/test/discard assignment.

    The only way to reproduce a published benchmark's numbers exactly, and the correct way to
    replace an internal (non-shareable) time split with a shareable index list.

    :param assignment: Sequence of partition names, or mapping of partition name to index list.
        Sequence form: length ``n``, values in ``{"train","valid","test","discard"}``. Mapping
        form: partition name -> index list; disjoint; unlisted indices go to ``discard``.
    :param fold_column: Per-record fold id, or ``None``. Values ``>= 0`` are fold ids; ``-1``
        means "always train". Yields ``n_splits = n_distinct(non-negative)`` folds.
    :param base: See :class:`chemsplit.base.BaseSplitter`. ``train_size``/``valid_size``/
        ``test_size`` MUST be ``None``.

    Notes
    -----
    No randomness; direct construction. Exactly one of ``assignment``/``fold_column`` must be
    given.

    Advantages
    ----------
    - The only way to reproduce a published benchmark's numbers exactly, which is the whole point of cross-paper comparability.
    - Zero ambiguity: the split is data, not an algorithm, so it can be shipped, diffed, and checksummed.
    - Lets an internal time split that can't be published be replaced with a shareable index list.

    Pitfalls
    --------
    - Comparability is the *only* guarantee — several widely used benchmark splits, including some MoleculeNet scaffold splits, contain near-duplicate leakage across the boundary, and inheriting the split inherits the flaw.
    - A published split is tied to a specific row order; filtering, deduplicating, or re-standardising the dataset silently shifts what the indices point to. Always re-key on InChIKey and verify with `chemsplit.audit.audit_split`.
    - Encourages leaderboard over-fitting, since the community tunes against one fixed test set for years.

    """

    splitter_id: ClassVar[str] = "predefined"
    family: ClassVar[str] = "baseline"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = (
        "smiles",
        "mol",
        "features",
        "interactions",
        "sequences",
    )
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = True
    deterministic_method: ClassVar[bool] = True

    def __init__(
        self,
        *,
        assignment: Any = None,
        fold_column: Any = None,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.assignment = assignment
        self.fold_column = fold_column
        if (assignment is None) == (fold_column is None):
            raise ParameterError(
                "PredefinedSplitter: exactly one of assignment/fold_column must be given"
            )
        for name in ("train_size", "valid_size", "test_size"):
            if getattr(self, name) is not None:
                raise ConfigurationError(f"PredefinedSplitter: {name} must be None")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        if self.fold_column is not None:
            return len({v for v in self.fold_column if v is not None and v >= 0})
        return 1

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        if self.assignment is not None:
            return [self._from_assignment(ctx)]
        return self._from_fold_column(ctx)

    def _from_assignment(self, ctx: _Context) -> SplitResult:
        n = ctx.n
        buckets: dict[str, list[int]] = {p: [] for p in _PARTITIONS}
        if isinstance(self.assignment, dict):
            seen: set[int] = set()
            for name, idxs in self.assignment.items():
                if name not in _PARTITIONS:
                    raise ParameterError(f"PredefinedSplitter: unknown partition name {name!r}")
                for i in idxs:
                    if i in seen:
                        raise ParameterError(f"PredefinedSplitter: index {i} assigned twice")
                    seen.add(i)
                    buckets[name].append(int(i))
            for i in range(n):
                if i not in seen:
                    buckets["discard"].append(i)
        else:
            values = list(self.assignment)
            if len(values) != n:
                raise ParameterError(
                    f"PredefinedSplitter: assignment has length {len(values)}, expected n={n}"
                )
            bad = sorted({v for v in values if v not in _PARTITIONS})
            if bad:
                raise ParameterError(f"PredefinedSplitter: invalid partition values {bad}")
            for i, v in enumerate(values):
                buckets[v].append(i)

        train = np.sort(np.asarray(buckets["train"], dtype=np.int64))
        valid = np.sort(np.asarray(buckets["valid"], dtype=np.int64))
        test = np.sort(np.asarray(buckets["test"], dtype=np.int64))
        discard = np.sort(np.asarray(buckets["discard"], dtype=np.int64))
        if train.size == 0 or test.size == 0:
            raise EmptyPartitionError("PredefinedSplitter: train and test must both be non-empty")

        params = self._sanitize_params(self.get_params())
        return SplitResult(
            train=train,
            test=test,
            valid=valid,
            discard=discard,
            groups=None,
            splitter_id=self.splitter_id,
            params=params,
            n_records=n,
            metadata={"source": "assignment", "realised_sizes": _realised_sizes(train, valid, test)},
        )

    def _from_fold_column(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        values = list(self.fold_column)
        if len(values) != n:
            raise ParameterError(
                f"PredefinedSplitter: fold_column has length {len(values)}, expected n={n}"
            )
        fold_ids = sorted({v for v in values if v >= 0})
        if not fold_ids:
            raise ParameterError("PredefinedSplitter: fold_column has no non-negative fold ids")
        results = []
        for f in fold_ids:
            test = np.sort(np.asarray([i for i, v in enumerate(values) if v == f], dtype=np.int64))
            train = np.sort(
                np.asarray([i for i, v in enumerate(values) if v != f], dtype=np.int64)
            )
            if train.size == 0 or test.size == 0:
                raise EmptyPartitionError(
                    f"PredefinedSplitter: fold {f} produced an empty train or test partition"
                )
            params = self._sanitize_params(self.get_params())
            results.append(
                SplitResult(
                    train=train,
                    test=test,
                    valid=np.array([], dtype=np.int64),
                    discard=np.array([], dtype=np.int64),
                    groups=None,
                    splitter_id=self.splitter_id,
                    params=params,
                    n_records=n,
                    metadata={
                        "source": "fold_column",
                        "fold_index": int(f),
                        "realised_sizes": _realised_sizes(train, np.array([], dtype=np.int64), test),
                    },
                )
            )
        return results
