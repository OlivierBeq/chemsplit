"""The ``property``/label-shift splitter family.

All five classes here are plain :class:`~chemsplit.base.BaseSplitter` subclasses
(``group_forming = False``): none of them forms groups, they all cut a sorted/derived axis
directly.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, Sequence

import numpy as np

from chemsplit._fp_similarity import SimilarityParamsMixin, compute_similarity_matrix
from chemsplit.base import BaseSplitter, SplitResult, _Context
from chemsplit.determinism import argmin_tiebreak, seed_for, stable_sort
from chemsplit.exceptions import ConfigurationError, ConstraintUnsatisfiableError, InputError, LabelError, ParameterError

__all__ = [
    "AdversarialSplitter",
    "LabelExtrapolationSplitter",
    "MOODSplitter",
    "PropertySplitter",
    "StratifiedDistributionSplitter",
]

EPS = 1e-9


# ---------------------------------------------------------------------------
# Shared helpers (local to this module — every family is built independently)
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively coerce ``value`` into something ``json.dumps``-round-trippable, for
    ``SplitResult.params`` (invariant I4). Callables/splitter instances/arrays are not JSON
    types, so they are given a stable, informative string/dict representation instead of raising.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, BaseSplitter):
        return {"class": type(value).__name__, "splitter_id": getattr(value, "splitter_id", None)}
    if callable(value):
        return f"<callable:{getattr(value, '__name__', repr(value))}>"
    return str(value)


def _build_result(
    splitter: BaseSplitter,
    ctx: _Context,
    *,
    train: np.ndarray,
    valid: np.ndarray,
    test: np.ndarray,
    discard: np.ndarray,
    metadata: dict[str, Any],
) -> SplitResult:
    """Non-group counterpart of ``GroupSplitter._build_result`` (chemsplit/base.py) — every
    ``property``-family splitter is a plain ``BaseSplitter``, so there is no shared builder to
    reuse from core infra; this local one keeps the pattern (sorted arrays, ``realised_sizes``,
    JSON-safe params) consistent across the five classes in this module.
    """
    full_metadata = {
        "realised_sizes": {
            "train": int(train.size),
            "valid": int(valid.size),
            "test": int(test.size),
        },
        **metadata,
    }
    return SplitResult(
        train=np.sort(np.asarray(train, dtype=np.int64)),
        valid=np.sort(np.asarray(valid, dtype=np.int64)),
        test=np.sort(np.asarray(test, dtype=np.int64)),
        discard=np.sort(np.asarray(discard, dtype=np.int64)),
        groups=None,
        splitter_id=splitter.splitter_id,
        params=_json_safe(splitter.get_params()),
        n_records=ctx.n,
        metadata=full_metadata,
    )


def _resolve_descriptor_values(
    ctx: _Context, property_: "str | Callable[[Any], float]", property_values: Sequence[float] | None
) -> np.ndarray:
    if property_values is not None:
        v = np.asarray(property_values, dtype=np.float64)
        if v.shape != (ctx.n,):
            raise ParameterError(f"property_values must have shape ({ctx.n},), got {v.shape}")
        return v
    if ctx.mols is None:
        raise InputError("descriptor computation requires molecules (accepts='smiles'/'mol')")
    if callable(property_) and not isinstance(property_, str):
        fn = property_
        name = getattr(property_, "__name__", "callable")
    else:
        from rdkit.Chem import Descriptors

        if not hasattr(Descriptors, property_):
            near = [n for n in dir(Descriptors) if property_.lower() in n.lower()][:5]
            raise ParameterError(
                f"unknown RDKit descriptor {property_!r}; closest names in "
                f"rdkit.Chem.Descriptors: {near}"
            )
        fn = getattr(Descriptors, property_)
        name = property_
    values = np.empty(ctx.n, dtype=np.float64)
    bad: list[int] = []
    for i, mol in enumerate(ctx.mols):
        if mol is None:
            bad.append(i)
            values[i] = math.nan
            continue
        try:
            values[i] = float(fn(mol))
        except Exception:
            bad.append(i)
            values[i] = math.nan
    if bad or not np.all(np.isfinite(values)):
        offenders = [i for i in range(ctx.n) if not np.isfinite(values[i])][:20]
        raise InputError(f"descriptor {name!r} produced non-finite values at indices {offenders}")
    return values


def _select_top_k_with_ties(
    order: list[int],
    v: np.ndarray,
    k: int,
    *,
    from_end: bool,
    tie_policy: Literal["by_index", "random", "keep_together"],
    rng: np.random.Generator | None,
) -> tuple[list[int], int]:
    """Select ``k`` indices from ``order`` (ascending-by-value, ties ascending-by-index), taking
    the highest ``k`` if ``from_end`` else the lowest ``k``, honouring ``tie_policy`` at the cut
    boundary. Returns ``(selected, overshoot)`` — ``overshoot`` is non-zero only
    for ``"keep_together"``, which may select more than ``k``.
    """
    n = len(order)
    if k <= 0:
        return [], 0
    if k >= n:
        return list(order), 0

    if from_end:
        boundary_val = v[order[n - k]]
        hi = n
        lo = n - k
        while lo > 0 and v[order[lo - 1]] == boundary_val:
            lo -= 1
        block = order[lo:hi]
        definite = order[hi:]  # strictly greater than boundary_val -> always selected (empty here)
        # everything strictly greater than boundary_val sits to the right of the block, i.e. none
        # (block already extends to n); definite-selected-outside-block are values > boundary_val
        strictly_beyond = [i for i in order[lo:] if v[i] != boundary_val]
    else:
        boundary_val = v[order[k - 1]]
        hi = k
        while hi < n and v[order[hi]] == boundary_val:
            hi += 1
        block = order[:hi]
        strictly_beyond = [i for i in order[:hi] if v[i] != boundary_val]

    tie_block = [i for i in block if v[i] == boundary_val]

    if tie_policy == "by_index":
        return (list(order[-k:]) if from_end else list(order[:k])), 0
    if tie_policy == "keep_together":
        selected = block
        overshoot = max(0, len(selected) - k)
        return list(selected), overshoot
    # "random": exactly k total; the non-tied members strictly on the selected side are forced in,
    # the remaining slots are filled by a seeded shuffle of the tie block.
    n_forced = len(strictly_beyond)
    needed_from_tie = max(0, k - n_forced)
    shuffled_tie = list(tie_block)
    if rng is not None:
        rng.shuffle(shuffled_tie)
    chosen_tie = shuffled_tie[:needed_from_tie]
    selected = list(strictly_beyond) + chosen_tie
    return selected, 0


def _cut_by_direction(
    ctx: _Context,
    v: np.ndarray,
    n_test: int,
    direction: str,
    tie_policy: str,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, int]:
    order = stable_sort(list(range(ctx.n)), key=lambda i: v[i])
    if direction == "high_test":
        test, overshoot = _select_top_k_with_ties(order, v, n_test, from_end=True, tie_policy=tie_policy, rng=rng)
    elif direction == "low_test":
        test, overshoot = _select_top_k_with_ties(order, v, n_test, from_end=False, tie_policy=tie_policy, rng=rng)
    elif direction == "extremes_test":
        lo_k = n_test // 2
        hi_k = n_test - lo_k
        lo_sel, lo_over = _select_top_k_with_ties(order, v, lo_k, from_end=False, tie_policy=tie_policy, rng=rng)
        hi_sel, hi_over = _select_top_k_with_ties(order, v, hi_k, from_end=True, tie_policy=tie_policy, rng=rng)
        test = list(dict.fromkeys(lo_sel + hi_sel))
        overshoot = lo_over + hi_over
    elif direction == "middle_test":
        start = (ctx.n - n_test) // 2
        test = order[start: start + n_test]
        overshoot = 0
    else:
        raise ParameterError(f"invalid direction {direction!r}")
    return np.asarray(sorted(set(int(i) for i in test)), dtype=np.int64), overshoot


# ---------------------------------------------------------------------------
# PropertySplitter
# ---------------------------------------------------------------------------


class PropertySplitter(BaseSplitter):
    """Split along a continuous molecular property.

    A single sorted cut on an RDKit descriptor (or a caller-supplied callable/array of
    precomputed values): the highest, lowest, both tails, or the central band of ``property``
    become ``test``. The validation band, when requested, is drawn adjacent to ``test`` on the
    train side so early stopping shares the same extrapolation direction as the test evaluation.

    Parameters
    ----------
    property: str or callable, default ``"MolWt"``
        An RDKit descriptor name resolvable through ``rdkit.Chem.Descriptors``, or a callable
        ``Mol -> float``.
    direction: {"high_test", "low_test", "extremes_test", "middle_test"}, default "high_test"
        Which end(s) of the sorted property values become ``test``.
    property_values: sequence of float, optional
        Precomputed values, bypassing RDKit descriptor computation.
    tie_policy: {"by_index", "random", "keep_together"}, default "by_index"
        How records tied at the cut boundary are resolved.

    Attributes
    ----------
    splitter_id: str
        ``"property"``.

    Advantages
    ----------
    - Directly models real extrapolations: fragment-to-lead growth (MW), applying a small-molecule model to peptides or PROTACs (size), and solubility/permeability range shifts (logP, TPSA).
    - Fully deterministic and easy to explain — `train_range` vs. `test_range` and `overlap` state exactly what was asked of the model.
    - No featurization, clustering, or seed needed — the cheapest hard split available.

    Pitfalls
    --------
    - Molecular weight correlates with almost everything, including assay artefacts, promiscuity, and the era a series was made — a performance drop may be confounded rather than caused by the property shift.
    - Because it's a single sorted cut, the test set is chemically homogeneous with strongly correlated errors, so the effective sample size is well below `n_test`.
    - **Not** a leakage-control split — a test molecule can be a close analogue of a training molecule that just happens to sit below the cut. Check `metadata["overlap"]` and `audit.nn_similarity_profile`.
    - `direction="middle_test"` is an interpolation test despite living in this family — don't report it as extrapolation.
    - The validation band sits adjacent to test by design, which keeps early stopping honest but makes the validation score optimistic relative to test.

    Notes
    -----
    Determinism: ``purpose="property.tie"`` only when ``tie_policy="random"``; otherwise seed-free
    (``deterministic_without_seed = True`` unless ``tie_policy == "random"``).
    """

    splitter_id = "property"
    family = "property"
    strictness = "strict"
    group_forming = False
    requires_labels = False
    accepts = ("smiles", "mol", "features")
    extras: tuple[str,...] = ()
    deterministic_method = True
    order_invariant = False

    def __init__(
        self,
        *,
        property: "str | Callable[[Any], float]" = "MolWt",
        direction: Literal["high_test", "low_test", "extremes_test", "middle_test"] = "high_test",
        property_values: Sequence[float] | None = None,
        tie_policy: Literal["by_index", "random", "keep_together"] = "by_index",
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.property = property
        self.direction = direction
        self.property_values = property_values
        self.tie_policy = tie_policy
        if direction not in ("high_test", "low_test", "extremes_test", "middle_test"):
            raise ParameterError(f"invalid direction {direction!r}")
        if tie_policy not in ("by_index", "random", "keep_together"):
            raise ParameterError(f"invalid tie_policy {tie_policy!r}")

    @property
    def deterministic_without_seed(self) -> bool:  # type: ignore[override]
        return self.tie_policy != "random"

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        v = _resolve_descriptor_values(ctx, self.property, self.property_values)
        rng = seed_for(ctx.rng_seeds, "property.tie", 0) if self.tie_policy == "random" else None
        test, overshoot = _cut_by_direction(ctx, v, ctx.sizes.n_test, self.direction, self.tie_policy, rng)

        remaining = np.asarray([i for i in range(ctx.n) if i not in set(test.tolist())], dtype=np.int64)
        n_valid = ctx.sizes.n_valid
        if n_valid > 0 and remaining.size:
            # validation band: the n_valid remaining records nearest the test band (by descriptor
            # value), so it shares the extrapolation direction of test.
            test_vals = v[test]
            lo_bound, hi_bound = (test_vals.min(), test_vals.max()) if test.size else (0.0, 0.0)
            dist = np.minimum(np.abs(v[remaining] - lo_bound), np.abs(v[remaining] - hi_bound))
            order_by_dist = stable_sort(list(range(remaining.size)), key=lambda j: dist[j])
            valid_local = order_by_dist[:n_valid]
            valid = remaining[valid_local]
            train = np.asarray([i for i in remaining if i not in set(valid.tolist())], dtype=np.int64)
        else:
            valid = np.array([], dtype=np.int64)
            train = remaining

        overlap = 0.0
        if test.size and train.size:
            tr_lo, tr_hi = v[train].min(), v[train].max()
            overlap = float(np.mean((v[test] >= tr_lo) & (v[test] <= tr_hi)))

        metadata: dict[str, Any] = {
            "property": self.property if isinstance(self.property, str) else "<callable>",
            "direction": self.direction,
            "train_range": [float(v[train].min()), float(v[train].max())] if train.size else None,
            "test_range": [float(v[test].min()), float(v[test].max())] if test.size else None,
            "valid_range": [float(v[valid].min()), float(v[valid].max())] if valid.size else None,
            "overlap": overlap,
            "n_ties_at_boundary": int(overshoot),
        }
        result = _build_result(self, ctx, train=train, valid=valid, test=test, discard=np.array([], dtype=np.int64), metadata=metadata)
        return [result]


# ---------------------------------------------------------------------------
# LabelExtrapolationSplitter
# ---------------------------------------------------------------------------


class LabelExtrapolationSplitter(BaseSplitter):
    """Train on one part of the label range, test on another.

    Parameters
    ----------
    direction: {"high_test", "low_test", "extremes_test"}, default "high_test"
    task_index: int, default 0
        Column of ``y`` to extrapolate on, when ``y`` is 2-D.
    buffer: float or int, default 0.0
        A gap between train and test in label units (float) or records (int); records inside the
        buffer go to ``discard``.
    tie_policy: {"by_index", "random", "keep_together"}, default "keep_together"
        Default differs from ``property``: splitting a block of identical labels across the
        boundary is meaningless for a label-based extrapolation.

    Advantages
    ----------
    - The most honest test of whether a model can rank compounds *better than its training data* — the real requirement for generative design and prioritising untested potency ranges.
    - `buffer` makes the extrapolation gap explicit and tunable, so performance can be reported as a function of gap width.
    - Needs no chemistry at all, so it works for any modality.

    Pitfalls
    --------
    - Brutal by construction — most regression models regress to their training mean and under-predict the held-out extreme, so a flat prediction can post a respectable RMSE with zero rank correlation. **Always report a ranking metric (Spearman, top-k enrichment) alongside RMSE/R²**, since R² can be negative while ranking is still useful, or vice versa.
    - Selecting the split with the labels makes it label-aware by construction: the test set is defined by `y`, so label noise at the extreme directly shapes the test population.
    - Extreme labels concentrate measurement artefacts, censored values, and transcription errors — exactly where data quality is worst.
    - Distributional metrics on a truncated label range aren't comparable to the same metrics on a random split — don't mix them in one table without saying so.
    - On binary labels this becomes a class holdout where the model never sees a positive example — a different, usually pointless, experiment.

    """

    splitter_id = "label_extrapolation"
    family = "property"
    strictness = "extrapolative"
    group_forming = False
    requires_labels = True
    accepts = ("smiles", "mol", "features", "interactions", "sequences")
    extras: tuple[str,...] = ()
    deterministic_method = True
    order_invariant = False

    def __init__(
        self,
        *,
        direction: Literal["high_test", "low_test", "extremes_test"] = "high_test",
        task_index: int = 0,
        buffer: "float | int" = 0.0,
        tie_policy: Literal["by_index", "random", "keep_together"] = "keep_together",
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.direction = direction
        self.task_index = task_index
        self.buffer = buffer
        self.tie_policy = tie_policy
        if direction not in ("high_test", "low_test", "extremes_test"):
            raise ParameterError(f"invalid direction {direction!r}")

    @property
    def deterministic_without_seed(self) -> bool:  # type: ignore[override]
        return self.tie_policy != "random"

    def _check_preconditions(self, ctx: _Context) -> None:
        y = np.asarray(ctx.y)
        if y.ndim == 2 and self.task_index >= y.shape[1]:
            raise ParameterError(f"task_index={self.task_index} out of range for y with {y.shape[1]} columns")
        col = y if y.ndim == 1 else y[:, self.task_index]
        if not np.all(np.isfinite(col.astype(np.float64))):
            raise LabelError("LabelExtrapolationSplitter requires finite numeric y")

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        y = np.asarray(ctx.y, dtype=np.float64)
        v = y if y.ndim == 1 else y[:, self.task_index]
        rng = seed_for(ctx.rng_seeds, "label.tie", 0) if self.tie_policy == "random" else None
        test, overshoot = _cut_by_direction(ctx, v, ctx.sizes.n_test, self.direction, self.tie_policy, rng)

        discard = np.array([], dtype=np.int64)
        remaining = np.asarray([i for i in range(ctx.n) if i not in set(test.tolist())], dtype=np.int64)
        if self.buffer:
            test_vals = v[test]
            if test_vals.size:
                lo_bound, hi_bound = test_vals.min(), test_vals.max()
                if isinstance(self.buffer, float) and not isinstance(self.buffer, bool):
                    mask = (np.abs(v[remaining] - lo_bound) < self.buffer) | (
                        np.abs(v[remaining] - hi_bound) < self.buffer
                    )
                    discard = remaining[mask]
                    remaining = remaining[~mask]
                else:
                    dist = np.minimum(np.abs(v[remaining] - lo_bound), np.abs(v[remaining] - hi_bound))
                    order_by_dist = stable_sort(list(range(remaining.size)), key=lambda j: dist[j])
                    k = min(int(self.buffer), remaining.size)
                    discard = remaining[order_by_dist[:k]]
                    remaining = remaining[order_by_dist[k:]]
            if discard.size > 0.5 * ctx.n:
                raise ConstraintUnsatisfiableError(
                    f"buffer={self.buffer!r} removed {discard.size}/{ctx.n} records (> 50%)"
                )

        n_valid = ctx.sizes.n_valid
        if n_valid > 0 and remaining.size:
            test_vals = v[test]
            lo_bound, hi_bound = (test_vals.min(), test_vals.max()) if test.size else (0.0, 0.0)
            dist = np.minimum(np.abs(v[remaining] - lo_bound), np.abs(v[remaining] - hi_bound))
            order_by_dist = stable_sort(list(range(remaining.size)), key=lambda j: dist[j])
            valid = remaining[order_by_dist[:n_valid]]
            train = remaining[order_by_dist[n_valid:]]
        else:
            valid = np.array([], dtype=np.int64)
            train = remaining

        metadata = {
            "direction": self.direction,
            "task_index": self.task_index,
            "train_label_range": [float(v[train].min()), float(v[train].max())] if train.size else None,
            "test_label_range": [float(v[test].min()), float(v[test].max())] if test.size else None,
            "buffer_records": int(discard.size),
            "n_ties_at_boundary": int(overshoot),
        }
        return [_build_result(self, ctx, train=train, valid=valid, test=test, discard=discard, metadata=metadata)]


# ---------------------------------------------------------------------------
# StratifiedDistributionSplitter
# ---------------------------------------------------------------------------


def _quantile_bins(v: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1), method="linear")
    edges = np.unique(edges)
    return np.clip(np.digitize(v, edges[1:-1], right=True), 0, len(edges) - 2)


class StratifiedDistributionSplitter(BaseSplitter):
    """Match the full label *distribution* (not just class balance) between train and test (design
   .3, ``E.3``).

    Parameters
    ----------
    n_bins: int, default 20
    binning: {"quantile", "uniform", "kmeans"}, default "quantile"
    match: {"histogram", "moments", "ks"}, default "histogram"
        ``"histogram"``: per-bin apportionment (deterministic, always succeeds). ``"moments"``: a
        seeded local swap hill-climb minimising ``|Δmean| + |Δstd| + |Δskew|``. ``"ks"``: restart
        with different seeds until the two-sample KS statistic is ``<= max_ks``.
    max_ks: float, default 0.05
    max_restarts: int, default 20

    Advantages
    ----------
    - Cuts metric variance on small regression datasets more effectively than class-level stratification, since it matches shape rather than just balance.
    - Reports `ks_statistic`, so "the partitions share a label distribution" is evidenced, not assumed.
    - Deterministic in its default `histogram` mode.

    Pitfalls
    --------
    - Changes nothing about chemical leakage — it's `stratified_random` with more bins, and calling it a "distribution-matched split" invites more rigour than it actually has.
    - Matching the label distribution makes the test set *easier* by construction, removing exactly the label shift a real prospective test would contain.
    - `match="ks"` can loop to the restart limit on tied or censored labels.
    - The swap optimisation in `moments` mode is a heuristic, and failure to converge is only reported through `n_swaps_accepted`.

    """

    splitter_id = "stratified_distribution"
    family = "property"
    strictness = "optimistic"
    group_forming = False
    requires_labels = True
    accepts = ("smiles", "mol", "features", "interactions", "sequences")
    extras: tuple[str,...] = ()
    deterministic_method = True
    order_invariant = False
    deterministic_without_seed = False

    def __init__(
        self,
        *,
        n_bins: int = 20,
        binning: Literal["quantile", "uniform", "kmeans"] = "quantile",
        match: Literal["histogram", "moments", "ks"] = "histogram",
        max_ks: float = 0.05,
        max_restarts: int = 20,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.n_bins = n_bins
        self.binning = binning
        self.match = match
        self.max_ks = max_ks
        self.max_restarts = max_restarts

    def _stratified_split(self, ctx: _Context, v: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_bins = min(self.n_bins, len(np.unique(v)))
        if self.binning == "uniform":
            edges = np.linspace(v.min(), v.max(), n_bins + 1)
            strata = np.clip(np.digitize(v, edges[1:-1], right=True), 0, n_bins - 1)
        elif self.binning == "kmeans":
            from sklearn.cluster import KMeans

            labels = KMeans(n_clusters=max(1, n_bins), n_init=10, random_state=0).fit_predict(v.reshape(-1, 1))
            order_of_centroids = np.argsort([v[labels == k].mean() for k in np.unique(labels)])
            remap = {int(old): new for new, old in enumerate(order_of_centroids)}
            strata = np.array([remap[int(k)] for k in labels])
        else:
            strata = _quantile_bins(v, n_bins)

        perm = rng.permutation(ctx.n)
        train_list: list[int] = []
        valid_list: list[int] = []
        test_list: list[int] = []
        for s in sorted(set(strata.tolist())):
            members = perm[strata[perm] == s]
            m = len(members)
            n_tr = round(m * ctx.sizes.n_train / ctx.n)
            n_va = round(m * ctx.sizes.n_valid / ctx.n)
            n_tr = min(n_tr, m)
            n_va = min(n_va, m - n_tr)
            train_list.extend(members[:n_tr].tolist())
            valid_list.extend(members[n_tr: n_tr + n_va].tolist())
            test_list.extend(members[n_tr + n_va:].tolist())
        return (
            np.asarray(sorted(train_list), dtype=np.int64),
            np.asarray(sorted(valid_list), dtype=np.int64),
            np.asarray(sorted(test_list), dtype=np.int64),
        )

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        from scipy import stats as sp_stats

        y = np.asarray(ctx.y, dtype=np.float64)
        v = y if y.ndim == 1 else y[:, 0]
        rng = seed_for(ctx.rng_seeds, "stratdist.permutation", 0)
        train, valid, test = self._stratified_split(ctx, v, rng)
        ks_stat = float(sp_stats.ks_2samp(v[train], v[test]).statistic) if train.size and test.size else 0.0
        n_swaps = 0

        if self.match == "moments" and train.size and test.size:
            swap_rng = seed_for(ctx.rng_seeds, "stratdist.swap", 0)
            train, test, n_swaps = self._moment_match(v, train, test, swap_rng)
            ks_stat = float(sp_stats.ks_2samp(v[train], v[test]).statistic)
        elif self.match == "ks" and train.size and test.size:
            attempt = 0
            best = (ks_stat, train, valid, test)
            while best[0] > self.max_ks and attempt < self.max_restarts:
                attempt += 1
                rng2 = seed_for(ctx.rng_seeds, "stratdist.permutation", attempt)
                tr2, va2, te2 = self._stratified_split(ctx, v, rng2)
                stat2 = float(sp_stats.ks_2samp(v[tr2], v[te2]).statistic) if tr2.size and te2.size else 1.0
                if stat2 < best[0]:
                    best = (stat2, tr2, va2, te2)
            ks_stat, train, valid, test = best
            if ks_stat > self.max_ks:
                raise ConstraintUnsatisfiableError(
                    f"StratifiedDistributionSplitter: could not reach max_ks={self.max_ks} within "
                    f"{self.max_restarts} restarts; best achieved KS statistic was {ks_stat:.4f}"
                )

        mean_delta = float(abs(v[train].mean() - v[test].mean())) if train.size and test.size else 0.0
        std_delta = float(abs(v[train].std() - v[test].std())) if train.size and test.size else 0.0
        metadata = {
            "n_bins": self.n_bins,
            "match": self.match,
            "ks_statistic": ks_stat,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "n_swaps_accepted": n_swaps,
        }
        return [_build_result(self, ctx, train=train, valid=valid, test=test, discard=np.array([], dtype=np.int64), metadata=metadata)]

    @staticmethod
    def _moment_match(
        v: np.ndarray, train: np.ndarray, test: np.ndarray, rng: np.random.Generator, max_restarts: int = 20
    ) -> tuple[np.ndarray, np.ndarray, int]:
        from scipy import stats as sp_stats

        train = train.copy()
        test = test.copy()

        def cost(tr: np.ndarray, te: np.ndarray) -> float:
            return (
                abs(v[tr].mean() - v[te].mean())
                + abs(v[tr].std() - v[te].std())
                + abs(float(sp_stats.skew(v[tr])) - float(sp_stats.skew(v[te])))
            )

        n_accepted = 0
        budget = max_restarts * (train.size + test.size)
        cur_cost = cost(train, test)
        order = rng.permutation(min(train.size, test.size))
        for step in range(min(budget, len(order))):
            i_local = int(order[step % len(order)])
            if i_local >= train.size or i_local >= test.size:
                continue
            new_train = train.copy()
            new_test = test.copy()
            new_train[i_local], new_test[i_local] = test[i_local], train[i_local]
            new_cost = cost(new_train, new_test)
            if new_cost < cur_cost:
                train, test = new_train, new_test
                cur_cost = new_cost
                n_accepted += 1
        return np.sort(train), np.sort(test), n_accepted


# ---------------------------------------------------------------------------
# MOODSplitter
# ---------------------------------------------------------------------------


def _distance_stat(F: np.ndarray, reference: np.ndarray, stat: str, knn_k: int, metric: str) -> np.ndarray:
    from chemsplit.metrics import pairwise_distances

    D = pairwise_distances(F, reference, metric=metric)
    if stat == "nn":
        return D.min(axis=1)
    if stat == "knn_mean":
        k = min(knn_k, D.shape[1])
        part = np.partition(D, k - 1, axis=1)[:,:k]
        return part.mean(axis=1)
    # "centroid"
    centroid = reference.mean(axis=0, keepdims=True) if not hasattr(reference, "toarray") else reference.toarray().mean(axis=0, keepdims=True)
    return pairwise_distances(F if not hasattr(F, "toarray") else F.toarray(), centroid, metric="euclidean").ravel()


def _discrepancy(obs: np.ndarray, target: np.ndarray, kind: str, n_bins: int) -> float:
    from scipy import stats as sp_stats

    if kind == "wasserstein":
        return float(sp_stats.wasserstein_distance(obs, target))
    if kind == "ks":
        return float(sp_stats.ks_2samp(obs, target).statistic)
    # "js" Jensen-Shannon over shared equal-width bins on [0,1] (distances are non-negative;
    # rescale both samples into [0,1] using their joint range for a shared histogram support)
    lo = min(obs.min(initial=0.0), target.min(initial=0.0))
    hi = max(obs.max(initial=1.0), target.max(initial=1.0))
    hi = hi if hi > lo else lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(obs, bins=edges, density=True)
    q, _ = np.histogram(target, bins=edges, density=True)
    p = p / max(p.sum(), EPS)
    q = q / max(q.sum(), EPS)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + EPS) / (b[mask] + EPS))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


class MOODSplitter(SimilarityParamsMixin, BaseSplitter):
    """Select, among candidate splitters, the one whose train→test distance distribution best
    matches the train→**deployment** distance distribution (MOOD: "Massive
    Out-Of-Distribution shift" splitter).

    Parameters
    ----------
    candidates: sequence of BaseSplitter
        Already-instantiated candidate splitters — string IDs are not accepted; a
        ``ParameterError`` names this explicitly.
    deployment_set: array-like
        The library you actually intend to screen. Required.
    distance_stat: {"nn", "knn_mean", "centroid"}, default "nn"
    discrepancy: {"wasserstein", "ks", "js"}, default "wasserstein"

    Advantages
    ----------
    - Reframes "which split is hardest?" as "which split is *representative* of my deployment?" — the only version of the question with a defensible answer.
    - Produces an auditable table of candidate scores, so the choice is evidence rather than taste.
    - Automatically falls back to a *random* split when that's genuinely appropriate, which no difficulty-ranking heuristic would do.

    Pitfalls
    --------
    - Requires the deployment library up front; without it the method is undefined, and a guessed deployment set silently decides the answer.
    - The selected split is chosen using a statistic computed from the data, so the reported score is mildly optimistic in a model-selection sense — the honest protocol selects the split on one dataset and reports on another, or discloses the selection.
    - Distance-distribution matching is a one-dimensional summary — two very different splits can produce identical NN-distance distributions.
    - Running five candidate splitters costs five splits, which is expensive on `O(n²)` candidates.
    - If the deployment set overlaps the training data, MOOD correctly picks a random split, which readers unfamiliar with the method may mistake for a weak evaluation.

    """

    splitter_id = "mood"
    family = "property"
    strictness = "strict"
    group_forming = False
    requires_labels = False
    accepts = ("smiles", "mol", "features")
    extras: tuple[str,...] = ()
    deterministic_method = True
    order_invariant = False
    deterministic_without_seed = False

    def __init__(
        self,
        *,
        candidates: "Sequence[BaseSplitter]" = (),
        deployment_set: Any = None,
        distance_stat: Literal["nn", "knn_mean", "centroid"] = "nn",
        knn_k: int = 5,
        discrepancy: Literal["wasserstein", "ks", "js"] = "wasserstein",
        n_bins: int = 50,
        return_all: bool = True,
        featurizer: Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        **base: Any,
    ) -> None:
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=n_jobs
        )
        BaseSplitter.__init__(self, **base)
        self.candidates = tuple(candidates)
        self.deployment_set = deployment_set
        self.distance_stat = distance_stat
        self.knn_k = knn_k
        self.discrepancy = discrepancy
        self.n_bins = n_bins
        self.return_all = return_all
        self._validate_similarity_params()
        for c in self.candidates:
            if not isinstance(c, BaseSplitter):
                raise ParameterError(
                    "MOODSplitter.candidates: string splitter IDs are not resolvable yet "
                    "(chemsplit.registry does not exist); pass instantiated BaseSplitter objects"
                )
        if deployment_set is None:
            raise ConfigurationError("MOODSplitter is undefined without deployment_set")
        if len(deployment_set) == 0:
            raise ParameterError("deployment_set must be non-empty")

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        from chemsplit._fp_similarity import resolve_featurizer

        featurizer = resolve_featurizer(self.featurizer)
        F_data = ctx.get_features(featurizer)
        if ctx.mols is not None:
            from rdkit import Chem

            deploy_mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else s for s in self.deployment_set]
            F_deploy = featurizer.transform(deploy_mols)
        else:
            F_deploy = np.asarray(self.deployment_set)

        target = _distance_stat(F_deploy, F_data, self.distance_stat, self.knn_k, self.metric)

        results = []
        errors: dict[str, str] = {}
        scores: list[list[Any]] = []
        for i, cand in enumerate(self.candidates):
            cid = getattr(cand, "splitter_id", type(cand).__name__)
            try:
                r = cand.split_result(ctx.smiles if ctx.mols is None else ctx.mols, ctx.y)[0]
                obs = _distance_stat(F_data[r.test], F_data[r.train], self.distance_stat, self.knn_k, self.metric)
                score = _discrepancy(obs, target, self.discrepancy, self.n_bins)
                results.append((cid, score, r))
                scores.append([cid, score])
            except Exception as exc:  # noqa: BLE001 - candidate isolation is intentional
                errors[cid] = str(exc)
                scores.append([cid, None])

        if not results:
            if errors:
                raise ConstraintUnsatisfiableError(
                    f"MOODSplitter: every candidate failed; first error: {next(iter(errors.values()))}"
                )
            raise ConstraintUnsatisfiableError("MOODSplitter: every candidate failed")

        best_id, best_score, best_result = argmin_tiebreak(lambda t: t[1], results)
        metadata = {
            "selected": best_id,
            "candidate_scores": scores,
            "candidate_errors": errors,
            "discrepancy": self.discrepancy,
            **best_result.metadata,
        }
        return [
            _build_result(
                self,
                ctx,
                train=best_result.train,
                valid=best_result.valid,
                test=best_result.test,
                discard=best_result.discard,
                metadata=metadata,
            )
        ]


# ---------------------------------------------------------------------------
# AdversarialSplitter
# ---------------------------------------------------------------------------


class AdversarialSplitter(SimilarityParamsMixin, BaseSplitter):
    """Use a train-vs-test discriminator either to *audit* an existing split or to *construct* a
    target covariate shift.

    Parameters
    ----------
    mode: {"audit", "construct"}, default "construct"
    target_auc: float, default 0.75
    base_splitter: BaseSplitter, default a fresh random split
    classifier: {"logreg", "gbdt"}, default "logreg"
    swap_frac: float, default 0.05

    Advantages
    ----------
    - `mode="audit"` is the cheapest possible check that a split is what it claims — one comparable number across splitters, datasets, and papers, worth attaching to every reported split.
    - `mode="construct"` lets you dial covariate shift to a chosen level and measure degradation as a function of shift, instead of arguing over which named split is "realistic".
    - `top_discriminative_features` names the bits or descriptors separating the partitions, often revealing an unintended confound — a salt, a project, a vendor.

    Pitfalls
    --------
    - A constructed split is optimised against a specific discriminator on specific features — a different model may see no shift at all, so the number isn't a property of the data alone.
    - Pushing AUC up tends to surface *trivial* separations first — molecular size, a common substructure, a fingerprint density artefact — rather than chemically interesting shift.
    - In `audit` mode a high AUC only shows the partitions differ, not whether the split is good or bad; interpreting it still requires knowing what shift you wanted.
    - The discriminator fits on the same features used to build the split, making `construct` mode circular in the same way as `latent_space`.
    - Convergence isn't guaranteed and is only reported, not enforced — a non-converged run shouldn't be described as a "target_auc = 0.75 split".

    """

    splitter_id = "adversarial"
    family = "property"
    strictness = "strict"
    group_forming = False
    requires_labels = False
    accepts = ("smiles", "mol", "features")
    extras: tuple[str,...] = ()
    deterministic_method = True
    order_invariant = False
    deterministic_without_seed = False

    def __init__(
        self,
        *,
        mode: Literal["audit", "construct"] = "construct",
        target_auc: float = 0.75,
        base_splitter: "BaseSplitter | None" = None,
        classifier: Literal["logreg", "gbdt"] = "logreg",
        cv: int = 5,
        max_iter: int = 50,
        swap_frac: float = 0.05,
        tolerance: float = 0.02,
        featurizer: Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        **base: Any,
    ) -> None:
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=n_jobs
        )
        BaseSplitter.__init__(self, **base)
        self.mode = mode
        self.target_auc = target_auc
        self.base_splitter = base_splitter
        self.classifier = classifier
        self.cv = cv
        self.max_iter = max_iter
        self.swap_frac = swap_frac
        self.tolerance = tolerance
        self._validate_similarity_params()
        if target_auc < 0.5:
            raise ParameterError(f"target_auc must be >= 0.5, got {target_auc!r}")
        if base_splitter is not None and not isinstance(base_splitter, BaseSplitter):
            raise ParameterError("base_splitter must be a BaseSplitter instance (or None)")

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        from chemsplit._fp_similarity import resolve_featurizer

        base = self.base_splitter
        if base is None:
            from chemsplit.splitters.baseline import RandomSplitter

            base = RandomSplitter(
                train_size=self.train_size, test_size=self.test_size, random_state=self.random_state
            )
        r = base.split_result(ctx.smiles if ctx.mols is None else ctx.mols, ctx.y)[0]
        train, test = r.train.copy(), r.test.copy()

        featurizer = resolve_featurizer(self.featurizer)
        F = ctx.get_features(featurizer)
        F_dense = F.toarray() if hasattr(F, "toarray") else np.asarray(F)

        auc_trace: list[float] = []
        it = 0
        auc = 0.5
        converged = False
        for it in range(self.max_iter):
            labels = np.zeros(ctx.n, dtype=np.int64)
            labels[test] = 1
            used = np.concatenate([train, test])
            seed = int(seed_for(ctx.rng_seeds, "adv.cv", it).integers(0, 2**31 - 1))
            clf = (
                LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=1000, random_state=seed)
                if self.classifier == "logreg"
                else HistGradientBoostingClassifier(max_depth=6, max_iter=200, learning_rate=0.1, random_state=seed)
            )
            cv = min(self.cv, int(np.bincount(labels[used]).min()) if len(np.unique(labels[used])) > 1 else 1)
            cv = max(cv, 2)
            try:
                scores = cross_val_score(clf, F_dense[used], labels[used], cv=cv, scoring="roc_auc")
                auc = float(scores.mean())
            except Exception:
                auc = 0.5
            auc_trace.append(auc)
            if self.mode == "audit" or abs(auc - self.target_auc) <= self.tolerance:
                converged = self.mode == "construct"
                break
            clf.fit(F_dense[used], labels[used])
            p = clf.predict_proba(F_dense)[:, 1]
            k = max(1, round(self.swap_frac * min(len(train), len(test))))
            if auc < self.target_auc:
                out_of_train = sorted(train, key=lambda i: (-p[i], i))[:k]
                out_of_test = sorted(test, key=lambda i: (p[i], i))[:k]
            else:
                out_of_train = sorted(train, key=lambda i: (p[i], i))[:k]
                out_of_test = sorted(test, key=lambda i: (-p[i], i))[:k]
            train = np.asarray(sorted(set(train.tolist()) - set(out_of_train) | set(out_of_test)), dtype=np.int64)
            test = np.asarray(sorted(set(test.tolist()) - set(out_of_test) | set(out_of_train)), dtype=np.int64)
            if self.mode == "audit":
                break

        metadata = {
            "mode": self.mode,
            "final_auc": auc,
            "auc_trace": auc_trace,
            "target_auc": self.target_auc,
            "converged": converged if self.mode == "construct" else True,
            "iterations": it + 1,
            "classifier": self.classifier,
        }
        if self.mode == "construct" and not converged:
            from chemsplit.exceptions import SizeToleranceWarning, warn_with_details

            warn_with_details(SizeToleranceWarning(f"AdversarialSplitter did not converge to target_auc within {self.max_iter} iterations", details=metadata))
        return [_build_result(self, ctx, train=train, valid=r.valid, test=test, discard=r.discard, metadata=metadata)]
