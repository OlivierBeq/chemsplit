"""Protocol wrapper splitters."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Literal

import numpy as np
import scipy.sparse as sp

from chemsplit.base import (
    BaseSplitter,
    GroupSplitter,
    SplitResult,
    Strictness,
    _Context,
    assign_groups_kfold,
)
from chemsplit.determinism import seed_for
from chemsplit.exceptions import ConfigurationError, ParameterError

__all__ = [
    "ApplicabilityDomainSplitter",
    "ExternalHoldoutSplitter",
    "GroupKFoldSplitter",
    "IntersectionSplitter",
    "NestedCVSplitter",
    "RepeatedSplitter",
    "ThreeWaySplitter",
]

_ACCEPTS: tuple[str,...] = ("smiles", "mol", "features", "interactions", "sequences")




def _resolve_splitter(design: str | BaseSplitter | None, param_name: str) -> BaseSplitter:
    """Resolve a constructor parameter that names another splitter, either as an already-
    instantiated object or as a registry id/class-name string."""
    if design is None:
        raise ParameterError(
            f"this splitter requires {param_name}= (a splitter instance, or a registry "
            "id/class-name string)"
        )
    if isinstance(design, str):
        from chemsplit.registry import get_splitter  # local: avoids a module-scope import cycle

        return get_splitter(design)
    return design


def _clone_with_overrides(splitter: BaseSplitter, **overrides: Any) -> BaseSplitter:
    """Reconstruct ``splitter`` with some constructor params overridden -- used to give a wrapped
    splitter fresh sizes and/or a derived seed without mutating the caller's original instance."""
    params = dict(splitter.get_params())
    params.update(overrides)
    return type(splitter)(**params)


def _select_ctx_X(ctx: _Context) -> Any:
    """The raw-ish X a wrapped splitter's own ``split_result`` can be called with (it re-runs its
    own preprocessing, so passing SMILES/mols/features straight through is correct, if not free)."""
    if ctx.smiles is not None:
        return ctx.smiles
    if ctx.mols is not None:
        return ctx.mols
    if ctx.raw_features is not None:
        return ctx.raw_features
    raise ConfigurationError(
        "this protocol wrapper needs smiles-, mol-, or features-kind input to delegate to its "
        "wrapped splitter"
    )


def _index_X(X: Any, idx: np.ndarray) -> Any:
    if isinstance(X, np.ndarray) or sp.issparse(X):
        return X[idx]
    return [X[i] for i in idx.tolist()]


def _build_result(
    splitter: BaseSplitter,
    ctx: _Context,
    *,
    train: np.ndarray,
    valid: np.ndarray,
    test: np.ndarray,
    discard: np.ndarray,
    n_records: int | None = None,
    metadata: dict[str, Any],
) -> SplitResult:
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
        params=splitter.get_params(),
        n_records=ctx.n if n_records is None else n_records,
        metadata=full_metadata,
    )


class GroupKFoldSplitter(GroupSplitter):
    """K-fold cross-validation whose folds respect a grouping, so no group straddles a fold
    boundary.

    :param grouper: supplies the group labels, as a :class:`~chemsplit.base.GroupSplitter`
        instance or a registry id. Ignored when ``groups=`` is passed to ``split()`` directly.
    :param n_splits: number of folds, or ``"auto"`` for one fold per group, i.e.
        leave-one-group-out.
    :param size_tolerance: how far a realised fold size may drift from its target before a
        :class:`SizeToleranceWarning` is issued.
    :param group_assignment: unused; the k-fold path always packs groups largest-first.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`. The size parameters must be
        left ``None``, since ``n_splits`` determines the fold sizes.
    :raises ConfigurationError: if any size parameter is set, or neither ``grouper`` nor
        ``groups`` is available.
    :raises ParameterError: if ``n_splits`` is neither an int above 1 nor ``"auto"``, or
        ``grouper`` resolves to something that is not group-forming.

    Notes
    -----
    Delegates to :func:`chemsplit.base.assign_groups_kfold` directly rather than the
    inherited ``GroupSplitter._partition``, because resolving ``n_splits="auto"`` requires knowing
    the group count before dispatch, which the inherited path does not support.

    Advantages
    ----------
    - Turns any grouping -- scaffold, cluster, assay, series, party -- into a full
      cross-validation protocol, testing every record exactly once under group-level holdout.
    - Greedy largest-first bin packing keeps folds much closer to equal size than random group
      assignment, which matters when one group holds a large share of the data.
    - Accepts externally computed group labels, so a grouping from any other tool composes
      cleanly.

    Pitfalls
    --------
    - Fold sizes stay unequal whenever group sizes are, so per-fold metrics come from different
      sample sizes. The fold sizes belong next to the scores.
    - The grouping determines everything: over Murcko scaffolds this inherits every weakness
      of `murcko_scaffold`, and "grouped" does not make a criterion strong.
    - Greedy bin packing is deterministic but not optimal, so an adversarial group-size
      distribution can still yield a badly skewed fold.
    - With few groups, `n_splits` is capped and the folds end up large and highly correlated.

    References
    ----------
    .. [1] Roberts, D. R.; Bahn, V.; Ciuti, S. et al. Cross-Validation Strategies for Data
       with Temporal, Spatial, Hierarchical, or Phylogenetic Structure. *Ecography* **2017**,
       40 (8), 913-929. https://doi.org/10.1111/ecog.02881
    .. [2] Hastie, T.; Tibshirani, R.; Friedman, J. *The Elements of Statistical Learning*, 2nd ed.;
       Springer: New York, **2009**; Chapter 7. https://doi.org/10.1007/978-0-387-84858-7
    """

    splitter_id: ClassVar[str] = "group_k_fold"
    family: ClassVar[str] = "protocol"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = _ACCEPTS
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        grouper: str | GroupSplitter | None = None,
        n_splits: int | Literal['auto'] = 5,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        **base: Any,
    ) -> None:
        super().__init__(
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            **base,
        )
        self.grouper = grouper
        if self.train_size is not None or self.valid_size is not None or self.test_size is not None:
            raise ConfigurationError(
                "GroupKFoldSplitter: fold sizes are determined by n_splits, not "
                "train_size/valid_size/test_size (leave them at their default None)"
            )

    def _group_labels(self, ctx: _Context) -> np.ndarray:
        from chemsplit._unionfind import dense_label_encode

        if ctx.groups_in is not None:
            return dense_label_encode(ctx.groups_in.tolist())
        grouper_template = _resolve_splitter(self.grouper, "grouper")
        # Re-seed after resolving: a resolved grouper is otherwise outside this splitter's
        # random_state. Only matters for a seed-needing grouper.
        grouper_seed = int(
            seed_for(ctx.rng_seeds, "groupkfold.grouper_seed", 0).integers(0, 2**31 - 1)
        )
        grouper = _clone_with_overrides(grouper_template, random_state=grouper_seed)
        sub_X = _select_ctx_X(ctx)
        return np.asarray(grouper.compute_groups(sub_X, ctx.y), dtype=np.int64)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Report how many splits will be yielded.

        :param X: the records, needed to resolve ``n_splits="auto"``.
        :param y: ignored.
        :param groups: precomputed group labels, which decide the fold count under ``"auto"``.
        :return: ``n_splits``, or the group count when it is ``"auto"``.
        """
        if self.n_splits != "auto":
            return int(self.n_splits)
        if groups is not None:
            return len(set(int(g) for g in groups))
        if X is None:
            return 1
        grouper = _resolve_splitter(self.grouper, "grouper")
        return len(set(int(g) for g in grouper.compute_groups(X, y)))

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        labels = self._group_labels(ctx)
        n_groups = len(set(int(g) for g in labels))
        n_splits = n_groups if self.n_splits == "auto" else int(self.n_splits)
        if n_splits < 2:
            raise ParameterError(f"n_splits must resolve to >= 2, got {n_splits}")
        if n_splits > n_groups:
            raise ParameterError(
                f"n_splits ({n_splits}) exceeds the number of groups ({n_groups})"
            )
        rng = seed_for(ctx.rng_seeds, "group.assign", 0)
        folds = assign_groups_kfold(labels, n_splits, rng)
        results = []
        for k in range(n_splits):
            test = folds[k]
            train = (
                np.sort(np.concatenate([folds[j] for j in range(n_splits) if j != k]))
                if n_splits > 1
                else np.array([], dtype=np.int64)
            )
            results.append(
                self._build_result(
                    ctx,
                    train=train,
                    valid=np.array([], dtype=np.int64),
                    test=test,
                    discard=np.array([], dtype=np.int64),
                    groups=labels,
                    extra_metadata={"fold_index": k, "n_splits": n_splits},
                )
            )
        return results


class ThreeWaySplitter(BaseSplitter):
    """Apply a wrapped splitter's criterion at BOTH boundaries: train<->valid and
    (train union valid)<->test.

    :param base_splitter: the splitter whose criterion carves each boundary, as an instance or
        a registry id. It runs once to cut train-plus-valid from test, then again on that
        subset to cut train from valid.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`. ``valid_size`` must be
        non-empty; at its default this degenerates to a two-way split.
    :raises ParameterError: if ``base_splitter`` is neither ``None``, a string nor a
        :class:`~chemsplit.base.BaseSplitter`.

    Notes
    -----
    Stage 1: run ``base_splitter`` sized ``train_size=n_train+n_valid, test_size=n_test`` to get
    (train_plus_valid, test). Stage 2: run a fresh clone of ``base_splitter`` (a different derived
    seed) on JUST the train_plus_valid subset, sized ``train_size=n_train, test_size=n_valid``,
    then map its local train/test indices back into the ORIGINAL index space via
    ``train_plus_valid_global[local_idx]``. Both stages' discards are unioned.

    Advantages
    ----------
    - Closes the leak where a scaffold-split test set sits behind a randomly split validation
      set: an easy validation set otherwise selects a model optimised for interpolation.
    - Applying the same criterion at both boundaries makes the validation score an honest, if
      optimistic, preview of the test score.
    - Exact index remapping lets the wrapper compose with every splitter without special cases.

    Pitfalls
    --------
    - A structurally split validation set is smaller and harder, so early stopping triggers
      sooner and hyperparameter choices get noisier. That is the price of not over-tuning.
    - ``valid_size`` counts records in the whole dataset, but stage 2 carves valid from the
      train-plus-valid subset, so a group-forming criterion drifts it;
      ``metadata["realised_sizes"]`` holds what was achieved.
    - Applying a group-forming criterion twice compounds that size drift.

    References
    ----------
    .. [1] Hastie, T.; Tibshirani, R.; Friedman, J. *The Elements of Statistical Learning*, 2nd ed.;
       Springer: New York, **2009**; Chapter 7. https://doi.org/10.1007/978-0-387-84858-7
    .. [2] Tropsha, A. Best Practices for QSAR Model Development, Validation, and Exploitation.
       *Mol. Inf.* **2010**, 29 (6-7), 476-488. https://doi.org/10.1002/minf.201000061
    """

    splitter_id: ClassVar[str] = "three_way"
    family: ClassVar[str] = "protocol"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    group_forming: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = _ACCEPTS
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(self, *, base_splitter: str | BaseSplitter | None = None, **base: Any) -> None:
        super().__init__(**base)
        self.base_splitter = base_splitter
        if base_splitter is None:
            raise ParameterError("ThreeWaySplitter requires base_splitter=")

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        outer_template = _resolve_splitter(self.base_splitter, "base_splitter")
        sub_X = _select_ctx_X(ctx)

        n_tv = ctx.sizes.n_train + ctx.sizes.n_valid
        seed1 = int(seed_for(ctx.rng_seeds, "threeway.outer_seed", 0).integers(0, 2**31 - 1))
        stage1 = _clone_with_overrides(
            outer_template,
            train_size=n_tv,
            valid_size=None,
            test_size=ctx.sizes.n_test,
            random_state=seed1,
        )
        r1 = list(stage1.split_result(sub_X, ctx.y))[0]
        train_plus_valid_global = r1.train
        test_global = r1.test
        discard_global = list(r1.discard)

        if ctx.sizes.n_valid > 0 and train_plus_valid_global.size > 1:
            local_X = _index_X(sub_X, train_plus_valid_global)
            local_y = ctx.y[train_plus_valid_global] if ctx.y is not None else None
            seed2 = int(seed_for(ctx.rng_seeds, "threeway.inner_seed", 0).integers(0, 2**31 - 1))
            stage2 = _clone_with_overrides(
                outer_template,
                train_size=ctx.sizes.n_train,
                valid_size=None,
                test_size=ctx.sizes.n_valid,
                random_state=seed2,
            )
            r2 = list(stage2.split_result(local_X, local_y))[0]
            train_global = train_plus_valid_global[r2.train]
            valid_global = train_plus_valid_global[r2.test]
            discard_global.extend(train_plus_valid_global[r2.discard].tolist())
        else:
            train_global = train_plus_valid_global
            valid_global = np.array([], dtype=np.int64)

        return [
            _build_result(
                self,
                ctx,
                train=train_global,
                valid=valid_global,
                test=test_global,
                discard=np.asarray(discard_global, dtype=np.int64),
                metadata={"stage_splitter": type(outer_template).__name__},
            )
        ]


class RepeatedSplitter(BaseSplitter):
    """Repeat a wrapped splitter under multiple derived seeds, for split-induced variance
    estimation.

    :param base_splitter: the splitter to repeat, as an instance or a registry id.
    :param n_repeats: how many repeats to run. Each derives its own ``random_state``, so
        repeats stay decorrelated even when ``base_splitter`` carries a fixed seed.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``n_repeats`` is below 1, or ``base_splitter`` is neither
        ``None``, a string nor a :class:`~chemsplit.base.BaseSplitter`.

    Advantages
    ----------
    - Between-split variance frequently exceeds between-model differences on chemical data, so
      a model comparison without repeats is not evidence. This is the cheapest way to get it.
    - Each repeat's seed is derived independently, so the repeats stay decorrelated even when
      ``base_splitter`` carries a fixed ``random_state``.
    - Wraps any splitter without knowing anything about it, and tags every result with
      ``metadata["repeat_index"]`` so repeats can be grouped downstream.

    Pitfalls
    --------
    - Repeating a *group-forming* splitter under `group_assignment="greedy_desc"` gives
      identical results every time, with no warning; only `"random"` assignment or a seeded
      clusterer varies.
    - The spread across repeats measures split variance, not model uncertainty, so it is not a
      model confidence interval.
    - Cost multiplies directly, so `O(n^2)` splitters get expensive fast.

    References
    ----------
    .. [1] Krstajic, D.; Buturovic, L. J.; Leahy, D. E.; Thomas, S. Cross-Validation Pitfalls When
       Selecting and Assessing Regression and Classification Models. *J. Cheminform.* **2014**,
       6 (1), 10. https://doi.org/10.1186/1758-2946-6-10
    .. [2] Picard, R. R.; Cook, R. D. Cross-Validation of Regression Models. *J. Am. Stat. Assoc.*
       **1984**, 79 (387), 575-583. https://doi.org/10.1080/01621459.1984.10478083
    """

    splitter_id: ClassVar[str] = "repeated"
    family: ClassVar[str] = "protocol"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    group_forming: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = _ACCEPTS
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        base_splitter: str | BaseSplitter | None = None,
        n_repeats: int = 10,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.base_splitter = base_splitter
        self.n_repeats = n_repeats
        if base_splitter is None:
            raise ParameterError("RepeatedSplitter requires base_splitter=")
        if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 1:
            raise ParameterError(f"n_repeats must be a positive int, got {n_repeats!r}")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Report how many splits will be yielded.

        :param X: ignored, as are ``y`` and ``groups``; the signature is sklearn\'s.
        :return: ``n_repeats`` times the wrapped splitter's own split count.
        """
        return int(self.n_repeats)

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        template = _resolve_splitter(self.base_splitter, "base_splitter")
        sub_X = _select_ctx_X(ctx)
        results: list[SplitResult] = []
        for k in range(self.n_repeats):
            seed_k = int(
                seed_for(ctx.rng_seeds, "repeated.base_seed", k).integers(0, 2**31 - 1)
            )
            cloned = _clone_with_overrides(template, random_state=seed_k)
            for r in cloned.split_result(sub_X, ctx.y):
                metadata = {**r.metadata, "repeat_index": k}
                results.append(
                    dataclasses.replace(
                        r,
                        splitter_id=self.splitter_id,
                        params=self.get_params(),
                        n_records=ctx.n,
                        metadata=metadata,
                    )
                )
        return results


class NestedCVSplitter(BaseSplitter):
    """Nested cross-validation: for each outer fold's training set, run an inner splitter to
    produce inner train/valid folds -- the sanctioned way to combine hyperparameter selection with
    an honest outer test score.

    :param outer_splitter: determines the outer train/test folds, as an instance or a registry
        id.
    :param inner_splitter: runs on each outer fold's train subset to produce inner train/valid
        pairs. A fresh, independently seeded clone runs per outer fold.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if either splitter is neither ``None``, a string nor a
        :class:`~chemsplit.base.BaseSplitter`.

    Notes
    -----
    Inner results are re-indexed from the outer-train-subset's local index space back into the
    ORIGINAL global index space before being wrapped in a `SplitResult` (``inner.test`` becomes
    this wrapper's ``valid``; the outer fold's own ``test`` passes through unchanged as this
    wrapper's ``test``).

    Advantages
    ----------
    - The only honest protocol when hyperparameters are tuned at all, since no tuning decision
      ever sees an outer test fold.
    - Each outer fold gets an independently seeded clone of ``inner_splitter``, and every
      result carries ``metadata["outer_fold"]``, so inner folds stay attributable.
    - Composes any outer criterion with any inner criterion, so the tuning distribution can
      match the evaluation distribution.

    Pitfalls
    --------
    - Expensive: `outer x (1 + inner)` splits and their model fits.
    - Skipping it and reporting the best inner-loop score is the most common silent source of
      optimism in QSAR papers, and it is invisible in the published numbers.
    - Re-clustering inside each outer fold changes the grouping, so inner and outer criteria
      differ even under one class, and the result does not record which a fold used.
    - Estimates the performance of the *whole tuning procedure* rather than one chosen model.
      The final model is refit on all the data and cannot inherit the nested estimate.


    References
    ----------
    .. [1] Varma, S.; Simon, R. Bias in Error Estimation When Using Cross-Validation for
       Model Selection.
       *BMC Bioinformatics* **2006**, 7, 91. https://doi.org/10.1186/1471-2105-7-91
    .. [2] Cawley, G. C.; Talbot, N. L. C. On Over-Fitting in Model Selection and Subsequent
       Selection
       Bias in Performance Evaluation. *J. Mach. Learn. Res.* **2010**, 11, 2079-2107.
       https://www.jmlr.org/papers/v11/cawley10a.html
    """

    splitter_id: ClassVar[str] = "nested_cv"
    family: ClassVar[str] = "protocol"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    group_forming: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = _ACCEPTS
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        outer_splitter: str | BaseSplitter | None = None,
        inner_splitter: str | BaseSplitter | None = None,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.outer_splitter = outer_splitter
        self.inner_splitter = inner_splitter
        if outer_splitter is None or inner_splitter is None:
            raise ParameterError(
                "NestedCVSplitter requires both outer_splitter= and inner_splitter="
            )

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Report how many splits will be yielded.

        :param X: ignored, as are ``y`` and ``groups``; the signature is sklearn\'s.
        :return: the outer fold count times the inner fold count.
        """
        outer = _resolve_splitter(self.outer_splitter, "outer_splitter")
        return outer.get_n_splits(X, y, groups)

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        outer_template = _resolve_splitter(self.outer_splitter, "outer_splitter")
        inner_template = _resolve_splitter(self.inner_splitter, "inner_splitter")
        sub_X = _select_ctx_X(ctx)

        # Re-seed the resolved outer splitter from this splitter's rng_seeds. _resolve_splitter
        # derives no seed of its own, and get_splitter() defaults to OS entropy, so without
        # this the same fixture and seed give a different outer split on every run.
        outer_seed = int(seed_for(ctx.rng_seeds, "nestedcv.outer_seed", 0).integers(0, 2**31 - 1))
        outer = _clone_with_overrides(outer_template, random_state=outer_seed)

        results: list[SplitResult] = []
        outer_results = outer.split_result(sub_X, ctx.y)
        for fold_idx, r in enumerate(outer_results):
            outer_train_global = r.train
            local_X = _index_X(sub_X, outer_train_global)
            local_y = ctx.y[outer_train_global] if ctx.y is not None else None
            seed_k = int(
                seed_for(ctx.rng_seeds, "nestedcv.inner_seed", fold_idx).integers(0, 2**31 - 1)
            )
            inner = _clone_with_overrides(inner_template, random_state=seed_k)
            for inner_r in inner.split_result(local_X, local_y):
                inner_train_global = outer_train_global[inner_r.train]
                inner_valid_global = outer_train_global[inner_r.test]
                discard_global = np.concatenate(
                    [r.discard, outer_train_global[inner_r.discard]]
                )
                metadata = {
                    "realised_sizes": {
                        "train": int(inner_train_global.size),
                        "valid": int(inner_valid_global.size),
                        "test": int(r.test.size),
                    },
                    "outer_fold": fold_idx,
                }
                results.append(
                    SplitResult(
                        train=np.sort(inner_train_global),
                        valid=np.sort(inner_valid_global),
                        test=np.sort(r.test),
                        discard=np.sort(discard_global),
                        groups=None,
                        splitter_id=self.splitter_id,
                        params=self.get_params(),
                        n_records=ctx.n,
                        metadata=metadata,
                    )
                )
        return results


class ExternalHoldoutSplitter(BaseSplitter):
    """Force a caller-supplied external dataset to be the entire test partition.

    :param X_external: the external holdout set, in the same form as the main ``X``. It is
        appended after the main dataset, so its records take global indices ``n`` to ``n+m-1``.
    :param y_external: labels for the external set. Not used by the split itself; carried
        through for downstream scoring.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`. The size parameters are
        unused, since every original record is train and every external record is test.
    :raises ConfigurationError: if ``X_external`` is missing.
    :raises InputError: at split time, if ``y_external`` is given at a length that does not
        match ``X_external``.

    Notes
    -----
    Unlike every other splitter, the returned ``n_records`` is ``n + m`` (main plus external), not
    ``n`` -- since the external records are real records that must appear somewhere in the
    ``SplitResult``, which must cover every record exactly once.

    Advantages
    ----------
    - The one evaluation nobody can accidentally tune against, since the data was assembled
      independently: a later campaign, another lab, a different vendor.
    - Overlap checking is on by default, which closes the step most often skipped when an
      "external" set quietly contains training compounds.
    - `max_similarity_to_train` turns "external" from a claim into a measurement.

    Pitfalls
    --------
    - "External" describes provenance, not chemistry. A set drawn from the same vendor
      catalogue is not external in any useful sense, which `max_similarity_to_train` shows.
    - Index semantics change: results index into the concatenated array. `external_offset`
      and the `source` column exist so callers do not mis-align labels.
    - A single external set is one sample of one distribution, so a good score is evidence
      rather than proof.
    - Overlap removal biases the training set by dropping exactly the compounds most similar to
      the evaluation set.


    References
    ----------
    .. [1] Golbraikh, A.; Tropsha, A. Beware of q2! *J. Mol. Graph. Model.* **2002**, 20 (4),
       269-276.
       https://doi.org/10.1016/S1093-3263(01)00123-1
    .. [2] Tropsha, A. Best Practices for QSAR Model Development, Validation, and Exploitation.
       *Mol. Inf.* **2010**, 29 (6-7), 476-488. https://doi.org/10.1002/minf.201000061
    """

    splitter_id: ClassVar[str] = "external_holdout"
    family: ClassVar[str] = "protocol"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    group_forming: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = True
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        X_external: Any = None,
        y_external: Any = None,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.X_external = X_external
        self.y_external = y_external
        if X_external is None or len(X_external) < 1:
            raise ParameterError("ExternalHoldoutSplitter requires a non-empty X_external=")
        if y_external is not None and len(y_external) != len(X_external):
            raise ParameterError(
                f"y_external has {len(y_external)} records but X_external has {len(X_external)}"
            )

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        m = len(self.X_external)
        train = np.arange(ctx.n, dtype=np.int64)
        test = np.arange(ctx.n, ctx.n + m, dtype=np.int64)
        metadata = {
            "realised_sizes": {"train": int(ctx.n), "valid": 0, "test": int(m)},
            "n_external": int(m),
        }
        return [
            SplitResult(
                train=train,
                valid=np.array([], dtype=np.int64),
                test=test,
                discard=np.array([], dtype=np.int64),
                groups=None,
                splitter_id=self.splitter_id,
                params=self.get_params(),
                n_records=ctx.n + m,
                metadata=metadata,
            )
        ]


class ApplicabilityDomainSplitter(BaseSplitter):
    """Produce a SERIES of test "bands" at increasing nearest-neighbour distance from train --
    a distance-vs-performance curve, not a single split.

    :param base_splitter: supplies the initial train/test partition, as an instance or a
        registry id. ``None`` uses a random split sized by this wrapper's own size parameters.
    :param n_bands: how many equal-count bands the test set is cut into, ordered by increasing
        nearest-neighbour distance to train.
    :param featurizer: featurizer alias used for the nearest-neighbour distances.
    :param metric: distance metric used for the nearest-neighbour distances.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``n_bands`` is below 1, or ``base_splitter`` is neither ``None``,
        a string nor a :class:`~chemsplit.base.BaseSplitter`.

    Notes
    -----
    ``get_n_splits()`` returns ``n_bands``; each yielded ``SplitResult`` shares the same
    ``train``, has ``test`` = that band's records, and ``discard`` = every other band's records
    plus whatever the wrapped splitter itself discarded or put in valid, so that each band's
    ``SplitResult`` covers every record on its own.

    Advantages
    ----------
    - Replaces "which split is correct?" with a measurement: error against distance from
      train. Every other split here is one point on that curve.
    - Produces the applicability-domain statement a deployed model needs, namely reliable below
      some distance and degrading beyond it.
    - Works on top of any base split, so the curve applies equally to a random, scaffold or
      time split.

    Pitfalls
    --------
    - Bands are subsets of one test set, so each is small and noisy. Merging the worst cases
      does not manufacture data.
    - The distance statistic depends on the metric and the fingerprint, so the curve's x-axis
      does not compare across representations.
    - A monotone-looking curve can be an artefact of a confound, such as molecular size
      increasing along the bands.

    References
    ----------
    .. [1] Sheridan, R. P.; Feuston, B. P.; Maiorov, V. N.; Kearsley, S. K. Similarity to
       Molecules in the Training Set Is a Good Discriminator for Prediction Accuracy in QSAR.
       *J. Chem. Inf. Comput. Sci.* **2004**, 44 (6), 1912-1928.
       https://doi.org/10.1021/ci049782w
    .. [2] Jaworska, J.; Nikolova-Jeliazkova, N.; Aldenberg, T. QSAR Applicability Domain
       Estimation by Projection of the Training Set in Descriptor Space: A Review.
       *Altern. Lab. Anim.* **2005**, 33 (5), 445-459.
       https://doi.org/10.1177/026119290503300508
    .. [3] Sahigara, F.; Mansouri, K.; Ballabio, D. et al. Comparison of Different Approaches
       to Define the Applicability Domain of QSAR Models. *Molecules* **2012**, 17 (5),
       4791-4810. https://doi.org/10.3390/molecules17054791
    """

    splitter_id: ClassVar[str] = "applicability_domain"
    family: ClassVar[str] = "protocol"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    group_forming: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        base_splitter: str | BaseSplitter | None = None,
        n_bands: int = 5,
        featurizer: str = "ecfp4",
        metric: str = "tanimoto",
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.base_splitter = base_splitter
        self.n_bands = n_bands
        self.featurizer = featurizer
        self.metric = metric
        if isinstance(n_bands, bool) or not isinstance(n_bands, int) or n_bands < 1:
            raise ParameterError(f"n_bands must be a positive int, got {n_bands!r}")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Report how many splits will be yielded.

        :param X: ignored, as are ``y`` and ``groups``; the signature is sklearn\'s.
        :return: ``n_bands``, one split per distance band.
        """
        return int(self.n_bands)

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        from chemsplit._fp_similarity import resolve_featurizer
        from chemsplit.metrics import nn_distance

        if self.base_splitter is None:
            from chemsplit.splitters.baseline import RandomSplitter

            seed0 = int(seed_for(ctx.rng_seeds, "ad.base_seed", 0).integers(0, 2**31 - 1))
            outer = RandomSplitter(
                train_size=ctx.sizes.n_train,
                valid_size=None,
                test_size=ctx.sizes.n_test,
                random_state=seed0,
            )
        else:
            # re-seed after resolving, as in NestedCVSplitter: resolving derives no seed, so
            # the outer split would otherwise ignore this splitter's random_state
            base_template = _resolve_splitter(self.base_splitter, "base_splitter")
            seed0 = int(seed_for(ctx.rng_seeds, "ad.base_seed", 0).integers(0, 2**31 - 1))
            outer = _clone_with_overrides(base_template, random_state=seed0)

        sub_X = _select_ctx_X(ctx)
        r = list(outer.split_result(sub_X, ctx.y))[0]
        train_idx, test_idx = r.train, r.test

        feat = resolve_featurizer(self.featurizer)
        F = ctx.get_features(feat)
        d = np.asarray(nn_distance(F[test_idx], F[train_idx], metric=self.metric))
        order = np.argsort(d, kind="stable")
        sorted_test = test_idx[order]
        sorted_d = d[order]

        n_bands = min(self.n_bands, len(sorted_test)) or 1
        band_groups = np.array_split(np.arange(len(sorted_test)), n_bands)
        all_test_set = set(test_idx.tolist())

        results: list[SplitResult] = []
        for b, positions in enumerate(band_groups):
            if positions.size == 0:
                continue
            band_test = sorted_test[positions]
            band_d = sorted_d[positions]
            other_test = np.asarray(
                sorted(all_test_set - set(band_test.tolist())), dtype=np.int64
            )
            discard = np.sort(np.concatenate([r.valid, r.discard, other_test]))
            metadata = {
                "realised_sizes": {
                    "train": int(train_idx.size),
                    "valid": 0,
                    "test": int(band_test.size),
                },
                "band_index": b,
                "distance_range": [float(band_d.min()), float(band_d.max())],
            }
            results.append(
                SplitResult(
                    train=np.sort(train_idx),
                    valid=np.array([], dtype=np.int64),
                    test=np.sort(band_test),
                    discard=discard,
                    groups=None,
                    splitter_id=self.splitter_id,
                    params=self.get_params(),
                    n_records=ctx.n,
                    metadata=metadata,
                )
            )
        return results


class IntersectionSplitter(BaseSplitter):
    """Compose two splitting criteria: run ``primary`` for train/valid/test, then enforce that no
    ``secondary`` group straddles the train/test (or train/valid) boundary.

    Answers questions like "train on historical data, test on the future, on scaffolds absent
    from train": pass ``primary=TemporalSplitter(...)`` and
    ``secondary=MurckoScaffoldSplitter()``.

    :param primary: the splitter supplying the initial train/valid/test partition, as an
        instance or a registry id.
    :param secondary: the group-forming splitter whose groups must not straddle a boundary, as
        an instance or a registry id. Used only for :meth:`compute_groups`.
    :param conflict_policy: for a train-side record whose ``secondary`` group also has a member
        in test, drop it from train or move it to test, growing test's group coverage.
        Train/valid conflicts always resolve by discarding.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``primary`` or ``secondary`` is of the wrong type, ``secondary``
        is not group-forming, or ``conflict_policy`` is unknown.
    :raises ConfigurationError: if ``secondary`` is missing.

    Advantages
    ----------
    - Composes any two of the criteria this library implements separately -- temporal,
      scaffold, source, similarity -- without a bespoke splitter for every pairing.
    - `n_conflicted_groups` reports the cost of the second constraint rather than absorbing it
      silently.
    - `secondary` is only ever used for grouping, so any existing `GroupSplitter` works
      unmodified.

    Pitfalls
    --------
    - `conflict_policy="discard"` can remove a large fraction of train when the two criteria
      disagree often, which `n_records_resolved` against `n_records` makes visible.
    - `"extend_test"` changes the realised test size unpredictably. `SizeToleranceWarning`
      fires but cannot fix it.
    - Only train/test and train/valid are checked, so test and valid may still share a
      `secondary` group.
    - Composing more than two criteria means nesting one `IntersectionSplitter` inside another,
      and the conflict-resolution order then matters and is not commutative in general.

    References
    ----------
    .. [1] A composition of two independent splitting criteria rather than a published
       method. The need to control several leakage channels at once is argued in
       Sheridan, R. P. Time-Split Cross-Validation as a Method for Estimating the Goodness of
       Prospective Prediction. *J. Chem. Inf. Model.* **2013**, 53 (4), 783-790.
       https://doi.org/10.1021/ci400084k
    """

    splitter_id: ClassVar[str] = "intersection"
    family: ClassVar[str] = "protocol"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = _ACCEPTS
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        primary: str | BaseSplitter | None = None,
        secondary: str | GroupSplitter | None = None,
        conflict_policy: Literal["discard", "extend_test"] = "discard",
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.primary = primary
        self.secondary = secondary
        self.conflict_policy = conflict_policy
        if conflict_policy not in ("discard", "extend_test"):
            raise ParameterError(f"invalid conflict_policy: {conflict_policy!r}")

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        X = _select_ctx_X(ctx)

        primary_template = _resolve_splitter(self.primary, "primary")
        seed0 = int(seed_for(ctx.rng_seeds, "intersection.primary", 0).integers(0, 2**31 - 1))
        primary = _clone_with_overrides(primary_template, random_state=seed0)
        r = primary.split_result(X, y=ctx.y, dates=ctx.dates, targets=ctx.targets)[0]
        train, valid, test, discard = r.train, r.valid, r.test, r.discard

        secondary_template = _resolve_splitter(self.secondary, "secondary")
        if not isinstance(secondary_template, GroupSplitter) or not (
            secondary_template.group_forming
        ):
            raise ParameterError("secondary must be a group-forming GroupSplitter")
        groups = np.asarray(secondary_template.compute_groups(X, ctx.y), dtype=np.int64)

        train_set = set(train.tolist())
        conflict_with_valid = set(groups[train].tolist()) & set(groups[valid].tolist())
        conflict_with_test = set(groups[train].tolist()) & set(groups[test].tolist())

        move_to_discard: set[int] = set()
        if conflict_with_valid:
            mask = np.isin(groups, sorted(conflict_with_valid))
            move_to_discard |= train_set & set(np.nonzero(mask)[0].tolist())

        move_to_test: set[int] = set()
        if conflict_with_test:
            mask = np.isin(groups, sorted(conflict_with_test))
            offending = (train_set & set(np.nonzero(mask)[0].tolist())) - move_to_discard
            if self.conflict_policy == "extend_test":
                move_to_test |= offending
            else:
                move_to_discard |= offending

        train = np.array(sorted(train_set - move_to_discard - move_to_test), dtype=np.int64)
        test = np.array(sorted(set(test.tolist()) | move_to_test), dtype=np.int64)
        discard = np.array(sorted(set(discard.tolist()) | move_to_discard), dtype=np.int64)

        result = _build_result(
            self,
            ctx,
            train=train,
            valid=valid,
            test=test,
            discard=discard,
            metadata={
                "primary_splitter_id": primary.splitter_id,
                "secondary_splitter_id": secondary_template.splitter_id,
                "n_conflicted_groups": len(conflict_with_test | conflict_with_valid),
                "n_records_resolved": len(move_to_discard) + len(move_to_test),
                "conflict_policy": self.conflict_policy,
            },
        )
        return [result]
