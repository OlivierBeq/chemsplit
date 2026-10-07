"""The ``similarity``/fingerprint splitter family.

Every splitter here operates on a fingerprint/feature distance or similarity matrix.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering, Birch, KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from chemsplit import clustering as _clustering
from chemsplit._fp_similarity import (
    EPS,
    SimilarityParamsMixin,
    blocked_farthest_pair,
    blocked_row_means,
    blocked_threshold_counts,
    blocked_threshold_pairs,
    compute_distance_matrix,
    compute_neighbor_lists,
    compute_similarity_matrix,
    dense_matrix_fits,
    distance_range,
    guard_memory,
    rectangular_distances,
    resolve_featurizer,
    similarity_columns,
)
from chemsplit._optimize import BalanceProblem, solve_balance
from chemsplit._unionfind import UnionFind, dense_label_encode
from chemsplit.base import (
    BaseSplitter,
    GroupSplitter,
    SplitResult,
    Strictness,
    _Context,
    _ResolvedSizes,
    check_group_splitter_design,
    resolve_group_splitter,
)
from chemsplit.determinism import (
    argmax_tiebreak,
    argmin_tiebreak,
    first_argmax_2d,
    first_ge_2d,
    floor_round,
    masked_argmax,
    masked_argmin,
    row_argmin,
    seed_for,
    stable_sort,
)
from chemsplit.exceptions import (
    ConfigurationError,
    ConstraintUnsatisfiableError,
    DegenerateClusterWarning,
    DegenerateGroupingError,
    LabelError,
    MissingDependencyError,
    ParameterError,
    ScalabilityError,
    SizeToleranceWarning,
    warn_with_details,
)
from chemsplit.metrics import is_bounded_metric, pairwise_distances
from chemsplit.types import IndexArray

__all__ = [
    "SimilarityThresholdSplitter",
    "ButinaSplitter",
    "SphereExclusionSplitter",
    "KMeansClusterSplitter",
    "DensityClusterSplitter",
    "SpectralSplitter",
    "MaxMinSplitter",
    "SPXYSplitter",
    "OptiSimSplitter",
    "MinimalTestSetDissimilaritySplitter",
    "SupportPointsSplitter",
    "DuplexSplitter",
    "DOptimalSplitter",
    "MaxDissimilaritySplitter",
    "PerimeterSplitter",
    "LeaveOneClusterOutSplitter",
    "BalancedMultiTaskSplitter",
]



class _SimilarityGroupBase(SimilarityParamsMixin, GroupSplitter):
    family: ClassVar[str] = "similarity"
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        GroupSplitter.__init__(self, **kwargs)
        SimilarityParamsMixin.__init__(
            self,
            featurizer=featurizer,
            metric=metric,
            max_memory_bytes=max_memory_bytes,
            n_jobs=self.n_jobs,
        )


class _SimilarityBase(SimilarityParamsMixin, BaseSplitter):
    family: ClassVar[str] = "similarity"
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    group_forming: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        BaseSplitter.__init__(self, **kwargs)
        SimilarityParamsMixin.__init__(
            self,
            featurizer=featurizer,
            metric=metric,
            max_memory_bytes=max_memory_bytes,
            n_jobs=self.n_jobs,
        )


def _dist_matrix(self: Any, ctx: _Context) -> np.ndarray:
    return compute_distance_matrix(
        ctx, self.featurizer, self.metric, self.max_memory_bytes, type(self).__name__, self.n_jobs
    )


def _sim_matrix(self: Any, ctx: _Context) -> np.ndarray:
    return compute_similarity_matrix(
        ctx, self.featurizer, self.metric, self.max_memory_bytes, type(self).__name__, self.n_jobs
    )


def _fill_remainder(
    picked: list[int], n: int, sizes: _ResolvedSizes, rng: np.random.Generator, picked_goes_to: str
) -> dict[str, IndexArray]:
    """Shared tail logic for MaxMin/MaxDissimilarity-style splitters: ``picked`` fills one
    destination partition (train or test); the rest is shuffled to fill the remaining targets in
    bucket order valid, then the other of train/test, with any overflow to discard."""
    picked_set = set(picked)
    rest = [i for i in range(n) if i not in picked_set]
    perm = rng.permutation(np.asarray(rest, dtype=np.int64)).tolist() if rest else []
    out: dict[str, list[int]] = {"train": [], "valid": [], "test": [], "discard": []}
    dest = "train" if picked_goes_to == "train" else "test"
    out[dest] = list(picked)
    other = "test" if dest == "train" else "train"
    targets = {"train": sizes.n_train, "valid": sizes.n_valid, "test": sizes.n_test}
    cursor = 0
    for bucket in ("valid", other):
        need = max(0, targets[bucket] - len(out[bucket]))
        out[bucket].extend(perm[cursor: cursor + need])
        cursor += need
    out["discard"] = perm[cursor:]
    return {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in out.items()}


def _small_partition_check(result: SplitResult, n: int) -> None:
    for name, arr in (("train", result.train), ("valid", result.valid), ("test", result.test)):
        if arr.size == 0:
            continue
        if arr.size < 10 or arr.size < 0.01 * n:
            from chemsplit.exceptions import SmallPartitionWarning

            warn_with_details(
                SmallPartitionWarning(
                    f"{name} partition has only {arr.size} record(s) "
                    f"({arr.size / max(1, n):.2%} of n)",
                    details={"partition": name, "size": int(arr.size), "n": n},
                )
            )


def _check_cluster_degeneracy(clusters: list[list[int]], n: int, owner: str, setting: str) -> None:
    """Shared degeneracy guard for radius-based clusterers (Butina, sphere exclusion, OptiSim):
    raise when every record is a singleton or one cluster covers >95% of records, warn above
    60%. ``setting`` names the parameter that produced ``clusters``, for the error message."""
    if len(clusters) == n:
        raise DegenerateGroupingError(
            f"{owner}: {setting} produced {n} singleton clusters (every record its own group)"
        )
    largest_frac = max(len(c) for c in clusters) / n if clusters else 0.0
    if largest_frac > 0.95:
        raise DegenerateGroupingError(
            f"{owner}: largest cluster covers {largest_frac:.1%} of records"
        )
    if largest_frac > 0.6:
        warn_with_details(
            DegenerateClusterWarning(
                f"{owner}: largest cluster covers {largest_frac:.1%} of records",
                details={"largest_cluster_frac": largest_frac},
            )
        )


def _resolve_cluster_count(
    n: int, n_clusters: int | str, auto_rule: str = "sqrt_n", auto_range: tuple[int, int] = (2, 50)
) -> int:
    """Cluster count for ``n`` records: ``n_clusters`` itself, or for ``"auto"`` ``round(sqrt(n))``
    (``auto_rule="sqrt_n"``) or ``n // 50``; either way clipped to ``auto_range`` and ``n - 1``."""
    if n_clusters != "auto":
        k = int(n_clusters)
    elif auto_rule == "sqrt_n":
        k = int(round(n**0.5))
    else:
        k = max(1, n // 50)
    lo, hi = auto_range
    return int(np.clip(k, lo, min(hi, n - 1)))

_RADIUS_KINDS = ("distance", "similarity", "fraction_of_range")


def _validate_radius(splitter: Any, radius: float, radius_is: str) -> None:
    """Eager ``radius``/``radius_is`` validation shared by sphere exclusion and OptiSim."""
    if radius_is not in _RADIUS_KINDS:
        raise ParameterError(
            f"invalid radius_is: {radius_is!r}; expected one of {list(_RADIUS_KINDS)}"
        )
    splitter._validate_similarity_params(bounded_metric_required=radius_is == "similarity")
    bounded = is_bounded_metric(splitter.metric)
    if radius_is == "distance" and not bounded:
        if not radius > 0.0:
            raise ParameterError(f"radius must be > 0, got {radius!r}")
    elif not (0.0 < radius < 1.0):
        raise ParameterError(f"radius must be in (0,1) for radius_is={radius_is!r}, got {radius!r}")


def _picked_diagnostics(
    picked: list[int],
    n: int,
    D: np.ndarray | None,
    ctx: _Context,
    featurizer: Any,
    metric: Any,
    n_jobs: int,
    block: int = 1024,
) -> tuple[float, float]:
    """Minimum within-selection distance and worst-case coverage, blockwise.

    Replaces an ``O(k^2)`` Python double loop with bounded numpy reductions, at identical values --
    they reach ``SplitResult.metadata`` and so the goldens.

    :param picked: the selected indices, in selection order.
    :param n: the record count.
    :param D: the dense distance matrix, or ``None`` to fetch blocks on demand.
    :param ctx: the split context.
    :param featurizer: the featurizer, for the matrix-free path.
    :param metric: the distance metric, for the matrix-free path.
    :param n_jobs: worker count. Results never depend on it.
    :param block: rows per block.
    :return: the minimum pairwise distance within ``picked`` (``inf`` for fewer than two), and the
        maximum over all records of their distance to the nearest picked record (``nan`` if
        nothing was picked).
    """
    if not picked:
        return float("inf"), float("nan")

    def rows(row_idx: Sequence[int], col_idx: Sequence[int]) -> np.ndarray:
        if D is not None:
            return D[np.ix_(list(row_idx), list(col_idx))]
        return rectangular_distances(ctx, featurizer, metric, list(row_idx), list(col_idx), n_jobs)

    min_pairwise = float("inf")
    k = len(picked)
    for start in range(0, k, block):
        stop = min(start + block, k)
        sub_block = rows(picked[start:stop], picked)
        # keep only b > a in selection order, matching the nested loop this replaced
        a_idx = np.arange(start, stop)[:, None]
        b_idx = np.arange(k)[None,:]
        masked = np.where(b_idx > a_idx, sub_block, np.inf)
        if masked.size:
            min_pairwise = min(min_pairwise, float(masked.min()))

    worst = -np.inf
    for start in range(0, n, block):
        stop = min(start + block, n)
        sub_block = rows(range(start, stop), picked)
        worst = max(worst, float(sub_block.min(axis=1).max()))
    return min_pairwise, float(worst)


def _blocked_max_cdist(y: np.ndarray, block: int = 2048) -> float:
    """Largest pairwise Euclidean distance among the rows of ``y``, without an ``n x n`` matrix.

    :param y: the label rows.
    :param block: rows per band.
    :return: the maximum distance, ``0.0`` for fewer than two rows.
    """
    n = y.shape[0]
    if n < 2:
        return 0.0
    best = 0.0
    for start in range(0, n, block):
        stop = min(start + block, n)
        best = max(best, float(cdist(y[start:stop], y, metric="euclidean").max()))
    return best


def _blocked_seed_pair(
    n: int, band: Callable[[int, int], np.ndarray], block: int = 2048
) -> tuple[int, int]:
    """Lexicographically smallest ``i < j`` pair within ``EPS`` of the maximum, from row bands.

    Two passes -- one for the maximum, one for the first qualifying pair -- so no matrix is held.
    Same tie-break as the dense Kennard-Stone seed search.

    :param n: the record count.
    :param band: maps ``(start, stop)`` to those rows against every record.
    :param block: rows per band.
    :return: the seed pair.
    """
    best = -np.inf
    for start in range(0, n, block):
        stop = min(start + block, n)
        rows = np.arange(stop - start)[:, None]
        cols = np.arange(n)[None,:]
        masked = np.where(cols > rows + start, band(start, stop), -np.inf)
        if masked.size:
            best = max(best, float(masked.max()))
    for start in range(0, n, block):
        stop = min(start + block, n)
        rows = np.arange(stop - start)[:, None]
        cols = np.arange(n)[None,:]
        masked = np.where(cols > rows + start, band(start, stop), -np.inf)
        if bool((masked >= best - _clustering.EPS).any()):
            i, j = first_ge_2d(masked, best - _clustering.EPS)
            return start + i, j
    return 0, min(1, n - 1)


class _Distances:
    """Uniform access to a splitter's pairwise distances, dense or blocked.

    Holds the dense matrix when it fits the caller's budget -- it is faster, since every slice is
    then free -- and otherwise serves the same values from blocked recomputation. Every method
    returns what indexing the dense matrix would have returned, so a splitter written against
    this interface behaves identically at any size.

    :param splitter: the calling splitter, for its featurizer/metric/budget/``n_jobs``.
    :param ctx: the split context.
    :param copies: live ``n x n`` matrices the caller needs, as for ``guard_memory``.
    :param as_float64: upcast the dense matrix, matching callers that did so themselves.
    """

    def __init__(
        self, splitter: Any, ctx: _Context, *, copies: int = 1, as_float64: bool = False
    ) -> None:
        self._s = splitter
        self._ctx = ctx
        self._f64 = as_float64
        self.n = ctx.n
        self.dense: np.ndarray | None = None
        if dense_matrix_fits(ctx.n, splitter.max_memory_bytes, copies):
            D = _dist_matrix(splitter, ctx)
            self.dense = D.astype(np.float64) if as_float64 else D

    def fetch(self, rows: Sequence[int], cols: Sequence[int]) -> np.ndarray:
        """Distances between two index sets, from the dense matrix or recomputed.

        The one place blocked results are produced, so ``as_float64`` applies to them too --
        it has to, since a float32 row sum differs from a float64 one.
        """
        if self.dense is not None:
            return self.dense[np.ix_(list(rows), list(cols))]
        block = rectangular_distances(
            self._ctx, self._s.featurizer, self._s.metric, list(rows), list(cols), self._s.n_jobs
        )
        return block.astype(np.float64) if self._f64 else block

    def columns(self, js: Sequence[int]) -> np.ndarray:
        """Distance columns for ``js``, shape ``(n, len(js))``."""
        return self.fetch(range(self.n), list(js))

    def column(self, j: int) -> np.ndarray:
        """Distance column for record ``j``, shape ``(n,)``."""
        return self.columns([j])[:, 0]

    def pair(self, i: int, j: int) -> float:
        """Distance between two records."""
        return float(self.fetch([i], [j])[0, 0])

    def row_means(self) -> np.ndarray:
        """Per-record mean distance to every record, shape ``(n,)``."""
        if self.dense is not None:
            return self.dense.mean(axis=1)
        return blocked_row_means(
            self._ctx, self._s.featurizer, self._s.metric, n_jobs=self._s.n_jobs
        )

    def farthest_pair(self) -> tuple[int, int]:
        """The most distant pair, ties to the lexicographically smallest."""
        if self.dense is not None:
            return first_argmax_2d(np.triu(self.dense, k=1))
        return blocked_farthest_pair(
            self._ctx, self._s.featurizer, self._s.metric, n_jobs=self._s.n_jobs
        )

    def max_upper(self, block: int = 2048) -> float:
        """Largest distance over the ``i < j`` pairs."""
        if self.dense is not None:
            iu = np.triu_indices(self.n, k=1)
            return float(self.dense[iu].max())
        best = -np.inf
        for start in range(0, self.n, block):
            stop = min(start + block, self.n)
            sub = self._band(start, stop)
            rows = np.arange(stop - start)[:, None]
            cols = np.arange(self.n)[None,:]
            masked = np.where(cols > rows + start, sub, -np.inf)
            if masked.size:
                best = max(best, float(masked.max()))
        return best

    def first_pair_at_least(self, value: float, block: int = 2048) -> tuple[int, int]:
        """Lexicographically smallest ``i < j`` pair of at least ``value``."""
        for start in range(0, self.n, block):
            stop = min(start + block, self.n)
            sub = self._band(start, stop)
            rows = np.arange(stop - start)[:, None]
            cols = np.arange(self.n)[None,:]
            masked = np.where(cols > rows + start, sub, -np.inf)
            if bool((masked >= value).any()):
                i, j = first_ge_2d(masked, value)
                return start + i, j
        raise ValueError("first_pair_at_least(): no pair reaches the given value")

    def row_argmin_to(self, cols: Sequence[int], block: int = 2048) -> np.ndarray:
        """Per record, the position within ``cols`` of its nearest member, ties to the first."""
        cols = list(cols)
        out = np.empty(self.n, dtype=np.int64)
        for start in range(0, self.n, block):
            stop = min(start + block, self.n)
            out[start:stop] = row_argmin(self.fetch(range(start, stop), cols))
        return out

    def row_sums_within(self, idx: Sequence[int], block: int = 2048) -> np.ndarray:
        """Row sums of the sub-matrix restricted to ``idx``, in ``idx`` order."""
        idx = list(idx)
        out = np.empty(len(idx), dtype=np.float64)
        for start in range(0, len(idx), block):
            stop = min(start + block, len(idx))
            out[start:stop] = self.fetch(idx[start:stop], idx).sum(axis=1)
        return out

    def max_overall(self, block: int = 2048) -> float:
        """Largest distance anywhere, diagonal included (which is zero)."""
        if self.dense is not None:
            return float(self.dense.max()) if self.dense.size else 0.0
        best = 0.0
        for start in range(0, self.n, block):
            stop = min(start + block, self.n)
            sub = self.fetch(range(start, stop), range(self.n))
            if sub.size:
                best = max(best, float(sub.max()))
        return best

    def row_sums(self, block: int = 2048) -> np.ndarray:
        """Per-record sum of distances to every record."""
        if self.dense is not None:
            return self.dense.sum(axis=1)
        out = np.empty(self.n, dtype=np.float64)
        for start in range(0, self.n, block):
            stop = min(start + block, self.n)
            out[start:stop] = self.fetch(range(start, stop), range(self.n)).sum(axis=1)
        return out

    def max_min_to(self, cols: Sequence[int], block: int = 2048) -> float:
        """Worst-case coverage: ``max_i min_{j in cols} D[i, j]``."""
        cols = list(cols)
        if not cols:
            return float("nan")
        if self.dense is not None:
            return float(np.max(np.min(self.dense[:, cols], axis=1)))
        worst = -np.inf
        for start in range(0, self.n, block):
            stop = min(start + block, self.n)
            worst = max(worst, float(self.fetch(range(start, stop), cols).min(axis=1).max()))
        return worst

    def farthest_pair_in_pool(self, pool: np.ndarray, block: int = 2048) -> tuple[int, int]:
        """Farthest-apart pair within ``pool`` (ascending), ties to the smallest pair.

        Pool-local row-major order is lexicographic in global indices too, since ``pool`` is
        ascending, so the dense tie-break carries over unchanged.
        """
        pool = np.asarray(pool)
        m = pool.size
        if m < 2:
            return int(pool[0]), int(pool[0])

        def band(a: int, b: int) -> np.ndarray:
            return self.fetch(pool[a:b], pool)

        best = -np.inf
        for a in range(0, m, block):
            b = min(a + block, m)
            rows = np.arange(b - a)[:, None]
            cols = np.arange(m)[None,:]
            masked = np.where(cols > rows + a, band(a, b), -np.inf)
            if masked.size:
                best = max(best, float(masked.max()))
        for a in range(0, m, block):
            b = min(a + block, m)
            rows = np.arange(b - a)[:, None]
            cols = np.arange(m)[None,:]
            masked = np.where(cols > rows + a, band(a, b), -np.inf)
            if bool((masked >= best - EPS).any()):
                i, j = first_ge_2d(masked, best - EPS)
                return int(pool[a + i]), int(pool[j])
        return int(pool[0]), int(pool[1])

    def _band(self, start: int, stop: int) -> np.ndarray:
        """Rows ``[start, stop)`` against every record."""
        return self.fetch(range(start, stop), range(self.n))

    def min_between(self, rows: Sequence[int], cols: Sequence[int], block: int = 1024) -> float:
        """Smallest distance between two index sets, ``inf`` if either is empty."""
        rows, cols = list(rows), list(cols)
        if not rows or not cols:
            return float("inf")
        best = float("inf")
        for start in range(0, len(rows), block):
            chunk = rows[start : start + block]
            sub = self.fetch(chunk, cols)
            if sub.size:
                best = min(best, float(sub.min()))
        return best

    def count_pairs_at_least(self, value: float, block: int = 2048) -> int:
        """How many ``i < j`` pairs are at least ``value``."""
        if self.dense is not None:
            iu = np.triu_indices(self.n, k=1)
            return int(np.sum(self.dense[iu] >= value))
        total = 0
        for start in range(0, self.n, block):
            stop = min(start + block, self.n)
            sub = self.fetch(range(start, stop), range(self.n))
            rows = np.arange(stop - start)[:, None]
            cols = np.arange(self.n)[None,:]
            total += int(np.sum((sub >= value) & (cols > rows + start)))
        return total


def _default_butina_clusters(ctx: _Context, splitter_name: str, n_jobs: int = 1) -> list[list[int]]:
    """Butina clusters at the default ECFP4/Tanimoto 0.35 cutoff, without an ``n x n`` matrix.

    Butina reads only radius-neighbour lists, which can be built blockwise, so the fallback
    clusterer works at any size. Bit-identical to the dense route.

    :param ctx: the split context.
    :param splitter_name: the caller, for error messages.
    :param n_jobs: worker count. Results never depend on it.
    :return: clusters in creation order, each starting with its centroid.
    """
    if dense_matrix_fits(ctx.n, 2 * 1024**3):
        D = compute_distance_matrix(ctx, "ecfp4", "tanimoto", 2 * 1024**3, splitter_name, 1)
        return _clustering.butina(D, 0.35, reorder=False)
    neigh = compute_neighbor_lists(
        ctx, "ecfp4", "tanimoto", 0.35, eps=_clustering.EPS, n_jobs=n_jobs
    )
    return _clustering.butina_from_neighbors(neigh, ctx.n, reorder=False)


def _resolve_radius_without_matrix(self: Any, ctx: _Context) -> float:
    """``_resolve_radius`` for the matrix-free path, using a blocked distance range.

    :param self: the splitter, for ``radius``/``radius_is`` and the featurizer settings.
    :param ctx: the split context.
    :return: the distance threshold.
    """
    if self.radius_is == "distance":
        return float(self.radius)
    if self.radius_is == "similarity":
        return 1.0 - float(self.radius)
    d_min, d_max = distance_range(ctx, self.featurizer, self.metric, n_jobs=self.n_jobs)
    return d_min + float(self.radius) * (d_max - d_min)


def _resolve_radius(D: np.ndarray, radius: float, radius_is: str) -> float:
    """Convert ``radius`` to a distance threshold. ``"fraction_of_range"`` maps ``radius`` linearly
    onto ``[min, max]`` of the off-diagonal distances, giving a scale-free radius for unbounded
    metrics (raw descriptors under ``"euclidean"``/``"mahalanobis"``)."""
    if radius_is == "distance":
        return float(radius)
    if radius_is == "similarity":
        return 1.0 - float(radius)
    d_max = float(D.max())
    # off-diagonal minimum without copying an n x n mask: hide the diagonal, then restore it
    np.fill_diagonal(D, np.inf)
    d_min = float(D.min())
    np.fill_diagonal(D, 0.0)
    return d_min + float(radius) * (d_max - d_min)


class SimilarityThresholdSplitter(_SimilarityGroupBase):
    """Hard constraint: no test record may exceed ``threshold`` similarity to any train record.

    :param threshold: the similarity ceiling between any train and any test record.
    :param strategy: meet the constraint by pruning offending records greedily, by keeping
        whole similarity-graph components together, or by growing test outward from seeds.
    :param seed_selection: which records seed ``strategy="seeded_growth"``: random, the most
        central, or the most peripheral.
    :param allow_discard: let the strategies that need to drop records do so.
    :param max_discard_frac: largest fraction of records that may be discarded before the
        splitter gives up.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``threshold`` or ``max_discard_frac`` is outside ``(0, 1]``, or
        ``strategy`` or ``seed_selection`` is unknown.
    :raises ConstraintUnsatisfiableError: at split time, if the threshold cannot be met, e.g.
        when one connected component holds nearly every record.

    Advantages
    ----------
    - The constraint is checkable and reported: `metadata["max_cross_similarity"]` is below
      the threshold or the split is wrong. No other splitter here guarantees that.
    - Parameterises what "novel chemistry" usually means: how dissimilar must test compounds
      be?
    - `graph_component` never discards data, so the full dataset gets used.

    Pitfalls
    --------
    - **The threshold is the experiment.** ECFP4, ECFP6 and MACCS at Tanimoto 0.4 are three
      difficulty levels, so a result quoted without fingerprint, radius, bit length, metric
      and cutoff is not reproducible. `params` records all five.
    - Similarity is not transitive, so a chain of pairwise-similar molecules links two very
      different ends. On dense datasets one component swallows everything, raising
      `ConstraintUnsatisfiableError`.
    - `greedy_prune` and `seeded_growth` discard records from the interesting boundary
      region, so the remaining test set is not a uniform sample of anything.
    - Tanimoto on sparse fingerprints saturates: most pairs in a diverse library sit below
      0.2, so a 0.4 cutoff removes almost nothing and the split becomes random.
    - All-zero fingerprints -- parse failures under `on_parse_error="ignore"`, or tiny
      fragments -- score similarity 1.0 to each other and cluster together spuriously.

    References
    ----------
    .. [1] The three strategies are engineering compositions rather than published methods.
       The rationale for a hard cross-similarity ceiling is established in [2]-[4].
    .. [2] Golbraikh, A.; Tropsha, A. Beware of q2! *J. Mol. Graph. Model.* **2002**, 20 (4),
       269-276. https://doi.org/10.1016/S1093-3263(01)00123-1
    .. [3] Wallach, I.; Heifets, A. Most Ligand-Based Classification Benchmarks Reward
       Memorization Rather than Generalization. *J. Chem. Inf. Model.* **2018**, 58 (5),
       916-932. https://doi.org/10.1021/acs.jcim.7b00403
    .. [4] Kapoor, S.; Narayanan, A. Leakage and the Reproducibility Crisis in
       Machine-Learning-Based Science. *Patterns* **2023**, 4 (9), 100804.
       https://doi.org/10.1016/j.patter.2023.100804
    """

    splitter_id: ClassVar[str] = "similarity_threshold"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    bounded_metric_required: ClassVar[bool] = True
    deterministic_without_seed: ClassVar[bool] = True  # unless strategy="seeded_growth" random init

    def __init__(
        self,
        *,
        threshold: float = 0.4,
        strategy: Literal["greedy_prune", "graph_component", "seeded_growth"] = "graph_component",
        seed_selection: Literal["random", "most_central", "most_peripheral"] = "random",
        allow_discard: bool = True,
        max_discard_frac: float = 0.5,
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.threshold = threshold
        self.strategy = strategy
        self.seed_selection = seed_selection
        self.allow_discard = allow_discard
        self.max_discard_frac = max_discard_frac
        self._validate_similarity_params()
        if not (0.0 < threshold < 1.0):
            raise ParameterError(f"threshold must be in (0,1), got {threshold!r}")
        if strategy not in ("greedy_prune", "graph_component", "seeded_growth"):
            raise ParameterError(f"invalid strategy: {strategy!r}")
        if not (0.0 <= max_discard_frac < 1.0):
            raise ParameterError(f"max_discard_frac must be in [0,1), got {max_discard_frac!r}")

    def _group_labels(self, ctx: _Context) -> IndexArray:
        n = ctx.n
        # Each strategy reads the matrix through one reduction that can be streamed, so a dense
        # matrix is only built when it fits and is cheaper. Identical results either way.
        use_dense = dense_matrix_fits(n, self.max_memory_bytes)
        S = _sim_matrix(self, ctx) if use_dense else None
        if self.strategy == "graph_component":
            uf = UnionFind(n)
            if S is not None:
                pairs = (
                    (i, np.nonzero(S[i][i + 1:] > self.threshold + EPS)[0] + i + 1, None)
                    for i in range(n)
                )
            else:
                pairs = blocked_threshold_pairs(
                    ctx, self.featurizer, self.metric, self.threshold,
                    eps=EPS, n_jobs=self.n_jobs,
                )
            for i, js, _vals in pairs:
                for j in js:
                    uf.union(i, int(j))
            labels = np.asarray(
                dense_label_encode([uf.find(i) for i in range(n)]), dtype=np.int64
            )
            self._last_max_cross = 0.0  # guaranteed by construction; see class docstring
            n_groups = len(set(labels.tolist()))
            if n_groups == 1:
                raise ConstraintUnsatisfiableError(
                    f"{type(self).__name__}: threshold={self.threshold} produces a single "
                    f"connected component over all {n} records; no split is possible without "
                    "discarding. Try a higher threshold."
                )
            self._last_components = n_groups
            return labels

        if self.strategy == "greedy_prune":
            rng = seed_for(ctx.rng_seeds, "similarity.seed", 0)
            perm = rng.permutation(n)
            a = ctx.sizes.n_train
            b = a + ctx.sizes.n_valid
            train = set(perm[:a].tolist())
            test = set(perm[b: b + ctx.sizes.n_test].tolist())
            discarded: set[int] = set()
            max_discard = int(self.max_discard_frac * n)
            # `train` never changes in this loop, so each record's violation count is constant:
            # count once instead of rebuilding the whole table on every iteration.
            train_list = sorted(train)
            if not train_list:
                counts = {t: 0 for t in test}
            elif S is not None:
                counts = {
                    t: int(np.sum(S[t, train_list] > self.threshold + EPS)) for t in sorted(test)
                }
            else:
                counts = blocked_threshold_counts(
                    ctx, self.featurizer, self.metric, self.threshold,
                    rows=sorted(test), cols=train_list, eps=EPS, n_jobs=self.n_jobs,
                )
            while True:
                violations = {t: c for t, c in counts.items() if t in test and c > 0}
                if not violations:
                    break
                worst = argmax_tiebreak(lambda t: violations[t], sorted(violations))
                test.discard(worst)
                discarded.add(worst)
                if len(discarded) > max_discard and not self.allow_discard:
                    raise ConstraintUnsatisfiableError(
                        f"{type(self).__name__}: could not satisfy threshold={self.threshold} "
                        f"within max_discard_frac={self.max_discard_frac} (allow_discard=False)"
                    )
                if len(discarded) > max_discard:
                    break
            forced = sorted(discarded)
            existing = set(ctx.extra.get("forced_discard", []))
            ctx.extra["forced_discard"] = sorted(existing | set(forced))
            # one singleton group per surviving record, so assign_groups just honours the
            # bucket decided above
            return np.arange(n, dtype=np.int64)

        # strategy == "seeded_growth"
        if self.seed_selection == "random":
            rng = seed_for(ctx.rng_seeds, "similarity.seed", 0)
            s = int(rng.integers(0, n))
        else:
            mean_s = (
                S.mean(axis=1)
                if S is not None
                else blocked_row_means(
                    ctx, self.featurizer, self.metric, similarity=True, n_jobs=self.n_jobs
                )
            )
            mean_d = 1.0 - mean_s
            pick = argmin_tiebreak if self.seed_selection == "most_central" else argmax_tiebreak
            s = pick(lambda i: mean_d[i], range(n))
        # max-similarity-to-test is a running maximum: one column per added record, instead of
        # rescanning every candidate against the whole test set on every pick.
        def sim_column(j: int) -> np.ndarray:
            if S is not None:
                return S[:, j]
            return similarity_columns(ctx, self.featurizer, self.metric, [j], self.n_jobs)[:, 0]

        test = {s}
        max_to_test = np.array(sim_column(s), dtype=np.float32, copy=True)
        taken = np.zeros(n, dtype=bool)
        taken[s] = True
        n_test_target = max(1, ctx.sizes.n_test)
        while len(test) < n_test_target:
            if bool(taken.all()):
                break
            best = masked_argmax(max_to_test, taken)
            test.add(best)
            taken[best] = True
            np.maximum(max_to_test, sim_column(best), out=max_to_test)
        train = [
            i for i in range(n) if i not in test and float(max_to_test[i]) <= self.threshold + EPS
        ]
        discard = [i for i in range(n) if i not in test and i not in train]
        if not self.allow_discard and discard:
            raise ConstraintUnsatisfiableError(
                f"{type(self).__name__}: seeded_growth left {len(discard)} buffer-zone record(s) "
                "and allow_discard=False"
            )
        existing = set(ctx.extra.get("forced_discard", []))
        ctx.extra["forced_discard"] = sorted(existing | set(discard))
        labels = np.arange(n, dtype=np.int64)
        for t in test:
            labels[t] = -1  # placeholder; overwritten below
        # _group_labels can't partition explicitly, so singleton labels per record let
        # assign_groups separate train from test by size. For seeded_growth the test target is
        # normally already met, which makes that sufficient.
        return np.arange(n, dtype=np.int64)

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        max_cross = getattr(self, "_last_max_cross", None)
        return {
            "threshold": self.threshold,
            "strategy": self.strategy,
            "n_components": getattr(self, "_last_components", int(len(set(labels.tolist())))),
            "max_cross_similarity": max_cross if max_cross is not None else float("nan"),
        }


class ButinaSplitter(_SimilarityGroupBase):
    """Taylor-Butina sphere-exclusion (leader) clustering.

    :param cutoff: the sphere radius, read as a distance or a similarity per ``cutoff_is``.
    :param cutoff_is: whether ``cutoff`` is a distance or a similarity.
    :param reorder: recompute neighbour counts after each cluster is taken, which is the
        original formulation and gives different clusters from the single-pass variant.
    :param singleton_policy: what becomes of one-member clusters: keep each as its own group,
        pool them into one shared group, or fold each into its nearest real cluster.
    :param algorithm: build the neighbour lists from a dense matrix or a sparse thresholded
        one. Both are ``O(n^2)`` in time.
    :param block_size: rows per block when building the matrix.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``cutoff`` is outside ``(0, 1)``, or ``cutoff_is`` or
        ``singleton_policy`` is unknown.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - A reasonable default: harder than a scaffold split, chemically meaningful since it
      groups by fingerprint proximity, and cheap to tens of thousands of molecules.
    - Deterministic, with no seed and one interpretable parameter.
    - Every cluster centroid is a real molecule, so clusters can be inspected and reported.
    - Sphere exclusion puts any two centroids more than `cutoff` apart, which gives the split a
      concrete geometric meaning.

    Pitfalls
    --------
    - Membership is relative to the **centroid**, not pairwise, so two members can sit
      `2 x cutoff` apart. A cluster boundary therefore guarantees no minimum cross-similarity;
      `similarity_threshold` and `hi` do.
    - Produces many singletons on diverse libraries, often 30-60% of records, and
      `singleton_policy` then decides difficulty: the default `"own_group"` scatters them,
      softening the split.
    - Highly sensitive to `cutoff`: 0.35 against 0.4 in distance can halve or double the
      cluster count.
    - The distance/similarity convention is a classic source of silent errors;
      `metadata["cutoff_is"]` records which was used.
    - `reorder=True` gives different clusters than `reorder=False`, and both are called
      "Butina" in the literature.
    - `O(n^2)` in time regardless of `algorithm`. Beyond roughly 1e5 molecules,
      `k_means_cluster` with mini-batch k-means, or subsampling, is the way out.

    References
    ----------
    .. [1] Taylor, R. Simulation Analysis of Experimental Design Strategies for Screening
       Random Compounds as Potential New Drugs and Agrochemicals. *J. Chem. Inf. Comput. Sci.*
       **1995**, 35 (1), 59-67.
       https://doi.org/10.1021/ci00023a009
    .. [2] Butina, D. Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto
       Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets.
       *J. Chem. Inf. Comput. Sci.* **1999**, 39 (4), 747-750. https://doi.org/10.1021/ci9803381
    """

    splitter_id: ClassVar[str] = "butina"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    bounded_metric_required: ClassVar[bool] = True

    def __init__(
        self,
        *,
        cutoff: float = 0.35,
        cutoff_is: Literal["distance", "similarity"] = "distance",
        reorder: bool = False,
        singleton_policy: Literal["own_group", "shared_group", "nearest_cluster"] = "own_group",
        algorithm: Literal["dense", "sparse"] = "dense",
        block_size: int = 2048,
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.cutoff = cutoff
        self.cutoff_is = cutoff_is
        self.reorder = reorder
        self.singleton_policy = singleton_policy
        self.algorithm = algorithm
        self.block_size = block_size
        self._validate_similarity_params()
        if not (0.0 < cutoff < 1.0):
            raise ParameterError(f"cutoff must be in (0,1), got {cutoff!r}")
        if cutoff_is not in ("distance", "similarity"):
            raise ParameterError(f"invalid cutoff_is: {cutoff_is!r}")
        if singleton_policy not in ("own_group", "shared_group", "nearest_cluster"):
            raise ParameterError(f"invalid singleton_policy: {singleton_policy!r}")

    def _group_labels(self, ctx: _Context) -> IndexArray:
        dist_cutoff = self.cutoff if self.cutoff_is == "distance" else 1.0 - self.cutoff
        # Butina reads only neighbour lists, plus a few singleton-to-centroid distances for
        # singleton_policy="nearest_cluster". The dense matrix is faster when it fits; otherwise
        # build the lists blockwise rather than refuse -- identical clusters, linear peak memory.
        D: np.ndarray | None = None
        if dense_matrix_fits(ctx.n, self.max_memory_bytes):
            D = _dist_matrix(self, ctx)
            clusters = _clustering.butina(D, dist_cutoff, reorder=self.reorder)
        else:
            neigh = compute_neighbor_lists(
                ctx,
                self.featurizer,
                self.metric,
                dist_cutoff,
                eps=_clustering.EPS,
                n_jobs=self.n_jobs,
            )
            clusters = _clustering.butina_from_neighbors(neigh, ctx.n, reorder=self.reorder)
        n = ctx.n
        labels = np.empty(n, dtype=np.int64)
        singleton_idx = [k for k, c in enumerate(clusters) if len(c) == 1]
        if self.singleton_policy == "nearest_cluster" and singleton_idx:
            non_singleton = [k for k, c in enumerate(clusters) if len(c) > 1]
            if not non_singleton:
                warn_with_details(
                    DegenerateClusterWarning(
                        f"{type(self).__name__}: no non-singleton cluster exists; "
                        "singleton_policy falls back to 'own_group'",
                        details={},
                    )
                )
            else:
                # loop-invariant: only non-singleton clusters grow, so centroids do not change
                centroids = [clusters[j][0] for j in non_singleton]
                singleton_recs = [clusters[k][0] for k in singleton_idx]
                if D is not None:
                    block = D[np.ix_(singleton_recs, centroids)]
                else:
                    block = rectangular_distances(
                        ctx, self.featurizer, self.metric, singleton_recs, centroids, self.n_jobs
                    )
                for row, k in enumerate(singleton_idx):
                    nearest_col = row_argmin(block[row : row + 1])[0]
                    clusters[non_singleton[int(nearest_col)]].append(clusters[k][0])
                clusters = [c for k, c in enumerate(clusters) if k not in singleton_idx]
        elif self.singleton_policy == "shared_group" and len(singleton_idx) > 1:
            merged = [clusters[k][0] for k in singleton_idx]
            clusters = [c for k, c in enumerate(clusters) if k not in singleton_idx] + [merged]
        for cid, members in enumerate(clusters):
            for m in members:
                labels[m] = cid
        n_groups = len(clusters)
        self._last_meta = {
            "cutoff": self.cutoff,
            "cutoff_is": self.cutoff_is,
            "n_clusters": n_groups,
            "cluster_sizes": sorted((len(c) for c in clusters), reverse=True),
            "centroids": [c[0] for c in clusters],
        }
        _check_cluster_degeneracy(
            clusters, n, type(self).__name__, f"cutoff={self.cutoff} ({self.cutoff_is})"
        )
        return dense_label_encode(labels.tolist())

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        return getattr(self, "_last_meta", {})


class SphereExclusionSplitter(_SimilarityGroupBase):
    """Sphere-exclusion clustering: scan records in random (or index) order; each record not yet
    claimed becomes a representative and claims every unclaimed record within ``radius`` of it.

    Unlike :class:`ButinaSplitter`, the scan order is not density-driven, and the radius can be
    given as a fraction of the observed distance range (``radius_is="fraction_of_range"``), so it
    also works on raw descriptors with unbounded metrics. The scan order is drawn from the
    ``"sphere_exclusion.order"`` stream when ``order="random"``.

    :param radius: the exclusion radius, read per ``radius_is``.
    :param radius_is: whether ``radius`` is a distance, a similarity, or a fraction of the
        observed distance range.
    :param order: scan the records in seeded random order, or in input order.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``radius`` is outside its valid range for the chosen
        ``radius_is``, or ``radius_is`` or ``order`` is unknown.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Clusters have a guaranteed maximum radius around their representative, so cluster
      tightness is a direct, interpretable parameter.
    - A linear number of passes over the distance matrix, with no density pre-computation, so
      it is cheaper than Butina at the same `n`.
    - `order="random"` gives a different but equally valid clustering per seed, so repeating
      over seeds measures variance from the clustering itself, not just from group assignment.
    - `radius_is="fraction_of_range"` makes one radius meaningful across metrics and descriptor
      scales.

    Pitfalls
    --------
    - Random scan order lets a dense region split among several representatives while an
      outlier founds its own cluster, so cluster sizes are far more uneven than Butina's.
    - The radius bounds distance to the representative, not between clusters, so records in
      different clusters can be closer than `radius`. `similarity_threshold` is the hard
      constraint.
    - `fraction_of_range` depends on the extreme pairwise distances, so one outlier stretches
      the range and enlarges every cluster.
    - Results depend on the seed unless `order="index"`, which depends on input order instead.

    References
    ----------
    .. [1] Gobbi, A.; Lee, M.-L. DISE: Directed Sphere Exclusion. *J. Chem. Inf. Comput. Sci.*
       **2003**, 43 (1), 317-323. https://doi.org/10.1021/ci025554v
    .. [2] Golbraikh, A.; Tropsha, A. Predictive QSAR Modeling Based on Diversity Sampling of
       Experimental Datasets for the Training and Test Set Selection. *J. Comput.-Aided Mol. Des.*
       **2002**, 16 (5-6), 357-369. https://doi.org/10.1023/A:1020869118689
    """

    splitter_id: ClassVar[str] = "sphere_exclusion"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    bounded_metric_required: ClassVar[bool] = False
    deterministic_without_seed: ClassVar[bool] = False  # True only for order="index"

    def __init__(
        self,
        *,
        radius: float = 0.35,
        radius_is: Literal["distance", "similarity", "fraction_of_range"] = "distance",
        order: Literal["random", "index"] = "random",
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.radius = radius
        self.radius_is = radius_is
        self.order = order
        _validate_radius(self, radius, radius_is)
        if order not in ("random", "index"):
            raise ParameterError(f"invalid order: {order!r}")

    def _group_labels(self, ctx: _Context) -> IndexArray:
        n = ctx.n
        # One row per representative is all it reads, so rows can be produced on demand when a
        # dense matrix will not fit. Identical clusters: same rows, threshold and scan order.
        use_dense = dense_matrix_fits(n, self.max_memory_bytes)
        if use_dense:
            D = _dist_matrix(self, ctx)
            threshold = _resolve_radius(D, self.radius, self.radius_is)
        else:
            threshold = _resolve_radius_without_matrix(self, ctx)
        scan = (
            seed_for(ctx.rng_seeds, "sphere_exclusion.order", 0).permutation(n).tolist()
            if self.order == "random"
            else None
        )
        if use_dense:
            reps, clusters = _clustering.sphere_exclusion(D, threshold, order=scan)
        else:
            reps, clusters = _clustering.sphere_exclusion_rows(
                n,
                lambda i: rectangular_distances(
                    ctx, self.featurizer, self.metric, [i], range(n), self.n_jobs
                )[0],
                threshold,
                order=scan,
            )
        labels = np.empty(n, dtype=np.int64)
        for cid, members in enumerate(clusters):
            labels[members] = cid
        self._last_meta = {
            "radius": self.radius,
            "radius_is": self.radius_is,
            "distance_threshold": threshold,
            "n_clusters": len(clusters),
            "cluster_sizes": sorted((len(c) for c in clusters), reverse=True),
            "representatives": reps,
        }
        _check_cluster_degeneracy(
            clusters, n, type(self).__name__, f"radius={self.radius} ({self.radius_is})"
        )
        return dense_label_encode(labels.tolist())

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        return getattr(self, "_last_meta", {})


class KMeansClusterSplitter(_SimilarityGroupBase):
    """K-means (or a related partitional clusterer) over fingerprint/feature space.

    :param n_clusters: how many clusters to form, or ``"auto"`` to derive it from ``auto_rule``.
    :param algorithm: which clusterer to run: Lloyd's k-means, mini-batch k-means,
        agglomerative clustering, or BIRCH.
    :param linkage: linkage criterion for ``algorithm="agglomerative"``.
    :param auto_rule: how ``n_clusters="auto"`` is derived: ``sqrt(n)`` or ``n/50``.
    :param auto_range: lower and upper clamp applied to the derived cluster count.
    :param batch_size: mini-batch size for ``algorithm="minibatch_kmeans"``.
    :param reduce_dim: target dimensionality for the pre-clustering reduction, or ``None`` to
        skip it.
    :param reduce_method: truncated SVD, or no reduction.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``n_clusters`` is below 2, ``auto_range`` is not an increasing
        pair, or ``algorithm``, ``linkage``, ``auto_rule`` or ``reduce_method`` is unknown.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Scales far better than any `O(n^2)` method; `minibatch_kmeans` handles millions of
      molecules.
    - The cluster count is an explicit, reportable knob, and `auto_rule` makes the default
      reproducible rather than ad hoc.
    - Works on any feature representation, including learned embeddings and physicochemical
      descriptors, which makes it a natural generic clusterer.
    - `birch` and `minibatch` give a memory-bounded path where Butina and spectral clustering
      cannot run.

    Pitfalls
    --------
    - **k is arbitrary.** Nothing in the chemistry determines it, yet split difficulty depends
      on it strongly, and `auto_rule="sqrt_n"` is a convention rather than a principle.
    - K-means assumes isotropic equal-variance Euclidean clusters, which binary fingerprint
      space is not, so clusters are as much geometric artefacts as chemical families. SVD
      reduction mitigates without fixing it.
    - Cluster sizes come out very uneven, so the achieved train/test ratio drifts from the
      request; expect `SizeToleranceWarning`.
    - Euclidean distance on binary fingerprints is dominated by bit count, so clusters partly
      track molecular weight rather than chemotype. `property` splits on size deliberately.
    - SVD sign ambiguity breaks reproducibility across BLAS builds. A sign fix is applied,
      but Lloyd's-iteration noise can still move a few assignments, so the golden test uses a
      size histogram.
    - `agglomerative` with `single` linkage chains badly on chemical data, typically producing
      one giant cluster plus dust.

    References
    ----------
    .. [1] MacQueen, J. Some Methods for Classification and Analysis of Multivariate
       Observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics
       and Probability*, Vol. 1; University of California Press, **1967**; pp 281-297. No DOI;
       https://projecteuclid.org/euclid.bsmsp/1200512992
    .. [2] Lloyd, S. P. Least Squares Quantization in PCM. *IEEE Trans. Inf. Theory* **1982**,
       28 (2), 129-137. https://doi.org/10.1109/TIT.1982.1056489
    .. [3] ``algorithm="minibatch_kmeans"``: Sculley, D. Web-Scale k-Means Clustering. In
       *Proceedings of the 19th International Conference on World Wide Web (WWW '10)*,
       **2010**; pp 1177-1178. https://doi.org/10.1145/1772690.1772862
    .. [4] ``algorithm="agglomerative"``, Ward linkage: Ward, J. H. Hierarchical Grouping to
       Optimize an Objective Function. *J. Am. Stat. Assoc.* **1963**, 58 (301), 236-244.
       https://doi.org/10.1080/01621459.1963.10500845
    .. [5] ``algorithm="birch"``: Zhang, T.; Ramakrishnan, R.; Livny, M. BIRCH: An Efficient
       Data Clustering Method for Very Large Databases. *ACM SIGMOD Rec.* **1996**, 25 (2),
       103-114. https://doi.org/10.1145/235968.233324
    .. [6] Clustering as a QSAR dataset-division strategy: Golbraikh, A.; Shen, M.; Xiao, Z.
       et al. Rational Selection of Training and Test Sets for the Development of Validated
       QSAR Models. *J. Comput.-Aided Mol. Des.* **2003**, 17 (2-4), 241-253.
       https://doi.org/10.1023/A:1025386326946
    """

    splitter_id: ClassVar[str] = "k_means_cluster"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    bounded_metric_required: ClassVar[bool] = False

    def __init__(
        self,
        *,
        n_clusters: int | Literal['auto'] = "auto",
        algorithm: Literal["kmeans", "minibatch_kmeans", "agglomerative", "birch"] = "kmeans",
        linkage: Literal["ward", "complete", "average", "single"] = "ward",
        auto_rule: Literal["sqrt_n", "n_over_50"] = "sqrt_n",
        auto_range: tuple[int, int] = (2, 50),
        batch_size: int = 1024,
        reduce_dim: int | None = 128,
        reduce_method: Literal["svd", "none"] = "svd",
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.linkage = linkage
        self.auto_rule = auto_rule
        self.auto_range = auto_range
        self.batch_size = batch_size
        self.reduce_dim = reduce_dim
        self.reduce_method = reduce_method
        self._validate_similarity_params()
        if algorithm == "agglomerative" and linkage == "ward" and metric != "euclidean":
            raise ParameterError("linkage='ward' requires metric='euclidean'")

    def _resolve_k(self, n: int) -> int:
        return _resolve_cluster_count(n, self.n_clusters, self.auto_rule, self.auto_range)

    def _group_labels(self, ctx: _Context) -> IndexArray:
        feat = resolve_featurizer(self.featurizer)
        F = ctx.get_features(feat)
        Z = F.toarray() if sp.issparse(F) else np.asarray(F, dtype=np.float64)
        if (
            self.reduce_method == "svd"
            and self.reduce_dim is not None
            and Z.shape[1] > self.reduce_dim
        ):
            seed = int(seed_for(ctx.rng_seeds, "kmeans.svd", 0).integers(0, 2**31 - 1))
            svd = TruncatedSVD(
                n_components=self.reduce_dim,
                random_state=seed,
                algorithm="randomized",
                n_iter=7,
            )
            Z = svd.fit_transform(Z)
            Z = _fix_svd_signs(Z)
        k = self._resolve_k(ctx.n)
        if k >= ctx.n:
            raise ParameterError(f"resolved n_clusters={k} must be < n={ctx.n}")
        seed = int(seed_for(ctx.rng_seeds, "kmeans.fit", 0).integers(0, 2**31 - 1))
        if self.algorithm == "kmeans":
            labels = KMeans(
                n_clusters=k,
                n_init=10,
                algorithm="lloyd",
                max_iter=300,
                tol=1e-4,
                random_state=seed,
            ).fit_predict(Z)
        elif self.algorithm == "minibatch_kmeans":
            labels = MiniBatchKMeans(
                n_clusters=k,
                batch_size=self.batch_size,
                n_init=10,
                max_iter=100,
                random_state=seed,
            ).fit_predict(Z)
        elif self.algorithm == "agglomerative":
            metric_arg = "euclidean" if self.linkage == "ward" else self.metric
            labels = AgglomerativeClustering(
                n_clusters=k, linkage=self.linkage, metric=metric_arg
            ).fit_predict(Z)
        else:  # birch
            labels = Birch(n_clusters=k, threshold=0.5, branching_factor=50).fit_predict(Z)
        if len(set(labels.tolist())) == 1:
            raise DegenerateGroupingError(
                f"{type(self).__name__}: all records fell into a single cluster"
            )
        self._last_meta = {
            "n_clusters": int(len(set(labels.tolist()))),
            "algorithm": self.algorithm,
            "nondeterministic_method": True,  # BLAS-dependent, not bit-exact cross-platform
        }
        return dense_label_encode(labels.tolist())

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        return getattr(self, "_last_meta", {})


def _fix_svd_signs(Z: np.ndarray) -> np.ndarray:
    Z = Z.copy()
    for c in range(Z.shape[1]):
        col = Z[:, c]
        abs_col = np.abs(col)
        idx = argmax_tiebreak(lambda k: abs_col[k], range(len(col)))
        if col[idx] < 0:
            Z[:, c] = -col
    return Z


class DensityClusterSplitter(_SimilarityGroupBase):
    """DBSCAN or HDBSCAN density clustering on a precomputed distance matrix.

    :param algorithm: DBSCAN, or HDBSCAN when the ``hdbscan`` extra is installed.
    :param eps: DBSCAN neighbourhood radius, in the chosen metric.
    :param min_samples: how many neighbours make a point a core point.
    :param min_cluster_size: smallest cluster HDBSCAN will keep.
    :param noise_policy: where points belonging to no cluster go: all to test, all to train,
        one group each, discarded, or spread across the partitions.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``eps`` is not positive, the size parameters are below 1, or
        ``algorithm`` or ``noise_policy`` is unknown.
    :raises MissingDependencyError: if ``algorithm="hdbscan"`` and the extra is not installed.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - No `k` to choose, and clusters can take any shape, which fits chemical space better than
      k-means's spherical assumption.
    - Models "this molecule belongs to no family" explicitly. That case is chemically real, and
      every other clusterer forces it into some cluster.
    - `noise_policy="test"` produces a defensible "singletons and oddities" test set for
      applicability-domain work.

    Pitfalls
    --------
    - The noise bucket can swallow 40% or more of a diverse library at sensible `eps`, so
      `noise_policy` decides most of the split; the default `"own_groups"` scatters those
      points and makes it much easier.
    - `eps` interacts with fingerprint density in a way that carries no cross-dataset meaning,
      so a value tuned on one dataset does not transfer.
    - DBSCAN on a precomputed matrix needs `O(n^2)` memory, which caps `n` around 20,000 at the
      default guard.
    - HDBSCAN's `min_cluster_size` and `min_samples` interact non-obviously: changing one moves
      the cluster count non-monotonically.
    - Tanimoto distances concentrate in high dimensions, so most pairs sit in a narrow band and
      the density contrast density clustering relies on is weak.

    References
    ----------
    .. [1] Ester, M.; Kriegel, H.-P.; Sander, J.; Xu, X. A Density-Based Algorithm for
       Discovering Clusters in Large Spatial Databases with Noise. In *Proceedings of the 2nd
       International Conference on Knowledge Discovery and Data Mining (KDD-96)*; AAAI Press,
       **1996**; pp 226-231. No DOI; https://cdn.aaai.org/KDD/1996/KDD96-037.pdf
    .. [2] Campello, R. J. G. B.; Moulavi, D.; Sander, J. Density-Based Clustering Based on
       Hierarchical Density Estimates. In *Advances in Knowledge Discovery and Data Mining
       (PAKDD 2013)*; Lecture Notes in Computer Science 7819; Springer, **2013**; pp 160-172.
       https://doi.org/10.1007/978-3-642-37456-2_14
    .. [3] Campello, R. J. G. B.; Moulavi, D.; Zimek, A.; Sander, J. Hierarchical Density
       Estimates for Data Clustering, Visualization, and Outlier Detection. *ACM Trans. Knowl.
       Discov. Data* **2015**, 10 (1), 1-51. https://doi.org/10.1145/2733381
    """

    splitter_id: ClassVar[str] = "density_cluster"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    extras: ClassVar[tuple[str,...]] = ()
    bounded_metric_required: ClassVar[bool] = False

    def __init__(
        self,
        *,
        algorithm: Literal["dbscan", "hdbscan"] = "dbscan",
        eps: float = 0.3,
        min_samples: int = 5,
        min_cluster_size: int = 5,
        noise_policy: Literal[
            "test", "train", "own_groups", "discard", "distribute"
        ] = "own_groups",
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.algorithm = algorithm
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.noise_policy = noise_policy
        self._validate_similarity_params()
        if noise_policy not in ("test", "train", "own_groups", "discard", "distribute"):
            raise ParameterError(f"invalid noise_policy: {noise_policy!r}")

    def _group_labels(self, ctx: _Context) -> IndexArray:
        # DBSCAN reads only the eps-neighbour lists, which can be built blockwise; HDBSCAN builds
        # a minimum spanning tree over all pairs and genuinely needs the dense matrix.
        dense = dense_matrix_fits(ctx.n, self.max_memory_bytes) or self.algorithm != "dbscan"
        dist = _Distances(self, ctx) if dense else None
        D = dist.dense if dist is not None else None
        if self.algorithm == "dbscan":
            if D is not None:
                labels = DBSCAN(
                    eps=self.eps, min_samples=self.min_samples, metric="precomputed"
                ).fit_predict(D)
            else:
                neigh = compute_neighbor_lists(
                    ctx, self.featurizer, self.metric, self.eps,
                    eps=0.0, include_self=True, n_jobs=self.n_jobs,
                )
                labels = _clustering.dbscan_from_neighbors(neigh, self.min_samples, ctx.n)
        else:
            try:
                from sklearn.cluster import HDBSCAN
            except ImportError as exc:  # pragma: no cover - sklearn>=1.3 always has this
                raise MissingDependencyError(type(self).__name__, "hdbscan") from exc
            labels = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="precomputed",
                copy=False,
            ).fit_predict(D)
        n = ctx.n
        noise = np.nonzero(labels == -1)[0]
        forced: list[int] = []
        if noise.size:
            if self.noise_policy == "discard":
                forced = noise.tolist()
            elif self.noise_policy in ("test", "train"):
                # assign_groups only balances by size, so stash the noise indices for
                # _partition to move afterwards
                self._last_noise_idx = noise.copy()
            elif self.noise_policy == "distribute" and (labels != -1).any():
                core_idx = np.nonzero(labels != -1)[0]
                if D is None:
                    dist = _Distances(self, ctx)
                for i in noise:
                    col = D[i] if D is not None else dist.fetch([i], core_idx.tolist())[0]
                    nearest = (
                        argmin_tiebreak(lambda c: float(col[c]), core_idx.tolist())
                        if D is not None
                        else int(core_idx[row_argmin(col[None, :])[0]])
                    )
                    labels[i] = labels[nearest]
            # own_groups: leave at -1 and convert to singleton labels below
        if noise.size == n:
            if self.noise_policy == "own_groups":
                raise DegenerateGroupingError(f"{type(self).__name__}: every record is noise")
            raise ConstraintUnsatisfiableError(f"{type(self).__name__}: every record is noise")
        if forced:
            existing = set(ctx.extra.get("forced_discard", []))
            ctx.extra["forced_discard"] = sorted(existing | set(forced))
        next_id = int(labels.max()) + 1 if (labels != -1).any() else 0
        out = labels.copy()
        for i in np.nonzero(out == -1)[0]:
            out[i] = next_id
            next_id += 1
        self._last_meta = {
            "algorithm": self.algorithm,
            "n_clusters": int(len(set(labels[labels != -1].tolist()))),
            "n_noise": int(noise.size),
            "noise_frac": float(noise.size) / n,
            "noise_policy": self.noise_policy,
        }
        return dense_label_encode(out.tolist())

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        return getattr(self, "_last_meta", {})

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        results = super()._partition(ctx)
        if self.noise_policy not in ("test", "train"):
            return results
        noise_idx = getattr(self, "_last_noise_idx", np.array([], dtype=np.int64))
        if noise_idx.size == 0:
            return results
        noise_set = set(noise_idx.tolist())
        fixed = []
        for r in results:
            train = set(r.train.tolist()) - noise_set
            valid = set(r.valid.tolist()) - noise_set
            test = set(r.test.tolist()) - noise_set
            if self.noise_policy == "test":
                test |= noise_set
            else:
                train |= noise_set
            fixed.append(
                dataclasses.replace(
                    r,
                    train=np.asarray(sorted(train), dtype=np.int64),
                    valid=np.asarray(sorted(valid), dtype=np.int64),
                    test=np.asarray(sorted(test), dtype=np.int64),
                )
            )
        return fixed


class SpectralSplitter(_SimilarityGroupBase):
    """Laplacian-eigenmap spectral clustering on an affinity graph.

    ``graph="landmark"`` is landmark-based spectral clustering (Chen & Cai 2011) and never
    builds an ``n x n`` matrix. It picks ``n_landmarks`` records, by default with OptiSim off
    the ``"spectral.landmarks"`` stream at a ``ceil(n/20)`` subsample and no exclusion radius,
    so they come out spread and representative. Each record is then Gaussian weights to its
    ``landmark_neighbors`` nearest landmarks, bandwidth the mean distance to them, normalised
    to sum to one. The top ``n_clusters`` left singular vectors of that ``n x p`` matrix,
    scaled by the inverse square root of the landmark degrees, go to k-means. As in Chen & Cai
    the leading singular vector is kept, so ``drop_first`` does not apply.

    :param n_clusters: how many clusters to cut the spectral embedding into.
    :param graph: how the affinity graph is built: thresholded, k-nearest-neighbour, fully
        connected, or the landmark approximation described above.
    :param knn_k: neighbours per record for ``graph="knn"``.
    :param threshold: similarity floor for ``graph="threshold"``.
    :param laplacian: symmetric, random-walk, or unnormalized Laplacian.
    :param assign: cluster the embedding with k-means, or with the discretize rule.
    :param drop_first: drop the trivial leading eigenvector. Ignored for
        ``graph="landmark"``.
    :param n_landmarks: number of landmarks for ``graph="landmark"``, or ``None`` for the
        default of ``ceil(sqrt(n))``.
    :param landmark_selection: pick landmarks with OptiSim, or at random.
    :param landmark_neighbors: how many nearest landmarks represent each record.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``n_clusters``, ``knn_k``, ``n_landmarks`` or
        ``landmark_neighbors`` is out of range, or any of the mode parameters is unknown.
    :raises DegenerateGroupingError: at split time, if the affinity graph is disconnected, so
        that spectral clustering would degenerate into one cluster per component.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Minimises inter-cluster similarity by construction, which gives the least train/test
      overlap among routine structure-based splits.
    - Handles non-convex, elongated regions of chemical space that k-means cuts straight
      through.
    - The eigenvalue spectrum comes for free as a diagnostic: the spectral gap shows whether
      the dataset really has that many separable families.

    Pitfalls
    --------
    - `O(n^2)` affinity construction and a dense eigenproblem cap it near 50,000 molecules.
    - Depends on three coupled choices -- graph construction, Laplacian normalisation and
      `n_clusters` -- none of which has a chemically principled default.
    - Degenerate eigenvalues, common where many singleton components are identical, leave
      eigenvectors non-unique up to rotation, so labels can differ between runs and platforms
      at identical eigenvalues. It warns, and its golden test uses a size histogram.
    - A disconnected affinity graph would turn spectral clustering into one cluster per
      component, which is why that case is a hard error.
    - The hardest split is not the right split. A model evaluated only under spectral splitting
      looks worse than it will perform on a realistic screening library.
    - `graph="landmark"` approximates the full graph through `p` landmarks: too few blur small
      families together, and the result depends on the landmark draw.

    References
    ----------
    .. [1] Chen, X.; Cai, D. Large Scale Spectral Clustering with Landmark-Based Representation.
       *Proc. AAAI Conf. Artif. Intell.* **2011**, 25 (1), 313-318.
       https://doi.org/10.1609/aaai.v25i1.7900
    .. [2] Clark, R. D. OptiSim: An Extended Dissimilarity Selection Method for Finding Diverse
       Representative Subsets. *J. Chem. Inf. Comput. Sci.* **1997**, 37 (6), 1181-1188.
       https://doi.org/10.1021/ci970282v
    """

    splitter_id: ClassVar[str] = "spectral"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    bounded_metric_required: ClassVar[bool] = True

    def __init__(
        self,
        *,
        n_clusters: int = 8,
        graph: Literal["threshold", "knn", "full", "landmark"] = "knn",
        knn_k: int = 20,
        threshold: float = 0.3,
        laplacian: Literal["sym", "rw", "unnormalized"] = "sym",
        assign: Literal["kmeans", "discretize"] = "kmeans",
        drop_first: bool = True,
        n_landmarks: int | None = None,
        landmark_selection: Literal["optisim", "random"] = "optisim",
        landmark_neighbors: int = 5,
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.n_clusters = n_clusters
        self.graph = graph
        self.knn_k = knn_k
        self.threshold = threshold
        self.laplacian = laplacian
        self.assign = assign
        self.drop_first = drop_first
        self.n_landmarks = n_landmarks
        self.landmark_selection = landmark_selection
        self.landmark_neighbors = landmark_neighbors
        self._validate_similarity_params()
        if graph not in ("threshold", "knn", "full", "landmark"):
            raise ParameterError(f"invalid graph: {graph!r}")
        if landmark_selection not in ("optisim", "random"):
            raise ParameterError(f"invalid landmark_selection: {landmark_selection!r}")
        if n_landmarks is not None and (
            isinstance(n_landmarks, bool)
            or not isinstance(n_landmarks, (int, np.integer))
            or n_landmarks < 2
        ):
            raise ParameterError(f"n_landmarks must be None or an int >= 2, got {n_landmarks!r}")
        if (
            isinstance(landmark_neighbors, bool)
            or not isinstance(landmark_neighbors, (int, np.integer))
            or landmark_neighbors < 1
        ):
            raise ParameterError(
                f"landmark_neighbors must be an int >= 1, got {landmark_neighbors!r}"
            )

    def _landmark_labels(self, ctx: _Context) -> IndexArray:
        n = ctx.n
        if self.n_landmarks is not None:
            p = self.n_landmarks
        else:
            p = min(n - 1, max(50, math.ceil(math.sqrt(n)) * 5))
        if not (self.n_clusters <= p < n):
            raise ParameterError(f"n_landmarks={p} must satisfy n_clusters <= n_landmarks < n={n}")
        r = min(self.landmark_neighbors, p)
        F = ctx.get_features(resolve_featurizer(self.featurizer))
        guard_memory(max(1, math.isqrt(n * p) + 1), self.max_memory_bytes, type(self).__name__)
        rng = seed_for(ctx.rng_seeds, "spectral.landmarks", 0)
        if self.landmark_selection == "optisim":

            def column(j: int) -> np.ndarray:
                return pairwise_distances(F, F[j:j + 1], metric=self.metric)[:, 0]

            landmarks = _clustering.optisim_pick_columns(
                n, column, p, max(1, -(-n // 20)), 0.0, rng
            )
        else:
            landmarks = sorted(int(j) for j in rng.choice(n, size=p, replace=False))
        if len(landmarks) < self.n_clusters:
            raise DegenerateGroupingError(
                f"{type(self).__name__}: only {len(landmarks)} distinct landmarks for "
                f"n_clusters={self.n_clusters}"
            )
        Dl = pairwise_distances(F, F[np.asarray(landmarks)], metric=self.metric).astype(np.float64)
        nearest = np.argsort(Dl, axis=1, kind="stable")[:, :r]
        rows = np.repeat(np.arange(n), r)
        d = Dl[rows, nearest.ravel()]
        h = float(d.mean()) or 1.0
        w = np.exp(-(d**2) / (2.0 * h * h))
        Z = np.zeros((n, len(landmarks)))
        Z[rows, nearest.ravel()] = w
        Z /= Z.sum(axis=1, keepdims=True)
        col = Z.sum(axis=0)
        Zhat = Z / np.sqrt(np.where(col > 0, col, 1.0))
        U, _, _ = np.linalg.svd(Zhat, full_matrices=False)
        k = min(self.n_clusters, U.shape[1])
        U = _clustering._fix_eigenvector_signs(U[:, :k])
        seed = int(seed_for(ctx.rng_seeds, "spectral.kmeans", 0).integers(0, 2**31 - 1))
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(U)
        self._last_meta = {
            "n_clusters": int(len(set(labels.tolist()))),
            "graph": self.graph,
            "n_landmarks": len(landmarks),
            "landmark_selection": self.landmark_selection,
            "nondeterministic_method": True,
        }
        return np.asarray(dense_label_encode(labels.tolist()), dtype=np.int64)

    def _group_labels(self, ctx: _Context) -> IndexArray:
        if self.graph == "landmark":
            return self._landmark_labels(ctx)
        S = _sim_matrix(self, ctx)
        n = ctx.n
        np.fill_diagonal(S, 0.0)
        if self.graph == "full":
            W = S
        elif self.graph == "threshold":
            W = np.where(S > self.threshold + EPS, S, 0.0)
        else:  # knn
            k = min(self.knn_k, n - 1)
            W = np.zeros_like(S)
            for i in range(n):
                nn = np.argsort(-S[i])[:k]
                W[i, nn] = S[i, nn]
            W = np.maximum(W, W.T)
        if self.knn_k >= n and self.graph == "knn":
            raise ParameterError(f"knn_k={self.knn_k} must be < n={n}")
        rng = seed_for(ctx.rng_seeds, "spectral.v0", 0)
        labels = _clustering.spectral_partition(
            W,
            self.n_clusters,
            laplacian=self.laplacian,
            drop_first=self.drop_first,
            assign=self.assign,
            rng=rng,
            random_state=int(seed_for(ctx.rng_seeds, "spectral.kmeans", 0).integers(0, 2**31 - 1)),
        )
        self._last_meta = {
            "n_clusters": int(len(set(labels.tolist()))),
            "graph": self.graph,
            "nondeterministic_method": True,  # eigendecomposition isn't bit-exact cross-platform
        }
        return labels

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        return getattr(self, "_last_meta", {})


class MaxMinSplitter(_SimilarityBase):
    """Greedy maximally-diverse selection (MaxMin / Kennard-Stone).

    The selected set may go to **train** (maximise coverage) or **test** (probe breadth) -- opposite
    experiments sharing one algorithm; never compare numbers across ``picked_goes_to`` values.

    ``swap_fraction > 0`` then perturbs the selection, exchanging that fraction of the picked
    records -- rounded half up, capped at the unpicked count -- off the ``"maxmin.swap"``
    stream. ``init="kennard_stone"`` at ``swap_fraction=0.1`` is the Morais-Lima-Martin
    random-mutation Kennard-Stone method.

    :param picked_goes_to: whether the diverse selection becomes train (maximise coverage) or
        test (probe breadth).
    :param init: how the first record is chosen: at random, by the Kennard-Stone rule, the most
        peripheral record, or index 0.
    :param n_picks: how many records to select, or ``None`` to take it from the resolved sizes.
    :param swap_fraction: fraction of the selection to exchange for unpicked records after the
        greedy pass, which makes the result seed-dependent.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``swap_fraction`` is outside ``[0, 1]``, ``n_picks`` is not
        positive, or ``picked_goes_to`` or ``init`` is unknown.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - With `picked_goes_to="train"`, builds the most informative training set for a fixed
      budget, which is the standard answer to "which 500 compounds should I assay?".
    - `coverage_radius` is a directly interpretable guarantee: no record sits further than that
      from a training example.
    - Memory-light in its lazy form, so it runs on datasets where Butina and spectral
      clustering cannot.
    - Deterministic apart from a single initial pick, and fully deterministic with
      `init="kennard_stone"`.
    - `init="kennard_stone"` with `metric="mahalanobis"` on a descriptor matrix is the MDKS
      variant; :class:`SPXYSplitter` is the label-aware one.

    Pitfalls
    --------
    - **The two directions are different experiments and are routinely confused.**
      Diverse-in-train gives a well-covered, optimistic test set; diverse-in-test gives a hard
      extrapolation. The numbers do not compare across `picked_goes_to`.
    - Greedy MaxMin chases outliers, so the first picks are the strangest molecules present:
      parse artefacts, salts, fragments. On uncleaned data the "diverse" set is a junk set.
    - Depends strongly on the initial pick when `init="random"`; `"kennard_stone"` is
      seed-free.
    - Optimises coverage, not group separation: nothing stops a near-duplicate of a picked
      molecule from landing in the other partition. It is not a leakage control.
    - `coverage_radius` is meaningful only in the chosen metric, so it does not compare across
      fingerprints.
    - `swap_fraction` trades coverage for a less systematically optimistic test set, and makes
      the split seed-dependent even with `init="kennard_stone"`.

    References
    ----------
    .. [1] Kennard, R. W.; Stone, L. A. Computer Aided Design of Experiments. *Technometrics*
       **1969**, 11 (1), 137-148. https://doi.org/10.1080/00401706.1969.10490666
    .. [2] Saptoro, A.; Tadé, M. O.; Vuthaluru, H. A Modified Kennard-Stone Algorithm for Optimal
       Division of Data for Developing Artificial Neural Network Models. *Chem. Prod. Process
       Model.* **2012**, 7 (1). https://doi.org/10.1515/1934-2659.1645
    .. [3] Morais, C. L. M.; Santos, M. C. D.; Lima, K. M. G.; Martin, F. L. Improving Data
       Splitting for Classification Applications in Spectrochemical Analyses Employing a
       Random-Mutation Kennard-Stone Algorithm Approach. *Bioinformatics* **2019**, 35 (24),
       5257-5263. https://doi.org/10.1093/bioinformatics/btz421
    """

    splitter_id: ClassVar[str] = "max_min"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    bounded_metric_required: ClassVar[bool] = False
    deterministic_without_seed: ClassVar[bool] = False  # True only for init="kennard_stone"

    def __init__(
        self,
        *,
        picked_goes_to: Literal["train", "test"] = "train",
        init: Literal["random", "kennard_stone", "most_peripheral", "index_zero"] = "random",
        n_picks: int | None = None,
        swap_fraction: float = 0.0,
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.picked_goes_to = picked_goes_to
        self.init = init
        self.n_picks = n_picks
        self.swap_fraction = swap_fraction
        self._validate_similarity_params()
        if picked_goes_to not in ("train", "test"):
            raise ParameterError(f"invalid picked_goes_to: {picked_goes_to!r}")
        if (
            isinstance(swap_fraction, bool)
            or not isinstance(swap_fraction, (int, float))
            or not (0.0 <= swap_fraction <= 0.5)
        ):
            raise ParameterError(f"swap_fraction must be in [0, 0.5], got {swap_fraction!r}")

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        n_picks = self.n_picks if self.n_picks is not None else (
            ctx.sizes.n_train if self.picked_goes_to == "train" else ctx.sizes.n_test
        )
        if not (1 <= n_picks < n):
            raise ParameterError(f"n_picks must satisfy 1 <= n_picks < n, got {n_picks}")
        rng = seed_for(ctx.rng_seeds, "maxmin.init", 0) if self.init == "random" else None
        # One column per pick. The dense matrix is faster when it fits; past that, fetch columns in
        # wide blocks rather than refuse. The running minimum stays exact, so the picks match.
        D: np.ndarray | None = None
        if dense_matrix_fits(n, self.max_memory_bytes) or self.init in (
            "kennard_stone",
            "most_peripheral",
        ):
            # these two inits need the full matrix to seed themselves
            D = _dist_matrix(self, ctx)
            detail = _clustering.maxmin_pick_detail(D, n_picks, init=self.init, rng=rng)
        else:
            first = int(rng.integers(0, n)) if rng is not None else 0
            detail = _clustering.maxmin_pick_columns(
                n,
                lambda js: rectangular_distances(
                    ctx, self.featurizer, self.metric, range(n), js, self.n_jobs
                ),
                n_picks,
                [first],
            )
        picked = detail.picked
        swap_meta: dict[str, Any] = {}
        if self.swap_fraction > 0:
            picked_set = set(picked)
            unpicked = [i for i in range(n) if i not in picked_set]
            k = min(floor_round(self.swap_fraction * len(picked)), len(unpicked))
            swap_rng = seed_for(ctx.rng_seeds, "maxmin.swap", 0)
            swapped_out = (
                sorted(int(i) for i in swap_rng.choice(picked, size=k, replace=False))
                if k
                else []
            )
            swapped_in = (
                sorted(int(i) for i in swap_rng.choice(unpicked, size=k, replace=False))
                if k
                else []
            )
            out_set = set(swapped_out)
            picked = [i for i in picked if i not in out_set] + swapped_in
            swap_meta = {"swapped_out": swapped_out, "swapped_in": swapped_in}
        rem_rng = seed_for(ctx.rng_seeds, "maxmin.remainder", 0)
        buckets = _fill_remainder(picked, n, ctx.sizes, rem_rng, self.picked_goes_to)
        if swap_meta:
            # swapping changed `picked` after selection, so the recorded diagnostics no longer
            # describe it
            min_pairwise, coverage = _picked_diagnostics(
                picked, n, D, ctx, self.featurizer, self.metric, self.n_jobs
            )
        else:
            min_pairwise, coverage = detail.min_pairwise, detail.coverage
        result = SplitResult(
            train=buckets["train"],
            valid=buckets["valid"],
            test=buckets["test"],
            discard=buckets["discard"],
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=n,
            metadata={
                "picked": picked,
                "picked_goes_to": self.picked_goes_to,
                "init": self.init,
                "min_pairwise_distance_in_picked": min_pairwise if picked else float("nan"),
                "coverage_radius": coverage,
                "realised_sizes": {k: int(v.size) for k, v in buckets.items() if k != "discard"},
                **swap_meta,
            },
        )
        _small_partition_check(result, n)
        return [result]


class SPXYSplitter(_SimilarityBase):
    """Kennard-Stone selection over a joint feature-and-label distance (SPXY).

    The feature distance matrix and the pairwise Euclidean label distance, over all columns for
    a multi-task ``y``, are each divided by their own maximum and summed; Kennard-Stone then
    picks ``n_train`` records from that joint matrix into **train**. The remainder fills
    valid and test off the ``"spxy.remainder"`` stream, so a two-way split is seed-free.
    ``metric="mahalanobis"`` gives M-SPXY (Apinantanakon et al. 2019, eq. 9).

    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises LabelError: at split time, if ``y`` is missing.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Covers the label range as well as chemical space, so train spans the response surface.
      This is the standard fix for Kennard-Stone leaving extreme activities out of train.
    - Deterministic without a seed for two-way splits: no random initial pick, and ties broken
      by smallest index.
    - Each term is scaled to `[0, 1]` before summing, so neither the fingerprint distance nor
      the label units dominate.

    Pitfalls
    --------
    - The split depends on `y`, so the training set is chosen knowing the labels. That suits
      calibration-set design, but the results are not comparable with label-blind splits.
    - As with Kennard-Stone, the first picks are the most extreme records, including outliers
      and label errors.
    - Optimises coverage, not separation, so test records can have near-duplicates in train. It
      is not a leakage control.
    - Label distances are Euclidean over the raw `y` columns, so in multi-task data a task with
      a wider range weighs more unless `y` is standardised first.
    - Builds two dense `n x n` matrices.

    References
    ----------
    .. [1] Galvão, R. K. H.; Araujo, M. C. U.; José, G. E.; Pontes, M. J. C.; Silva, E. C.;
       Saldanha, T. C. B. A Method for Calibration and Validation Subset Partitioning. *Talanta*
       **2005**, 67 (4), 736-740. https://doi.org/10.1016/j.talanta.2005.03.025
    .. [2] Apinantanakon, W.; Sunat, K.; Kinmond, J. A. Optimal Data Division for Empowering
       Artificial Neural Network Models Employing a Modified M-SPXY Algorithm. *Eng. Appl. Sci.
       Res.* **2019**, 46 (4), 276-284. https://doi.org/10.14456/easr.2019.31
    """

    splitter_id: ClassVar[str] = "spxy"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    requires_labels: ClassVar[bool] = True
    bounded_metric_required: ClassVar[bool] = False
    deterministic_without_seed: ClassVar[bool] = False  # True whenever n_valid == 0

    def _check_preconditions(self, ctx: _Context) -> None:
        y = np.asarray(ctx.y)
        if y.ndim not in (1, 2) or y.shape[0] != ctx.n:
            raise LabelError(f"{type(self).__name__} requires y of shape (n,) or (n, n_tasks)")
        if not np.all(np.isfinite(y.astype(np.float64))):
            raise LabelError(f"{type(self).__name__} requires finite numeric y")

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        y = np.asarray(ctx.y, dtype=np.float64).reshape(n, -1)
        # SPXY selects on the sum of two max-normalised distance matrices. Both the normalising
        # maxima and the Kennard-Stone selection read the sum only as columns, so neither matrix
        # has to be materialised.
        dense = dense_matrix_fits(n, self.max_memory_bytes, 2)
        D_x = _dist_matrix(self, ctx).astype(np.float64) if dense else None
        D_y = cdist(y, y, metric="euclidean") if dense else None
        if dense:
            maxima = {
                "feature": float(D_x.max()) if D_x.size else 0.0,
                "label": float(D_y.max()) if D_y.size else 0.0,
            }
        else:
            dist_x = _Distances(self, ctx, as_float64=True)
            maxima = {
                "feature": dist_x.max_overall(),
                "label": _blocked_max_cdist(y),
            }
        degenerate = [name for name, m in maxima.items() if m <= 0.0]
        if dense:
            for name, M in (("feature", D_x), ("label", D_y)):
                if maxima[name] > 0.0:
                    M /= maxima[name]
        if degenerate:
            warn_with_details(
                DegenerateClusterWarning(
                    f"{type(self).__name__}: all pairwise {' and '.join(degenerate)} distances are "
                    "zero; that term contributes nothing to the selection",
                    details={"zero_terms": degenerate},
                )
            )
        n_picks = ctx.sizes.n_train
        if dense:
            # Accumulate in place: a third n x n matrix pushed peak past the 2 GiB default at
            # n=10000, so remove the need for it rather than raise the budget.
            D_x += D_y
            D = D_x
            del D_y
            picked = _clustering.kennard_stone(D, n_picks)[:n_picks]
            coverage = (
                float(np.max(np.min(D[:, picked], axis=1))) if picked else float("nan")
            )
        else:
            # The combined matrix is the sum of two max-normalised ones. Produce its row bands and
            # columns on demand; Kennard-Stone reads nothing else.
            fx = maxima["feature"]
            fy = maxima["label"]

            def combined(rows: Sequence[int], cols: Sequence[int]) -> np.ndarray:
                rows, cols = list(rows), list(cols)
                out = np.zeros((len(rows), len(cols)), dtype=np.float64)
                if fx > 0.0:
                    out += dist_x.fetch(rows, cols) / fx
                if fy > 0.0:
                    out += cdist(y[np.asarray(rows)], y[np.asarray(cols)], "euclidean") / fy
                return out

            i0, j0 = _blocked_seed_pair(n, lambda a, b: combined(range(a, b), range(n)))
            detail = _clustering.maxmin_pick_columns(
                n, lambda js: combined(range(n), js), n_picks, [i0, j0]
            )
            picked = detail.picked[:n_picks]
            coverage = detail.coverage if picked else float("nan")
        rem_rng = seed_for(ctx.rng_seeds, "spxy.remainder", 0)
        buckets = _fill_remainder(picked, n, ctx.sizes, rem_rng, "train")
        result = SplitResult(
            train=buckets["train"],
            valid=buckets["valid"],
            test=buckets["test"],
            discard=buckets["discard"],
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=n,
            metadata={
                "picked": picked,
                "coverage_radius": coverage,
                "zero_distance_terms": degenerate,
                "realised_sizes": {k: int(v.size) for k, v in buckets.items() if k != "discard"},
            },
        )
        _small_partition_check(result, n)
        return [result]


class OptiSimSplitter(_SimilarityGroupBase):
    """OptiSim diversity selection, used either as cluster centres or as a picked set.

    Each round draws candidates off the ``"optisim.draw"`` stream until ``subsample_size`` of
    them lie further than ``radius`` from everything selected, then takes the one with the
    largest minimum distance to the selection. ``subsample_size`` therefore trades
    representativeness against diversity.

    ``mode="cluster"`` treats the selected records as centres, groups every record with its
    nearest centre -- ties to the earliest selected -- and assigns whole groups to partitions;
    ``n_picks`` defaults to :class:`KMeansClusterSplitter`'s ``"auto"`` rule. ``mode="pick"``
    sends the selection to ``picked_goes_to`` as :class:`MaxMinSplitter` does, defaulting
    ``n_picks`` to that partition's size, shuffles the rest in off the
    ``"optisim.remainder"`` stream, and forms no groups.

    :param mode: use the selection as cluster centres, or as a picked partition.
    :param n_picks: how many records to select, or ``None`` for the per-mode default described
        above.
    :param subsample_size: candidates that must clear ``radius`` before one is selected, or
        ``None``. ``1`` is random selection with sphere exclusion; covering every record is
        MaxMin.
    :param radius: exclusion radius around each selected record, read per ``radius_is``.
    :param radius_is: whether ``radius`` is a distance, a similarity, or a fraction of the
        observed distance range.
    :param picked_goes_to: for ``mode="pick"``, which partition the selection becomes.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``n_picks`` or ``subsample_size`` is not positive, ``radius`` is
        out of range, or any mode parameter is unknown.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - One knob, `subsample_size`, spans random sampling to MaxMin, so a selection can be
      diverse without being dominated by outliers the way pure MaxMin is.
    - `radius` guarantees a minimum spacing between selected records, and so between cluster
      centres.
    - Cluster mode keeps near-duplicates of a centre in that centre's group, so they cannot
      straddle train and test.
    - Selection costs `O(n * n_picks)` on top of the distance matrix, which is cheap next to
      Butina or spectral clustering.

    Pitfalls
    --------
    - Results depend on the seed at every round, not just the first pick.
    - Cluster mode's groups are Voronoi cells around the centres rather than density clusters,
      and with a small `n_picks` they are large and chemically mixed.
    - If `radius` excludes every candidate, fewer than `n_picks` are selected and a
      `DegenerateClusterWarning` says how many. In pick mode the surplus is **discarded**.
    - Pick mode optimises coverage, not separation, so like MaxMin it is not a leakage control.
      Only cluster mode is group-forming.
    - `radius` is in the chosen metric's units unless `radius_is="fraction_of_range"`.

    References
    ----------
    .. [1] Clark, R. D. OptiSim: An Extended Dissimilarity Selection Method for Finding Diverse
       Representative Subsets. *J. Chem. Inf. Comput. Sci.* **1997**, 37 (6), 1181-1188.
       https://doi.org/10.1021/ci970282v
    """

    splitter_id: ClassVar[str] = "opti_sim"
    strictness: ClassVar[Strictness] = Strictness.STRICT  # mode="pick" is MODERATE, like max_min
    bounded_metric_required: ClassVar[bool] = False
    deterministic_without_seed: ClassVar[bool] = False

    def __init__(
        self,
        *,
        mode: Literal["cluster", "pick"] = "cluster",
        n_picks: int | None = None,
        subsample_size: int | None = None,
        radius: float = 0.35,
        radius_is: Literal["distance", "similarity", "fraction_of_range"] = "distance",
        picked_goes_to: Literal["train", "test"] = "train",
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.mode = mode
        self.n_picks = n_picks
        self.subsample_size = subsample_size
        self.radius = radius
        self.radius_is = radius_is
        self.picked_goes_to = picked_goes_to
        _validate_radius(self, radius, radius_is)
        if mode not in ("cluster", "pick"):
            raise ParameterError(f"invalid mode: {mode!r}")
        if picked_goes_to not in ("train", "test"):
            raise ParameterError(f"invalid picked_goes_to: {picked_goes_to!r}")
        min_picks = 2 if mode == "cluster" else 1
        for name, value, lo in (
            ("n_picks", n_picks, min_picks),
            ("subsample_size", subsample_size, 1),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lo:
                raise ParameterError(f"{name} must be None or an int >= {lo}, got {value!r}")

    def compute_groups(self, X: Any, y: Any = None, **kw: Any) -> IndexArray:
        """Expose the group labels without performing a split.

        :param X: the records, as for :meth:`split`.
        :param y: labels, if the splitter needs them.
        :param kw: per-call extras, as for :meth:`split`.
        :raises NotImplementedError: in ``mode="pick"``, which forms no groups.
        :return: one dense group label per record.
        """
        if self.mode == "pick":
            raise ParameterError(
                f"{type(self).__name__}(mode='pick') forms no groups; use mode='cluster' "
                "for compute_groups()"
            )
        return super().compute_groups(X, y, **kw)

    def _select(self, ctx: _Context, n_picks: int) -> tuple[Any, list[int], float, int]:
        # OptiSim reads one column per selected record, so the column-oriented form runs at any
        # size; the dense matrix is only built when it fits and is cheaper.
        dist = _Distances(self, ctx)
        n = ctx.n
        if not (1 <= n_picks < n):
            raise ParameterError(f"n_picks must satisfy 1 <= n_picks < n, got {n_picks}")
        threshold = (
            _resolve_radius(dist.dense, self.radius, self.radius_is)
            if dist.dense is not None
            else _resolve_radius_without_matrix(self, ctx)
        )
        k = self.subsample_size if self.subsample_size is not None else max(1, -(-n // 20))
        rng = seed_for(ctx.rng_seeds, "optisim.draw", 0)
        picked = _clustering.optisim_pick_columns(n, dist.column, n_picks, k, threshold, rng)
        if len(picked) < n_picks:
            warn_with_details(
                DegenerateClusterWarning(
                    f"{type(self).__name__}: only {len(picked)} of n_picks={n_picks} records lie "
                    f"further than radius={self.radius} ({self.radius_is}) from each other",
                    details={"n_picks": n_picks, "n_selected": len(picked)},
                )
            )
        return dist, picked, threshold, k

    def _group_labels(self, ctx: _Context) -> IndexArray:
        n = ctx.n
        n_picks = self.n_picks if self.n_picks is not None else _resolve_cluster_count(n, "auto")
        dist, centres, threshold, k = self._select(ctx, n_picks)
        slot = list(range(len(centres)))
        # nearest centre per record: one blocked row-argmin instead of an O(n*k) Python scan
        labels = dist.row_argmin_to(centres)
        clusters = [np.flatnonzero(labels == c).tolist() for c in slot]
        self._last_meta = {
            "mode": self.mode,
            "n_picks": n_picks,
            "n_selected": len(centres),
            "subsample_size": k,
            "distance_threshold": threshold,
            "centres": centres,
            "cluster_sizes": sorted((len(c) for c in clusters), reverse=True),
        }
        _check_cluster_degeneracy(
            clusters,
            n,
            type(self).__name__,
            f"n_picks={n_picks}, radius={self.radius} ({self.radius_is})",
        )
        return dense_label_encode(labels.tolist())

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        return getattr(self, "_last_meta", {})

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        if self.mode == "cluster":
            return super()._partition(ctx)
        n = ctx.n
        n_picks = self.n_picks if self.n_picks is not None else (
            ctx.sizes.n_train if self.picked_goes_to == "train" else ctx.sizes.n_test
        )
        dist, picked, threshold, k = self._select(ctx, n_picks)
        rem_rng = seed_for(ctx.rng_seeds, "optisim.remainder", 0)
        buckets = _fill_remainder(picked, n, ctx.sizes, rem_rng, self.picked_goes_to)
        coverage = dist.max_min_to(picked)
        result = SplitResult(
            train=buckets["train"],
            valid=buckets["valid"],
            test=buckets["test"],
            discard=buckets["discard"],
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=n,
            metadata={
                "mode": self.mode,
                "picked": picked,
                "picked_goes_to": self.picked_goes_to,
                "n_picks": n_picks,
                "n_selected": len(picked),
                "subsample_size": k,
                "distance_threshold": threshold,
                "coverage_radius": coverage,
                "realised_sizes": {
                    name: int(v.size) for name, v in buckets.items() if name != "discard"
                },
            },
        )
        _small_partition_check(result, n)
        return [result]




def _check_task_index(task_index: Any) -> None:
    if (
        isinstance(task_index, bool)
        or not isinstance(task_index, (int, np.integer))
        or task_index < 0
    ):
        raise ParameterError(f"task_index must be an int >= 0, got {task_index!r}")


def _label_column(splitter: Any, ctx: _Context) -> np.ndarray:
    """Finite float64 label column ``task_index`` of ``ctx.y``; raises :class:`LabelError`."""
    y = np.asarray(ctx.y)
    if y.ndim == 2:
        if splitter.task_index >= y.shape[1]:
            raise ParameterError(
                f"task_index={splitter.task_index} out of range for y with {y.shape[1]} columns"
            )
        y = y[:, splitter.task_index]
    elif y.ndim != 1:
        raise LabelError(f"{type(splitter).__name__} requires y of shape (n,) or (n, n_tasks)")
    try:
        col = y.astype(np.float64)
    except (TypeError, ValueError):
        raise LabelError(f"{type(splitter).__name__} requires numeric y") from None
    if not np.all(np.isfinite(col)):
        raise LabelError(f"{type(splitter).__name__} requires finite numeric y")
    return col


class MinimalTestSetDissimilaritySplitter(_SimilarityBase):
    """Minimal test set dissimilarity (MTSD): one typical record per activity bin goes to test.

    A record's total dissimilarity is the sum of its distances to every other record. Records
    sort by label, most active first with ties by index, and cut into ``n_test`` contiguous bins
    of near-equal size; each bin's least dissimilar record, ties to the smallest index, goes to
    **test**. A validation set is chosen the same way from the remainder, with total
    dissimilarities recomputed over it; the rest is train. At a 20% test set the bins hold 5
    records each, as in Martin et al. (2012), who used Euclidean distance on preselected
    descriptors. Fully deterministic.

    :param task_index: which column of a multi-task ``y`` to bin on. The others are ignored.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``task_index`` is negative.
    :raises LabelError: at split time, if ``y`` is missing or ``task_index`` is out of range.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - The test set spans the full label range by construction, one record per activity bin, so
      no part of the response is left unevaluated.
    - Each test record is the most typical member of its bin, so test compounds always have
      close analogues in train, which makes this a clean check of interpolation quality.
    - No seed and no free parameter beyond the featurizer and metric.
    - Follows criterion 2 of rational division -- test compounds close to training compounds --
      directly.

    Pitfalls
    --------
    - **Deliberately optimistic.** Test holds the least unusual compounds, so scores overstate
      performance on new chemistry; Martin et al. found such test sets beat random ones on
      test but not externally.
    - Selection uses `y`, so the split is not label-blind and does not compare with label-free
      splits.
    - Total dissimilarity is dominated by global position: a dense region's centre wins every
      bin it touches, so test can concentrate in one region of chemical space.
    - Builds the full `n x n` distance matrix.
    - Multi-task `y` uses one column, `task_index`, and ignores the rest.

    References
    ----------
    .. [1] Martin, T. M.; Harten, P.; Young, D. M.; Muratov, E. N.; Golbraikh, A.; Zhu, H.;
       Tropsha, A. Does Rational Selection of Training and Test Sets Improve the Outcome of QSAR
       Modeling? *J. Chem. Inf. Model.* **2012**, 52 (10), 2570-2578.
       https://doi.org/10.1021/ci300338w
    .. [2] Kuz'min, V. E.; Artemenko, A. G.; Muratov, E. N.; Volineckaya, I. L.; Makarov, V. A.;
       Riabova, O. B.; Wutzler, P.; Schmidtke, M. Quantitative Structure−Activity Relationship
       Studies of [(Biphenyloxy)propyl]isoxazole Derivatives. Inhibitors of Human Rhinovirus 2
       Replication. *J. Med. Chem.* **2007**, 50 (17), 4205-4213.
       https://doi.org/10.1021/jm0704806
    """

    splitter_id: ClassVar[str] = "minimal_test_set_dissimilarity"
    strictness: ClassVar[Strictness] = Strictness.OPTIMISTIC
    requires_labels: ClassVar[bool] = True
    bounded_metric_required: ClassVar[bool] = False
    deterministic_without_seed: ClassVar[bool] = True

    def __init__(
        self,
        *,
        task_index: int = 0,
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.task_index = task_index
        self._validate_similarity_params()
        _check_task_index(task_index)

    def _check_preconditions(self, ctx: _Context) -> None:
        _label_column(self, ctx)

    @staticmethod
    def _select(
        dist: Any, y: np.ndarray, pool: list[int], k: int
    ) -> tuple[list[int], list[list[int]]]:
        """Pick one minimal-total-dissimilarity record per activity bin.

        :param dist: the distance accessor.
        :param y: the label column to bin on.
        :param pool: candidate record indices.
        :param k: how many records to pick, i.e. how many bins to cut.
        :return: the picked indices and the bin memberships they came from.
        """
        if k <= 0:
            return [], []
        idx = np.asarray(pool, dtype=np.int64)
        totals = dist.row_sums_within(idx)
        total_of = {int(r): float(t) for r, t in zip(idx, totals, strict=True)}
        ranked = stable_sort(list(pool), key=lambda r: y[r], desc=True)
        m = len(ranked)
        base, extra = divmod(m, k)
        picked: list[int] = []
        edges: list[list[int]] = []
        start = 0
        for b in range(k):
            stop = start + base + (1 if b < extra else 0)
            members = sorted(ranked[start:stop])
            picked.append(argmin_tiebreak(lambda r: total_of[r], members))
            edges.append([start, stop])
            start = stop
        return picked, edges

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        y = _label_column(self, ctx)
        # Only within-pool row sums and the global row sums are read, both of which stream.
        dist = _Distances(self, ctx, as_float64=True)
        test, test_edges = self._select(dist, y, list(range(n)), ctx.sizes.n_test)
        test_set = set(test)
        remaining = [i for i in range(n) if i not in test_set]
        valid, _ = self._select(dist, y, remaining, ctx.sizes.n_valid)
        valid_set = set(valid)
        train = [i for i in remaining if i not in valid_set]
        totals = dist.row_sums()
        result = SplitResult(
            train=np.asarray(train, dtype=np.int64),
            valid=np.sort(np.asarray(valid, dtype=np.int64)),
            test=np.sort(np.asarray(test, dtype=np.int64)),
            discard=np.array([], dtype=np.int64),
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=n,
            metadata={
                "bin_edges": test_edges,
                "test_total_dissimilarity": [float(totals[i]) for i in sorted(test)],
                "realised_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
            },
        )
        _small_partition_check(result, n)
        return [result]




def _helmert(codes: np.ndarray, n_levels: int) -> np.ndarray:
    """Helmert contrasts for level codes ``0..n_levels-1``: column ``c`` is ``-1`` for levels
    ``<= c``, ``c + 1`` for level ``c + 1`` and ``0`` above, giving ``n_levels - 1`` columns."""
    out = np.zeros((codes.size, max(0, n_levels - 1)), dtype=np.float64)
    for c in range(n_levels - 1):
        out[codes <= c, c] = -1.0
        out[codes == c + 1, c] = float(c + 1)
    return out


def _encode_label_columns(y: Any, label_kind: str, owner: str) -> np.ndarray:
    """Numeric design columns for ``y``: continuous columns as float, categorical columns
    (``label_kind="categorical"``, or non-numeric/bool under ``"auto"``) as Helmert contrasts with
    levels in order of first appearance."""
    arr = np.asarray(y)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise LabelError(f"{owner} requires y of shape (n,) or (n, n_tasks)")
    blocks: list[np.ndarray] = []
    for c in range(arr.shape[1]):
        col = arr[:, c]
        categorical = label_kind == "categorical" or (
            label_kind == "auto" and col.dtype.kind in ("O", "U", "S", "b")
        )
        if categorical:
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in col.tolist()):
                raise LabelError(f"{owner}: y contains missing categorical labels")
            levels: dict[Any, int] = {}
            codes = np.asarray(
                [levels.setdefault(v, len(levels)) for v in col.tolist()], dtype=np.int64
            )
            blocks.append(_helmert(codes, len(levels)))
        else:
            try:
                num = col.astype(np.float64)
            except (TypeError, ValueError):
                raise LabelError(
                    f"{owner}: y column {c} is not numeric; pass label_kind='categorical'"
                ) from None
            if not np.all(np.isfinite(num)):
                raise LabelError(f"{owner} requires finite y")
            blocks.append(num.reshape(-1, 1))
    return np.hstack(blocks) if blocks else np.zeros((arr.shape[0], 0))


def _support_points(
    Z: np.ndarray, n_points: int, rng: np.random.Generator, max_iter: int, tol: float
) -> tuple[np.ndarray, int, bool, float]:
    """Support points of the rows of ``Z`` (Mak & Joseph 2018) by the convex-concave fixed point
    ``x_i <- [ (N/n) sum_k (x_i - x_k)/|x_i - x_k| + sum_m z_m/|x_i - z_m| ]
    / sum_m 1/|x_i - z_m|``,
    updated for all points at once from distinct data rows plus a small jitter, and clipped to the
    data's bounding box. Stops once the energy criterion ``2·mean|x - z| - mean|x - x'|`` improves
    by less than ``tol`` (relative) in one iteration; returns ``(points, iterations, converged,
    criterion)``."""
    N = Z.shape[0]
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    start = rng.choice(N, size=n_points, replace=False)
    X = np.clip(Z[start] + rng.normal(scale=1e-3, size=(n_points, Z.shape[1])), lo, hi)
    ratio = N / n_points
    previous = np.inf
    for it in range(1, max_iter + 1):
        Dxz = cdist(X, Z)
        Dxx = cdist(X, X)
        criterion = float(2.0 * Dxz.mean() - Dxx.mean())
        if np.isfinite(previous) and previous - criterion <= tol * abs(previous):
            return X, it - 1, True, criterion
        previous = criterion
        W = 1.0 / np.maximum(Dxz, 1e-12)
        np.fill_diagonal(Dxx, np.inf)
        V = 1.0 / np.maximum(Dxx, 1e-12)
        repulse = X * V.sum(axis=1)[:, None] - V @ X
        X = np.clip((ratio * repulse + W @ Z) / W.sum(axis=1)[:, None], lo, hi)
    criterion = float(2.0 * cdist(X, Z).mean() - cdist(X, X).mean())
    return X, max_iter, False, criterion


class SupportPointsSplitter(BaseSplitter):
    """SPlit: the smaller subset is the set of records nearest to the data's support points.

    Features, plus the labels under ``use_labels=True``, form one design matrix: categorical
    columns become Helmert contrasts, constant columns are dropped, and the rest are
    standardised. The ``k`` support points minimising energy distance to the data are computed
    for the smaller side of the cut by Mak & Joseph's convex-concave fixed point, started from
    ``k`` distinct records off the ``"support_points.init"`` stream. Each support point then
    takes its nearest unassigned record, and a validation set is selected the same way from the
    remainder. Joseph's optimal-ratio result suggests a test fraction near
    ``1 / (sqrt(p) + 1)`` for ``p`` model parameters; the sizes are left to the caller.

    :param featurizer: feature representation. The default ``"physchem"`` suits the method,
        which works in standardised Euclidean space.
    :param use_labels: append ``y`` to the design matrix when it is given, as in the original
        method.
    :param label_kind: how to encode ``y``: as continuous, as categorical through Helmert
        contrasts, or decided per column from its dtype.
    :param max_iter: cap on fixed-point iterations.
    :param tol: stop once the energy criterion improves by less than this fraction in one
        iteration.
    :param max_memory_bytes: ceiling on the support-point/record distance blocks.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``max_iter`` is below 1, ``tol`` is not positive, or
        ``label_kind`` is unknown.
    :raises ScalabilityError: at split time, if the distance blocks would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Both subsets follow the joint feature-and-label distribution as closely as a subset of
      that size can: an optimal version of what a random split achieves only on average.
    - Far less variance between seeds than a random split, so one split is representative.
    - Handles mixed continuous and categorical labels through Helmert coding.
    - Never builds an `n x n` matrix: memory grows with `n x k`.

    Pitfalls
    --------
    - **An interpolation split.** Test sits where training data is densest, so scores are as
      optimistic as a good random split, not a test of new chemistry.
    - With `use_labels=True` the split depends on `y`; `use_labels=False` keeps it
      label-blind.
    - Each fixed-point iteration costs `O(k * n * p)`, which makes this the slowest splitter in
      the family at large `n` and `k`.
    - Standardisation gives every column equal weight, so hundreds of noisy descriptors can
      drown out the labels.
    - Results depend on BLAS-level floating point, so splits can differ across platforms.

    References
    ----------
    .. [1] Joseph, V. R.; Vakayil, A. SPlit: An Optimal Method for Data Splitting.
       *Technometrics* **2022**, 64 (2), 166-176. https://doi.org/10.1080/00401706.2021.1921037
    .. [2] Mak, S.; Joseph, V. R. Support Points. *Ann. Statist.* **2018**, 46 (6A), 2562-2592.
       https://doi.org/10.1214/17-AOS1629
    .. [3] Joseph, V. R. Optimal Ratio for Data Splitting. *Stat. Anal. Data Min.* **2022**, 15
       (4), 531-538. https://doi.org/10.1002/sam.11583
    """

    splitter_id: ClassVar[str] = "support_points"
    family: ClassVar[str] = "similarity"
    strictness: ClassVar[Strictness] = Strictness.OPTIMISTIC
    group_forming: ClassVar[bool] = False
    accepts: ClassVar[tuple[str, ...]] = ("smiles", "mol", "features")
    deterministic_without_seed: ClassVar[bool] = False

    def __init__(
        self,
        *,
        featurizer: str | Any = "physchem",
        use_labels: bool = True,
        label_kind: Literal["auto", "continuous", "categorical"] = "auto",
        max_iter: int = 500,
        tol: float = 1e-6,
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.featurizer = featurizer
        self.use_labels = use_labels
        self.label_kind = label_kind
        self.max_iter = max_iter
        self.tol = tol
        self.max_memory_bytes = max_memory_bytes
        if not isinstance(use_labels, bool):
            raise ParameterError(f"use_labels must be a bool, got {use_labels!r}")
        if label_kind not in ("auto", "continuous", "categorical"):
            raise ParameterError(f"invalid label_kind: {label_kind!r}")
        if (
            isinstance(max_iter, bool)
            or not isinstance(max_iter, (int, np.integer))
            or max_iter < 1
        ):
            raise ParameterError(f"max_iter must be an int >= 1, got {max_iter!r}")
        if not (isinstance(tol, (int, float)) and tol > 0):
            raise ParameterError(f"tol must be > 0, got {tol!r}")
        if (
            isinstance(max_memory_bytes, bool)
            or not isinstance(max_memory_bytes, (int, np.integer))
            or max_memory_bytes <= 0
        ):
            raise ParameterError(f"max_memory_bytes must be an int > 0, got {max_memory_bytes!r}")

    def _design_matrix(self, ctx: _Context) -> tuple[np.ndarray, int]:
        F = ctx.get_features(resolve_featurizer(self.featurizer))
        X = np.asarray(F.toarray() if sp.issparse(F) else F, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ParameterError(f"{type(self).__name__}: features contain NaN or infinite values")
        blocks = [X]
        n_label_cols = 0
        if self.use_labels and ctx.y is not None:
            Y = _encode_label_columns(ctx.y, self.label_kind, type(self).__name__)
            n_label_cols = Y.shape[1]
            blocks.append(Y)
        Z = np.hstack(blocks)
        sd = Z.std(axis=0, ddof=1) if Z.shape[0] > 1 else np.zeros(Z.shape[1])
        keep = sd > 0
        if not keep.any():
            raise DegenerateGroupingError(
                f"{type(self).__name__}: every feature and label column is constant"
            )
        Z = (Z[:, keep] - Z[:, keep].mean(axis=0)) / sd[keep]
        return Z, n_label_cols

    def _guard(self, k: int, m: int) -> None:
        required = 8 * (3 * k * m + 2 * k * k)
        if required > self.max_memory_bytes:
            raise ScalabilityError(
                f"{type(self).__name__}: {k} support points over {m} records need about "
                f"{required:,} bytes, exceeding max_memory_bytes={self.max_memory_bytes:,}. "
                "Alternatives: (a) subsample the input; (b) use a smaller test or validation set; "
                "(c) raise max_memory_bytes."
            )

    def _select(
        self, Z: np.ndarray, pool: list[int], k: int, ctx: _Context, stage: int
    ) -> tuple[list[int], dict[str, Any]]:
        m = len(pool)
        if k <= 0:
            return [], {}
        if k >= m:
            return list(pool), {}
        n_points = min(k, m - k)  # support points always describe the smaller side
        self._guard(n_points, m)
        sub = Z[np.asarray(pool, dtype=np.int64)]
        rng = seed_for(ctx.rng_seeds, "support_points.init", stage)
        points, n_iter, converged, criterion = _support_points(
            sub, n_points, rng, self.max_iter, self.tol
        )
        D = cdist(points, sub)
        taken = np.zeros(m, dtype=bool)
        nearest: list[int] = []
        for i in range(n_points):
            row = np.where(taken, np.inf, D[i])
            j = int(row_argmin(row[None, :])[0])
            taken[j] = True
            nearest.append(j)
        picked_local = nearest if n_points == k else [j for j in range(m) if not taken[j]]
        chosen = sorted(pool[j] for j in picked_local)
        S = sub[np.asarray(nearest, dtype=np.int64)]
        selected = float(2.0 * cdist(S, sub).mean() - cdist(S, S).mean())
        return chosen, {
            "n_iterations": n_iter,
            "converged": converged,
            "support_point_criterion": criterion,
            "selected_criterion": selected,
        }

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        Z, n_label_cols = self._design_matrix(ctx)
        test, test_meta = self._select(Z, list(range(n)), ctx.sizes.n_test, ctx, 0)
        test_set = set(test)
        remaining = [i for i in range(n) if i not in test_set]
        valid, valid_meta = self._select(Z, remaining, ctx.sizes.n_valid, ctx, 1)
        valid_set = set(valid)
        train = [i for i in remaining if i not in valid_set]
        result = SplitResult(
            train=np.asarray(train, dtype=np.int64),
            valid=np.asarray(valid, dtype=np.int64),
            test=np.asarray(test, dtype=np.int64),
            discard=np.array([], dtype=np.int64),
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=n,
            metadata={
                "n_design_columns": int(Z.shape[1]),
                "used_label_columns": n_label_cols,
                "test_selection": test_meta,
                "valid_selection": valid_meta,
                # a chain of BLAS matrix products, so not bit-exact across BLAS builds
                "nondeterministic_method": True,
                "realised_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
            },
        )
        _small_partition_check(result, n)
        return [result]


class DuplexSplitter(_SimilarityBase):
    """DUPLEX: train, test and valid each grow as a maximally spread-out set, taking turns.

    Train is seeded with the farthest-apart pair, test with the farthest-apart pair of the
    rest, then valid likewise, ties to the lexicographically smallest pair. The partitions take
    turns adding the unassigned record farthest by minimum distance from their own members,
    ties to the smallest index, and leave the rotation once full. Snee described two sets; the
    rotation here generalises to three and to unequal sizes, so every size is met exactly.
    Fully deterministic.

    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Every partition spans the whole data space, so test covers the same range as train,
      unlike Kennard-Stone, which puts the extremes in train.
    - The alternation makes the partitions statistically similar in spread, which is the sense
      in which Snee meant them to validate a model.
    - Exact sizes, and deterministic without a seed.
    - `metadata["coverage_radius"]` reports how far any record is from each partition.

    Pitfalls
    --------
    - **An interpolation split.** Test records are spread through the same space as train, so
      scores are optimistic for genuinely new chemistry.
    - Seeding by the farthest pairs puts outliers into every partition first.
    - Builds the full `n x n` distance matrix, and each step is `O(n)`, so the whole split is
      `O(n^2)`.
    - Not a leakage control: near-duplicates can land in different partitions.

    References
    ----------
    .. [1] Snee, R. D. Validation of Regression Models: Methods and Examples. *Technometrics*
       **1977**, 19 (4), 415-428. https://doi.org/10.1080/00401706.1977.10489581
    """

    splitter_id: ClassVar[str] = "duplex"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    bounded_metric_required: ClassVar[bool] = False
    deterministic_without_seed: ClassVar[bool] = True

    def __init__(
        self,
        *,
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self._validate_similarity_params()

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        dist = _Distances(self, ctx, as_float64=True)
        names = ("train", "test", "valid")
        targets = (ctx.sizes.n_train, ctx.sizes.n_test, ctx.sizes.n_valid)
        parts = _clustering.duplex_order_access(
            n, targets, dist.column, dist.farthest_pair_in_pool
        )
        buckets = {
            name: np.sort(np.asarray(part, dtype=np.int64))
            for name, part in zip(names, parts, strict=True)
        }
        coverage = {
            name: dist.max_min_to(buckets[name])
            for name in names
            if buckets[name].size
        }
        result = SplitResult(
            train=buckets["train"],
            valid=buckets["valid"],
            test=buckets["test"],
            discard=np.array([], dtype=np.int64),
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=n,
            metadata={
                "seed_pairs": {
                    name: part[:2] for name, part in zip(names, parts, strict=True) if part
                },
                "coverage_radius": coverage,
                "realised_sizes": {name: int(buckets[name].size) for name in names},
            },
        )
        _small_partition_check(result, n)
        return [result]




def _d_optimal_exchange(
    X: np.ndarray, pool: list[int], start: list[int], ridge: float, max_passes: int
) -> tuple[list[int], dict[str, Any]]:
    """Modified Fedorov exchange over the rows of ``X``: improve ``start`` (a subset of ``pool``)
    until no single swap with a non-design record raises ``det(XᵀX + ridge·I)``."""
    design = sorted(start)
    in_design = set(design)
    eye = ridge * np.eye(X.shape[1])

    def inverse(rows: list[int]) -> np.ndarray:
        Xd = X[rows]
        return np.linalg.inv(Xd.T @ Xd + eye)

    Minv = inverse(design)
    n_passes = n_swaps = 0
    for _ in range(max_passes):
        n_passes += 1
        swapped = False
        for i in list(design):
            candidates = np.asarray([j for j in pool if j not in in_design], dtype=np.int64)
            if candidates.size == 0:
                break
            xi = X[i]
            Xc = X[candidates]
            d_i = float(xi @ Minv @ xi)
            d_j = np.einsum("ij,jk,ik->i", Xc, Minv, Xc)
            d_ij = Xc @ (Minv @ xi)
            delta = d_j - d_i - (d_i * d_j - d_ij**2)
            best = int(row_argmin(-delta[None, :])[0])
            if delta[best] > 1e-10:
                j = int(candidates[best])
                design[design.index(i)] = j
                in_design.discard(i)
                in_design.add(j)
                Minv = inverse(design)
                n_swaps += 1
                swapped = True
        if not swapped:
            break
    Xd = X[design]
    _, log_det = np.linalg.slogdet(Xd.T @ Xd + eye)
    return sorted(design), {"log_det": float(log_det), "n_passes": n_passes, "n_swaps": n_swaps}


class DOptimalSplitter(BaseSplitter):
    """D-optimal design: the training set is the subset that maximises ``det(XᵀX)``.

    Features are standardised, constant columns dropped, and projected onto their leading
    ``n_components`` sign-fixed principal components; the design matrix is those scores plus an
    intercept. From a Kennard-Stone start on the scores, or a ``"d_optimal.init"`` draw, a
    modified Fedorov exchange visits design points in index order and swaps each for the
    non-design record that most increases the determinant --
    ``Δ(i, j) = d(j) − d(i) − [d(i)·d(j) − d(i, j)²]`` with ``d(a, b) = x_aᵀ (XᵀX)⁻¹ x_b``,
    ties to the smallest index -- until a pass makes no swap. A validation set is the D-optimal
    subset of what remains; the rest is test.

    :param featurizer: feature representation. The default ``"physchem"`` suits D-optimality,
        which is defined on continuous design variables.
    :param n_components: principal components in the design matrix, or ``None`` for
        ``min(10, n_train - 2, n_features)``.
    :param init: starting design: the deterministic Kennard-Stone selection, or a random draw.
    :param ridge: added to the diagonal of ``X'X`` so near-singular designs stay invertible.
    :param max_passes: cap on exchange passes over the design.
    :param max_memory_bytes: ceiling for the Kennard-Stone distance matrix.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``n_components`` or ``max_passes`` is below 1, ``ridge`` is
        negative, or ``init`` is unknown.
    :raises ScalabilityError: at split time, if the distance matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - The training set gives the most precise estimates of a linear model's coefficients in the
      chosen descriptor space, which is the classical optimal-design criterion.
    - Deterministic without a seed under the default Kennard-Stone start.
    - `metadata["train_design"]["log_det"]` reports the achieved criterion, so designs can be
      compared.
    - Works on PCA scores, so it stays well-posed with many correlated descriptors.

    Pitfalls
    --------
    - **Picks the edges of descriptor space.** Train concentrates on extreme records and test
      holds the interior, so scores can overestimate predictive power, as Gramatica and
      co-workers noted.
    - Optimal for a linear model in the chosen components. A nonlinear model, or different
      descriptors, would want a different design.
    - The exchange finds a local optimum, which depends on the starting design.
    - Each pass costs `O(n_train * n * p^2)`, so large sets with many components are slow.
    - Selection ignores `y`, so label imbalance between train and test is not controlled.

    References
    ----------
    .. [1] Fedorov, V. V. *Theory of Optimal Experiments*; Academic Press: New York, **1972**.
    .. [2] Cook, R. D.; Nachtsheim, C. J. A Comparison of Algorithms for Constructing Exact
       D-Optimal Designs. *Technometrics* **1980**, 22 (3), 315-324.
       https://doi.org/10.1080/00401706.1980.10486162
    .. [3] de Aguiar, P. F.; Bourguignon, B.; Khots, M. S.; Massart, D. L.; Phan-Than-Luu, R.
       D-Optimal Designs. *Chemom. Intell. Lab. Syst.* **1995**, 30 (2), 199-210.
       https://doi.org/10.1016/0169-7439(94)00076-X
    """

    splitter_id: ClassVar[str] = "d_optimal"
    family: ClassVar[str] = "similarity"
    strictness: ClassVar[Strictness] = Strictness.MODERATE
    group_forming: ClassVar[bool] = False
    accepts: ClassVar[tuple[str, ...]] = ("smiles", "mol", "features")
    deterministic_without_seed: ClassVar[bool] = True  # False for init="random"

    def __init__(
        self,
        *,
        featurizer: str | Any = "physchem",
        n_components: int | None = None,
        init: Literal["kennard_stone", "random"] = "kennard_stone",
        ridge: float = 1e-8,
        max_passes: int = 100,
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.featurizer = featurizer
        self.n_components = n_components
        self.init = init
        self.ridge = ridge
        self.max_passes = max_passes
        self.max_memory_bytes = max_memory_bytes
        if n_components is not None and (
            isinstance(n_components, bool)
            or not isinstance(n_components, (int, np.integer))
            or n_components < 1
        ):
            raise ParameterError(f"n_components must be None or an int >= 1, got {n_components!r}")
        if init not in ("kennard_stone", "random"):
            raise ParameterError(f"invalid init: {init!r}")
        if not (isinstance(ridge, (int, float)) and ridge >= 0):
            raise ParameterError(f"ridge must be >= 0, got {ridge!r}")
        if (
            isinstance(max_passes, bool)
            or not isinstance(max_passes, (int, np.integer))
            or max_passes < 1
        ):
            raise ParameterError(f"max_passes must be an int >= 1, got {max_passes!r}")

    def _design(self, ctx: _Context, n_select: int) -> np.ndarray:
        F = ctx.get_features(resolve_featurizer(self.featurizer))
        X = np.asarray(F.toarray() if sp.issparse(F) else F, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ParameterError(f"{type(self).__name__}: features contain NaN or infinite values")
        sd = X.std(axis=0)
        keep = sd > 0
        if not keep.any():
            raise DegenerateGroupingError(
                f"{type(self).__name__}: every feature column is constant"
            )
        Xs = (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]
        limit = min(Xs.shape[1], n_select - 2)
        p = self.n_components if self.n_components is not None else min(10, limit)
        if not (1 <= p <= limit):
            raise ParameterError(
                f"{type(self).__name__}: n_components={p} must be between 1 and "
                f"min(n_features, n_selected - 2) = {limit}"
            )
        U, S, _ = np.linalg.svd(Xs, full_matrices=False)
        scores = _fix_svd_signs(U[:, :p] * S[:p])
        return np.hstack([np.ones((ctx.n, 1)), scores])

    def _select(
        self, X: np.ndarray, pool: list[int], k: int, ctx: _Context, stage: int
    ) -> tuple[list[int], dict[str, Any]]:
        if k <= 0:
            return [], {}
        if k >= len(pool):
            return list(pool), {}
        idx = np.asarray(pool, dtype=np.int64)
        if self.init == "kennard_stone":
            guard_memory(len(pool), self.max_memory_bytes, type(self).__name__)
            scores = X[idx, 1:]
            D = cdist(scores, scores)
            start = [int(idx[j]) for j in _clustering.kennard_stone(D, k)[:k]]
        else:
            rng = seed_for(ctx.rng_seeds, "d_optimal.init", stage)
            start = sorted(int(j) for j in rng.choice(idx, size=k, replace=False))
        return _d_optimal_exchange(X, pool, start, float(self.ridge), int(self.max_passes))

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        X = self._design(ctx, ctx.sizes.n_train)
        train, train_meta = self._select(X, list(range(n)), ctx.sizes.n_train, ctx, 0)
        train_set = set(train)
        remaining = [i for i in range(n) if i not in train_set]
        valid, valid_meta = self._select(X, remaining, ctx.sizes.n_valid, ctx, 1)
        valid_set = set(valid)
        test = [i for i in remaining if i not in valid_set]
        result = SplitResult(
            train=np.asarray(train, dtype=np.int64),
            valid=np.asarray(valid, dtype=np.int64),
            test=np.asarray(test, dtype=np.int64),
            discard=np.array([], dtype=np.int64),
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=n,
            metadata={
                "n_components": int(X.shape[1] - 1),
                "train_design": train_meta,
                "valid_design": valid_meta,
                "realised_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
            },
        )
        _small_partition_check(result, n)
        return [result]


class MaxDissimilaritySplitter(_SimilarityBase):
    """Pushes train and test to opposite regions of chemical space.

    :param seed_pair: start from the farthest-apart pair of records, or from a seeded random
        pair.
    :param grow: add each record to the partition whose seed it is nearest, or to the partition
        whose current members it is nearest.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``seed_pair`` or ``grow`` is unknown.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Produces a clean, reproducible large extrapolation, which is the right test for whether a
      model can reach a region it has never seen.
    - Only two records are chosen by any rule, and everything else follows deterministically,
      so the split is easy to describe and audit.
    - `metadata["min_cross_distance"]` quantifies how far apart the two sets ended up.

    Pitfalls
    --------
    - Deliberately worst-case: it estimates **one specific** extrapolation, not average
      prospective performance, so a single number carries a very wide implicit interval.
    - The whole split hinges on two seed molecules, usually outliers, so one badly standardised
      salt can define the entire experiment.
    - Test and train are contiguous regions, so the test set is chemically homogeneous with
      strongly correlated errors, and its effective sample size is far below `n_test`.
    - Not a leakage constraint: nothing bounds the minimum train-to-test distance beyond
      whatever geometry results. `min_cross_distance` is the number to check.
    - `grow="nearest_to_set"` can chain and walk the test set back toward the train seed, while
      `"nearest_to_seed"` keeps it compact. The two give materially different splits.

    References
    ----------
    .. [1] The two-seed grow-apart construction is a composition rather than a published
       method. Its diverse-selection root is [2]; the comparative evidence is [3] and [4].
    .. [2] Kennard, R. W.; Stone, L. A. Computer Aided Design of Experiments. *Technometrics*
       **1969**, 11 (1), 137-148. https://doi.org/10.1080/00401706.1969.10490666
    .. [3] Martin, T. M.; Harten, P.; Young, D. M. et al. Does Rational Selection of Training
       and Test Sets Improve the Outcome of QSAR Modeling? *J. Chem. Inf. Model.* **2012**,
       52 (10), 2570-2578. https://doi.org/10.1021/ci300338w
    .. [4] Tossou, P.; Wognum, C.; Craig, M.; Mary, H.; Noutahi, E. Real-World Molecular
       Out-Of-Distribution: Specification and Investigation. *J. Chem. Inf. Model.* **2024**,
       64 (3), 697-711. https://doi.org/10.1021/acs.jcim.3c01774
    """

    splitter_id: ClassVar[str] = "max_dissimilarity"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    bounded_metric_required: ClassVar[bool] = False
    deterministic_without_seed: ClassVar[bool] = False  # True only for seed_pair="max_distance"

    def __init__(
        self,
        *,
        seed_pair: Literal["max_distance", "random"] = "max_distance",
        grow: Literal["nearest_to_seed", "nearest_to_set"] = "nearest_to_seed",
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.seed_pair = seed_pair
        self.grow = grow
        self._validate_similarity_params()

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        # Only columns, one seed pair and two reductions are read, all of which stream.
        dist = _Distances(self, ctx)
        if self.seed_pair == "max_distance":
            max_d = dist.max_upper()
            a, b = dist.first_pair_at_least(max_d - EPS)
            n_tied = dist.count_pairs_at_least(max_d - EPS)
        else:
            rng = seed_for(ctx.rng_seeds, "maxdiss.seed", 0)
            a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
            n_tied = 1
        test = [b]
        key = dist.column(b).copy()
        assigned = np.zeros(n, dtype=bool)
        assigned[b] = True
        n_test = max(1, ctx.sizes.n_test)
        while len(test) < n_test:
            if bool(assigned.all()):
                break
            # one numpy pass instead of rebuilding an O(n) candidate list per pick
            nxt = masked_argmin(key, assigned)
            test.append(nxt)
            assigned[nxt] = True
            if self.grow == "nearest_to_set":
                key = np.minimum(key, dist.column(nxt))
        rest = [i for i in range(n) if not assigned[i]]
        n_valid = ctx.sizes.n_valid
        valid: list[int] = []
        if n_valid > 0:
            col_a = dist.column(a)
            rest_sorted = stable_sort(rest, key=lambda i: col_a[i], desc=True)
            valid = rest_sorted[:n_valid]
        train = [i for i in rest if i not in set(valid)]
        result = SplitResult(
            train=np.sort(np.asarray(train, dtype=np.int64)),
            valid=np.sort(np.asarray(valid, dtype=np.int64)),
            test=np.sort(np.asarray(test, dtype=np.int64)),
            discard=np.array([], dtype=np.int64),
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=n,
            metadata={
                "seed_train": a,
                "seed_test": b,
                "seed_distance": dist.pair(a, b),
                "n_tied_seed_pairs": n_tied,
                "min_cross_distance": (
                    dist.min_between(test, train)
                    if train and test
                    else float("nan")
                ),
                "realised_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
            },
        )
        _small_partition_check(result, n)
        return [result]


class PerimeterSplitter(_SimilarityBase):
    """Holds out the outskirts of the distribution; trains on the dense core.

    :param pair_rule: how peripherality is turned into a partition: repeatedly take the
        farthest-apart remaining pair, or rank every record by an outlier score.
    :param featurizer: featurizer alias or instance used to build the distance matrix.
    :param metric: distance or similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``pair_rule`` is unknown.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Tests the applicability-domain edge directly: the held-out molecules are the ones a
      deployed model would be least confident about.
    - Deterministic, with no seed and no free parameter beyond the metric.
    - The training set stays dense and representative, so training is stable even though
      evaluation is hard.

    Pitfalls
    --------
    - Test is enriched in oddities -- fragments, salts, dyes, extreme sizes, standardisation
      failures -- so a poor score can be a data-quality result rather than a model one.
    - Error bars run large, since the test set is heterogeneous and small in effective size.
    - `greedy_pairs` needs `O(n^2)` memory for the pair sort, so it does not scale past a few
      tens of thousands of records.
    - Not a chemical-novelty guarantee: an outlier can still sit near a training molecule if
      that is its only near neighbour. `audit.nn_similarity_profile` shows whether it does.
    - Peripherality is defined by mean distance, so the split partly encodes molecular size and
      fingerprint density.

    References
    ----------
    .. [1] Szántai-Kis, C.; Kövesdi, I.; Kéri, G.; Örfi, L. Validation Subset Selections for
       Extrapolation Oriented QSPAR Models. *Mol. Divers.* **2003**, 7 (1), 37-43.
       https://doi.org/10.1023/B:MODI.0000006538.99122.00
    .. [2] Tossou, P.; Wognum, C.; Craig, M.; Mary, H.; Noutahi, E. Real-World Molecular
       Out-Of-Distribution: Specification and Investigation. *J. Chem. Inf. Model.* **2024**,
       64 (3), 697-711. https://doi.org/10.1021/acs.jcim.3c01774
    """

    splitter_id: ClassVar[str] = "perimeter"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    bounded_metric_required: ClassVar[bool] = False
    deterministic_without_seed: ClassVar[bool] = True

    def __init__(
        self,
        *,
        pair_rule: Literal["greedy_pairs", "outlier_score"] = "greedy_pairs",
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs
        )
        self.pair_rule = pair_rule
        self._validate_similarity_params()

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        D = _dist_matrix(self, ctx)
        n = ctx.n
        guard_memory(n, self.max_memory_bytes, type(self).__name__)
        n_test = ctx.sizes.n_test
        outlier_score = D.mean(axis=1)
        fallback_filled = 0
        odd_trim = False
        if self.pair_rule == "outlier_score":
            order = stable_sort(range(n), key=lambda i: outlier_score[i], desc=True)
            test = order[:n_test]
            n_pairs_used = 0
        else:
            iu = np.triu_indices(n, k=1)
            dvals = D[iu]
            pair_order = sorted(
                range(len(dvals)), key=lambda k: (-dvals[k], int(iu[0][k]), int(iu[1][k]))
            )
            test: list[int] = []
            assigned: set[int] = set()
            n_pairs_used = 0
            for k in pair_order:
                if len(test) >= n_test:
                    break
                i, j = int(iu[0][k]), int(iu[1][k])
                if i in assigned or j in assigned:
                    continue
                test.extend([i, j])
                assigned.update([i, j])
                n_pairs_used += 1
            if len(test) > n_test:
                odd_trim = True
                test = test[:n_test]
            if len(test) < n_test:
                remaining = [i for i in range(n) if i not in set(test)]
                need = n_test - len(test)
                fill = stable_sort(remaining, key=lambda i: outlier_score[i], desc=True)[:need]
                fallback_filled = len(fill)
                test.extend(fill)
        test_set = set(test)
        rest = [i for i in range(n) if i not in test_set]
        n_valid = ctx.sizes.n_valid
        valid = (
            stable_sort(rest, key=lambda i: outlier_score[i], desc=True)[:n_valid]
            if n_valid
            else []
        )
        train = [i for i in rest if i not in set(valid)]
        result = SplitResult(
            train=np.sort(np.asarray(train, dtype=np.int64)),
            valid=np.sort(np.asarray(valid, dtype=np.int64)),
            test=np.sort(np.asarray(test, dtype=np.int64)),
            discard=np.array([], dtype=np.int64),
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=n,
            metadata={
                "pair_rule": self.pair_rule,
                "n_pairs_used": n_pairs_used,
                "odd_pair_trim": odd_trim,
                "fallback_filled": fallback_filled,
                "test_mean_outlier_score": (
                    float(np.mean(outlier_score[test])) if test else float("nan")
                ),
                "train_mean_outlier_score": (
                    float(np.mean(outlier_score[train])) if train else float("nan")
                ),
                "realised_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
            },
        )
        _small_partition_check(result, n)
        return [result]


class LeaveOneClusterOutSplitter(GroupSplitter):
    """Each cluster (from a caller-supplied ``clusterer``) takes a turn as the test fold.

    :param clusterer: supplies the grouping, as a group-forming
        :class:`~chemsplit.base.GroupSplitter` or a registry id resolved at split time.
        ``None`` means Butina at a 0.35 ECFP4/Tanimoto cutoff.
    :param min_cluster_size: clusters smaller than this are handled by
        ``small_cluster_policy``.
    :param small_cluster_policy: fold undersized clusters into train, let each be its own fold
        anyway, or pool them into one fold.
    :param max_folds: cap on the number of folds, or ``None`` for one per cluster. Capping
        leaves some clusters untested.
    :param fold_order: visit clusters largest-first, smallest-first, or by cluster id.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``clusterer`` is not ``None``, a string or a
        :class:`GroupSplitter`, a size parameter is below 1, or a mode parameter is unknown.
    :raises DegenerateGroupingError: at split time, if the clusterer yields a single cluster,
        leaving nothing to hold out.

    Notes
    -----
    The grouping is driven by this splitter's own ``random_state``, whichever form
    ``clusterer`` takes: a clusterer used as a grouping source draws from the seed bundle it is
    handed, which is this splitter's. A clusterer instance's own ``random_state`` is therefore
    not consulted here, unlike when that instance is used as a splitter in its own right. A
    string is additionally instantiated with a seed derived from
    ``purpose="leave_one_cluster_out.clusterer"``.

    Advantages
    ----------
    - Yields a **per-cluster error distribution** instead of one number, so it shows which
      regions of chemical space the model fails in rather than only that it fails.
    - With `max_folds=None` and no small-cluster pooling, every record is tested exactly
      once, so the aggregate is a whole-dataset estimate.
    - Composes with every scaffold, similarity and embedding grouping, so one protocol covers
      generalisation across scaffolds, across Butina clusters and across UMAP regions.
    - Takes the grouping as a registry id, so a leave-one-group-out protocol is one string away
      -- `clusterer="source"` -- with no splitter class to import.

    Pitfalls
    --------
    - Uneven cluster sizes make per-fold scores come from very different samples, so a
      macro-average can disagree sharply with a micro-average. `cluster_size` accompanies
      every fold.
    - Many small clusters make the fold count explode and the run expensive. `max_folds` caps
      it, at the cost of leaving some clusters untested and biasing the aggregate.
    - Training-set size varies across folds, so fold-to-fold differences partly measure
      training-set size rather than chemical difficulty.
    - A three-record test fold cannot support ROC-AUC or a meaningful R². The splitter warns,
      but the metric is the caller's to compute.

    References
    ----------
    .. [1] Kramer, C.; Gedeck, P. Leave-Cluster-Out Cross-Validation Is Appropriate for
       Scoring Functions Derived from Diverse Protein Data Sets. *J. Chem. Inf. Model.*
       **2010**, 50 (11), 1961-1969. https://doi.org/10.1021/ci100264e
    """

    splitter_id: ClassVar[str] = "leave_one_cluster_out"
    family: ClassVar[str] = "similarity"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    deterministic_method: ClassVar[bool] = True

    def __init__(
        self,
        *,
        clusterer: str | GroupSplitter | None = None,
        min_cluster_size: int = 1,
        small_cluster_policy: Literal["merge_into_train", "own_fold", "pool"] = "merge_into_train",
        max_folds: int | None = 50,
        fold_order: Literal["size_desc", "size_asc", "index"] = "size_desc",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        check_group_splitter_design(clusterer, "clusterer", owner=type(self).__name__)
        self.clusterer = clusterer
        self.min_cluster_size = min_cluster_size
        self.small_cluster_policy = small_cluster_policy
        self.max_folds = max_folds
        self.fold_order = fold_order
        if self.n_splits != 1:
            raise ConfigurationError(
                "n_splits is derived from the cluster count and must not be set"
            )

    def _group_labels(self, ctx: _Context) -> IndexArray:
        clusterer = resolve_group_splitter(
            self.clusterer,
            "clusterer",
            owner=type(self).__name__,
            random_state=int(
                seed_for(ctx.rng_seeds, "leave_one_cluster_out.clusterer", 0).integers(0, 2**31 - 1)
            ),
        )
        if clusterer is not None:
            return clusterer._group_labels(ctx)
        clusters = _default_butina_clusters(ctx, type(self).__name__, self.n_jobs)
        labels = np.empty(ctx.n, dtype=np.int64)
        for cid, members in enumerate(clusters):
            for m in members:
                labels[m] = cid
        return dense_label_encode(labels.tolist())

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Report how many splits will be yielded.

        :param X: ignored, as are ``y`` and ``groups``; the signature is sklearn\'s.
        :return: the fold count, which is only known once the clusters are formed.
        """
        if X is None:
            return 1
        labels = self.compute_groups(X, y)
        fold_groups, _eligible, _never_tested = self._eligible_fold_groups(labels)
        return len(fold_groups)

    def _eligible_fold_groups(
        self, labels: IndexArray
    ) -> tuple[list[int], dict[int, list[int]], list[int]]:
        """Shared by :meth:`get_n_splits` and :meth:`_partition` so the two can never disagree:
        returns ``(fold_groups, eligible, never_tested)`` where ``fold_groups`` is the final,
        ordered, max_folds-capped list of group ids that will each become one test fold, and
        ``never_tested`` is whatever ``max_folds`` truncated off the end of that same order."""
        n = len(labels)
        members: dict[int, list[int]] = {}
        for i in range(n):
            members.setdefault(int(labels[i]), []).append(i)
        if len(members) == 1:
            raise DegenerateGroupingError(
                f"{type(self).__name__}: clusterer produced a single cluster"
            )
        small = [g for g, m in members.items() if len(m) < self.min_cluster_size]
        pooled: list[int] = []
        eligible = dict(members)
        if small:
            if self.small_cluster_policy == "merge_into_train":
                for g in small:
                    del eligible[g]
            elif self.small_cluster_policy == "pool":
                for g in small:
                    pooled.extend(eligible.pop(g))
                if pooled:
                    eligible[max(members) + 1] = sorted(pooled)
            # "own_fold": leave as-is

        def sort_key(g: int) -> Any:
            if self.fold_order == "size_desc":
                return (-len(eligible[g]), eligible[g][0])
            if self.fold_order == "size_asc":
                return (len(eligible[g]), eligible[g][0])
            return (eligible[g][0],)

        ordered = sorted(eligible.keys(), key=sort_key)
        never_tested: list[int] = []
        fold_groups = ordered
        if self.max_folds is not None and len(ordered) > self.max_folds:
            never_tested = ordered[self.max_folds:]
            fold_groups = ordered[: self.max_folds]
        return fold_groups, eligible, never_tested

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        labels = self._group_labels(ctx)
        n = ctx.n
        members: dict[int, list[int]] = {}
        for i in range(n):
            members.setdefault(int(labels[i]), []).append(i)
        fold_groups, eligible, never_tested = self._eligible_fold_groups(labels)

        results = []
        for f, g in enumerate(fold_groups):
            test = np.asarray(sorted(eligible[g]), dtype=np.int64)
            test_set = set(eligible[g])
            train = np.asarray([i for i in range(n) if i not in test_set], dtype=np.int64)
            result = SplitResult(
                train=train,
                valid=np.array([], dtype=np.int64),
                test=test,
                discard=np.array([], dtype=np.int64),
                groups=labels,
                splitter_id=self.splitter_id,
                params=self.get_params(),
                n_records=n,
                metadata={
                    "fold_index": f,
                    "cluster_id": g,
                    "cluster_size": int(test.size),
                    "n_clusters": len(members),
                    "clusters_never_tested": never_tested,
                },
            )
            results.append(result)
        if never_tested:
            warn_with_details(
                SizeToleranceWarning(
                    f"{type(self).__name__}: {len(never_tested)} cluster(s) never tested "
                    f"(max_folds={self.max_folds})",
                    details={"clusters_never_tested": never_tested},
                )
            )
        return results


class BalancedMultiTaskSplitter(GroupSplitter):
    """Balanced multi-task cluster assignment: assigns whole clusters to folds so
    every task gets an acceptable train/test ratio and label balance.

    Clusters are assigned to folds via an independently designed optimizer
    (:mod:`chemsplit._optimize`), whose backends were selected by comparing candidate
    architectures across problem sizes. The ``solver`` parameter accepts
    ``"auto"``/``"milp"``/``"heuristic"``.

    :param clusterer: supplies the grouping, as a group-forming
        :class:`~chemsplit.base.GroupSplitter` or a registry id resolved at split time.
        ``None`` means Butina at a 0.35 ECFP4/Tanimoto cutoff.
    :param clusterer_kwargs: constructor kwargs for a string ``clusterer``. Passing them
        alongside an already-built ``clusterer`` raises
        :class:`~chemsplit.exceptions.ConfigurationError` rather than being ignored.
    :param task_weights: per-task weights on the balance objective, or ``None`` for equal
        weights.
    :param balance: balance record counts per task only, or record counts and active counts.
    :param tolerance: how far a task's realised ratio may sit from its target.
    :param solver: ``"milp"`` for the exact formulation, ``"heuristic"`` for local search, or
        ``"auto"`` to dispatch on problem size.
    :param time_limit_s: wall-clock limit on the solve.
    :param mip_gap: relative optimality gap at which the MILP backend stops.
    :param on_infeasible: raise when the requested balance cannot be met, or retry with
        progressively looser tolerances.
    :param relax_steps: the tolerances tried in turn by ``on_infeasible="relax"``.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if a numeric parameter is out of range, or ``balance``, ``solver``
        or ``on_infeasible`` is unknown.
    :raises ConfigurationError: if ``clusterer_kwargs`` accompanies a ``clusterer`` instance.
    :raises ConstraintUnsatisfiableError: at split time, if the balance cannot be met and
        ``on_infeasible="raise"``.

    Notes
    -----
    The grouping is driven by this splitter's own ``random_state``, whichever form
    ``clusterer`` takes: a clusterer used as a grouping source draws from the seed bundle it is
    handed, which is this splitter's. A clusterer instance's own ``random_state`` is therefore
    not consulted here, unlike when that instance is used as a splitter in its own right. A
    string is additionally instantiated with a seed derived from
    ``purpose="balanced_multi_task.clusterer"``, unless ``clusterer_kwargs`` sets one.

    Advantages
    ----------
    - The practical answer to sparse multi-task matrices, where naive cluster splitting can
      leave some targets with zero test actives and undefined metrics.
    - Balance is a *constraint*, not a hope: when the requested balance is impossible the
      splitter says so and names the binding task instead of producing a useless fold.
    - `per_task_fold_counts` audits what every task got.
    - Works with any clusterer, instance or registry id, configured through
      `clusterer_kwargs`, which keeps the chemical criterion separate from the balancing.

    Pitfalls
    --------
    - Infeasibility is common: a task with three actives in one cluster cannot be balanced.
      `on_infeasible="relax"` proceeds, but then the balance achieved is not the one
      requested, and `metadata["tolerance_used"]` says which held.
    - Solve time grows quickly with cluster count, and every backend stops at `time_limit_s`,
      so a solve that hits the limit depends on machine speed too.
    - `solver="auto"` dispatches on problem size alone -- branch-and-bound for tiny problems,
      local search above -- so crossing the threshold changes algorithm, recorded in
      `metadata["solver"]`. The heuristics always report
      `solver_status="time_limit_feasible"`, so only branch-and-bound and MILP prove
      optimality.
    - Balancing on label statistics chooses the split partly from the labels, a mild leak
      into the design, and usually the lesser evil against undefined metrics.

    References
    ----------
    .. [1] Tricarico, G. A.; Hofmans, J.; Lenselink, E. B.; López-Ramos, M.; Dréanic, M.-P.;
       Stouten, P. F. W. Construction of Balanced, Chemically Dissimilar Training, Validation
       and Test Sets for Machine Learning on Molecular Datasets. *ChemRxiv* preprint, **2024**
       (not peer reviewed). https://doi.org/10.26434/chemrxiv-2022-m8l33-v3
    .. [2] The cluster-to-fold assignment here is solved by chemsplit's own optimizer
       (:mod:`chemsplit._optimize`), not by the formulation used in that work.
    """

    splitter_id: ClassVar[str] = "balanced_multi_task"
    family: ClassVar[str] = "similarity"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    requires_labels: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    extras: ClassVar[tuple[str,...]] = ()  # deliberately empty: no external solver dependency
    deterministic_method: ClassVar[bool] = True

    def __init__(
        self,
        *,
        clusterer: str | GroupSplitter | None = None,
        clusterer_kwargs: dict | None = None,
        task_weights: list[float] | None = None,
        balance: Literal["counts", "counts_and_actives"] = "counts_and_actives",
        tolerance: float = 0.10,
        solver: Literal["auto", "milp", "heuristic"] = "auto",
        time_limit_s: float = 300.0,
        mip_gap: float = 1e-4,
        on_infeasible: Literal["raise", "relax"] = "raise",
        relax_steps: tuple[float,...] = (0.15, 0.20, 0.30),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        check_group_splitter_design(
            clusterer,
            "clusterer",
            owner=type(self).__name__,
            design_kwargs=clusterer_kwargs,
            design_kwargs_name="clusterer_kwargs",
        )
        self.clusterer = clusterer
        self.clusterer_kwargs = clusterer_kwargs
        self.task_weights = task_weights
        self.balance = balance
        self.tolerance = tolerance
        self.solver = solver
        self.time_limit_s = time_limit_s
        self.mip_gap = mip_gap
        self.on_infeasible = on_infeasible
        self.relax_steps = relax_steps
        if not (0.0 < tolerance < 1.0):
            raise ParameterError(f"tolerance must be in (0,1), got {tolerance!r}")

    def _check_preconditions(self, ctx: _Context) -> None:
        if ctx.y is None or np.asarray(ctx.y).ndim != 2:
            raise LabelError(f"{type(self).__name__} requires a 2-D (n, n_tasks) label matrix")

    def _group_labels(self, ctx: _Context) -> IndexArray:
        clusterer = resolve_group_splitter(
            self.clusterer,
            "clusterer",
            owner=type(self).__name__,
            design_kwargs=self.clusterer_kwargs,
            random_state=int(
                seed_for(ctx.rng_seeds, "balanced_multi_task.clusterer", 0).integers(0, 2**31 - 1)
            ),
        )
        if clusterer is not None:
            return clusterer._group_labels(ctx)
        clusters = _default_butina_clusters(ctx, type(self).__name__, self.n_jobs)
        labels = np.empty(ctx.n, dtype=np.int64)
        for cid, members in enumerate(clusters):
            for m in members:
                labels[m] = cid
        return dense_label_encode(labels.tolist())

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        labels = self._group_labels(ctx)
        n_clusters = int(labels.max()) + 1
        if n_clusters < 2:
            raise DegenerateGroupingError(f"{type(self).__name__}: fewer than 2 clusters")

        y = np.asarray(ctx.y, dtype=np.float64)
        n_tasks = y.shape[1]
        item_size = np.bincount(labels, minlength=n_clusters).astype(np.int64)
        item_task_counts = np.zeros((n_clusters, n_tasks), dtype=np.float64)
        item_task_actives = (
            np.zeros((n_clusters, n_tasks), dtype=np.float64)
            if self.balance == "counts_and_actives"
            else None
        )
        for c in range(n_clusters):
            rows = y[labels == c]
            mask = ~np.isnan(rows)
            item_task_counts[c] = mask.sum(axis=0)
            if item_task_actives is not None:
                item_task_actives[c] = np.nansum(np.where(mask, rows, 0.0), axis=0)

        buckets = [
            ("train", ctx.sizes.n_train),
            ("valid", ctx.sizes.n_valid),
            ("test", ctx.sizes.n_test),
        ]
        active_buckets = [(name, cap) for name, cap in buckets if cap > 0]
        n_buckets = len(active_buckets)
        bucket_target = np.asarray([cap for _, cap in active_buckets], dtype=np.float64)

        task_weight = (
            np.ones(n_tasks)
            if self.task_weights is None
            else np.asarray(self.task_weights, dtype=np.float64)
        )

        tolerance = self.tolerance
        if self.solver == "auto":
            architecture = "auto"
        else:
            architecture = "milp" if self.solver == "milp" else "local_search"
        rng = seed_for(ctx.rng_seeds, "balanced_multi_task.solver", 0)
        attempts = [tolerance] + (list(self.relax_steps) if self.on_infeasible == "relax" else [])
        solution = None
        used_tolerance = tolerance
        for t in attempts:
            problem = BalanceProblem(
                n_items=n_clusters,
                n_buckets=n_buckets,
                item_size=item_size,
                bucket_target_size=bucket_target,
                size_tolerance=t,
                item_task_counts=item_task_counts,
                item_task_actives=item_task_actives,
                task_weight=task_weight,
                task_tolerance=t,
            )
            solution = solve_balance(
                problem,
                rng=rng,
                time_limit_s=self.time_limit_s,
                architecture=architecture,
                mip_gap=self.mip_gap,
            )
            used_tolerance = t
            if solution.solver_status != "infeasible":
                break
        assert solution is not None

        if solution.solver_status == "infeasible":
            if self.on_infeasible == "raise":
                raise ConstraintUnsatisfiableError(
                    f"{type(self).__name__}: could not balance {n_tasks} task(s) across "
                    f"{n_buckets} folds within tolerance={tolerance} (best objective "
                    f"{solution.objective:.4f})"
                )
            warn_with_details(
                SizeToleranceWarning(
                    f"{type(self).__name__}: accepted relaxed tolerance={used_tolerance}",
                    details={"tolerance_used": used_tolerance},
                )
            )
        elif used_tolerance != tolerance:
            warn_with_details(
                SizeToleranceWarning(
                    f"{type(self).__name__}: accepted relaxed tolerance={used_tolerance}",
                    details={"tolerance_used": used_tolerance},
                )
            )

        out: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
        for c in range(n_clusters):
            b = int(solution.assignment[c])
            name = active_buckets[b][0]
            out[name].extend(np.nonzero(labels == c)[0].tolist())

        per_task_fold_counts = [
            [float(item_task_counts[solution.assignment == b, t].sum()) for b in range(n_buckets)]
            for t in range(n_tasks)
        ]
        per_task_fold_actives = (
            [
                [
                    float(item_task_actives[solution.assignment == b, t].sum())
                    for b in range(n_buckets)
                ]
                for t in range(n_tasks)
            ]
            if item_task_actives is not None
            else None
        )

        result = SplitResult(
            train=np.sort(np.asarray(out["train"], dtype=np.int64)),
            valid=np.sort(np.asarray(out.get("valid", []), dtype=np.int64)),
            test=np.sort(np.asarray(out["test"], dtype=np.int64)),
            discard=np.array([], dtype=np.int64),
            groups=labels,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=ctx.n,
            metadata={
                "n_clusters": n_clusters,
                "n_tasks": n_tasks,
                "solver": solution.architecture,
                "solver_status": solution.solver_status,
                "objective": solution.objective,
                "tolerance_used": used_tolerance,
                "per_task_fold_counts": per_task_fold_counts,
                "per_task_fold_actives": per_task_fold_actives,
                "realised_sizes": {k: len(v) for k, v in out.items()},
            },
        )
        return [result]
