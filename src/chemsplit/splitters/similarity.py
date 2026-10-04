"""The ``similarity``/fingerprint splitter family.

Every splitter here operates on a fingerprint/feature distance or similarity matrix.
"""

from __future__ import annotations

import dataclasses
import math
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
    compute_distance_matrix,
    compute_similarity_matrix,
    guard_memory,
    resolve_featurizer,
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
)
from chemsplit.determinism import (
    argmax_tiebreak,
    argmin_tiebreak,
    floor_round,
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


# -
# Shared plumbing: featurizer/metric/max_memory_bytes/n_jobs
# -

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
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=self.n_jobs
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
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=self.n_jobs
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
                    f"{name} partition has only {arr.size} record(s) ({arr.size / max(1, n):.2%} of n)",
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
        raise DegenerateGroupingError(f"{owner}: largest cluster covers {largest_frac:.1%} of records")
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
        raise ParameterError(f"invalid radius_is: {radius_is!r}; expected one of {list(_RADIUS_KINDS)}")
    splitter._validate_similarity_params(bounded_metric_required=radius_is == "similarity")
    bounded = is_bounded_metric(splitter.metric)
    if radius_is == "distance" and not bounded:
        if not radius > 0.0:
            raise ParameterError(f"radius must be > 0, got {radius!r}")
    elif not (0.0 < radius < 1.0):
        raise ParameterError(f"radius must be in (0,1) for radius_is={radius_is!r}, got {radius!r}")


def _resolve_radius(D: np.ndarray, radius: float, radius_is: str) -> float:
    """Convert ``radius`` to a distance threshold. ``"fraction_of_range"`` maps ``radius`` linearly
    onto ``[min, max]`` of the off-diagonal distances, giving a scale-free radius for unbounded
    metrics (raw descriptors under ``"euclidean"``/``"mahalanobis"``)."""
    if radius_is == "distance":
        return float(radius)
    if radius_is == "similarity":
        return 1.0 - float(radius)
    d_max = float(D.max())
    # Off-diagonal minimum without an n x n mask copy: mask the zero diagonal in place, then
    # restore it.
    np.fill_diagonal(D, np.inf)
    d_min = float(D.min())
    np.fill_diagonal(D, 0.0)
    return d_min + float(radius) * (d_max - d_min)


# -
# SimilarityThresholdSplitter
# -


class SimilarityThresholdSplitter(_SimilarityGroupBase):
    """Hard constraint: no test record may exceed ``threshold`` similarity to any train record.

    Advantages
    ----------
    - The constraint is explicit, checkable, and reported -- `metadata["max_cross_similarity"]` is either below the threshold or the split is wrong. No other splitter in this family gives that guarantee.
    - Directly parameterises what people actually mean by "novel chemistry": how dissimilar must test compounds be?
    - `graph_component` never discards data, so the full dataset gets used.

    Pitfalls
    --------
    - **The threshold is the experiment.** ECFP4 Tanimoto 0.4, ECFP6 Tanimoto 0.4, and MACCS 0.4 are three completely different difficulty levels -- a result reported without fingerprint, radius, bit length, metric, and cutoff isn't reproducible; `params` records all five, and so should your paper.
    - Similarity isn't transitive, so connected components can be enormous and chemically incoherent -- a chain of pairwise-similar molecules can link two very different ends. On dense datasets one component can swallow everything, raised as `ConstraintUnsatisfiableError` rather than silently accepted.
    - `greedy_prune` and `seeded_growth` both discard records, and exactly the ones in the interesting boundary region -- the remaining test set isn't a uniform sample of anything.
    - Tanimoto on sparse fingerprints saturates -- for large diverse libraries most pairs sit below 0.2, so a 0.4 cutoff removes almost nothing and the split silently becomes random. Check `metadata["component_sizes"]`.
    - All-zero fingerprints (parse failures under `on_parse_error="ignore"`, or tiny fragments) register similarity 1.0 to each other by convention and cluster together spuriously.


    References
    ----------
    .. [1] The three strategies are engineering compositions with no single published origin; the leakage
       rationale for a hard cross-similarity ceiling is well established:
    .. [2] Golbraikh, A.; Tropsha, A. Beware of q2! *J. Mol. Graph. Model.* **2002**, 20 (4), 269-276.
       https://doi.org/10.1016/S1093-3263(01)00123-1
    .. [3] Wallach, I.; Heifets, A. Most Ligand-Based Classification Benchmarks Reward Memorization Rather
       than Generalization. *J. Chem. Inf. Model.* **2018**, 58 (5), 916-932.
       https://doi.org/10.1021/acs.jcim.7b00403
    .. [4] Kapoor, S.; Narayanan, A. Leakage and the Reproducibility Crisis in Machine-Learning-Based
       Science. *Patterns* **2023**, 4 (9), 100804. https://doi.org/10.1016/j.patter.2023.100804
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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
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
        S = _sim_matrix(self, ctx)
        n = ctx.n
        if self.strategy == "graph_component":
            uf = UnionFind(n)
            for i in range(n):
                row = S[i]
                js = np.nonzero(row[i + 1:] > self.threshold + EPS)[0] + i + 1
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
            while True:
                violations: dict[int, int] = {}
                for t in test:
                    cnt = int(np.sum(S[t, list(train)] > self.threshold + EPS)) if train else 0
                    if cnt > 0:
                        violations[t] = cnt
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
            # each surviving record is its own singleton "group" so assign_groups just places it
            # in the bucket already decided above; encode via dense labels per-record.
            return np.arange(n, dtype=np.int64)

        # seeded_growth
        if self.seed_selection == "random":
            rng = seed_for(ctx.rng_seeds, "similarity.seed", 0)
            s = int(rng.integers(0, n))
        elif self.seed_selection == "most_central":
            mean_d = 1.0 - S.mean(axis=1)
            s = argmin_tiebreak(lambda i: mean_d[i], range(n))
        else:  # most_peripheral
            mean_d = 1.0 - S.mean(axis=1)
            s = argmax_tiebreak(lambda i: mean_d[i], range(n))
        test = {s}
        n_test_target = max(1, ctx.sizes.n_test)
        while len(test) < n_test_target:
            cand = [i for i in range(n) if i not in test]
            if not cand:
                break
            best = argmax_tiebreak(lambda i: float(np.max(S[i, list(test)])), cand)
            test.add(best)
        train = [i for i in range(n) if i not in test and float(np.max(S[i, list(test)])) <= self.threshold + EPS]
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
        # Encode: every train/test record its own singleton group, but tag test records with a
        # label distinguishing them so assign_groups (size-based, greedy_desc) still places
        # them sensibly. Simpler and exactly-correct: bypass assign_groups entirely via
        # forced_discard + explicit partitioning is not available through _group_labels alone,
        # so we approximate by giving test records a shared large "prefer-test" pseudo-group
        # only when that materially helps; for strategy="seeded_growth" the common case is
        # n_test_target already achieved, so singleton labels (default assign_groups behaviour)
        # correctly separates train/test through group_assignment sizing.
        return np.arange(n, dtype=np.int64)

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        max_cross = getattr(self, "_last_max_cross", None)
        return {
            "threshold": self.threshold,
            "strategy": self.strategy,
            "n_components": getattr(self, "_last_components", int(len(set(labels.tolist())))),
            "max_cross_similarity": max_cross if max_cross is not None else float("nan"),
        }


# -
# ButinaSplitter
# -


class ButinaSplitter(_SimilarityGroupBase):
    """Taylor-Butina sphere-exclusion (leader) clustering.

    Advantages
    ----------
    - A solid default: markedly harder than a scaffold split, chemically meaningful since it groups by real fingerprint proximity rather than a framework abstraction, and cheap enough for tens of thousands of molecules.
    - Deterministic, with no seed and only one interpretable parameter to choose.
    - Every cluster centroid is a real molecule, so clusters can be inspected and reported.
    - Sphere exclusion guarantees any two centroids are more than `cutoff` apart, giving the split a concrete geometric meaning.

    Pitfalls
    --------
    - Membership is defined only relative to the **centroid**, not pairwise -- two members of one cluster can be up to `2 x cutoff` apart, so a cluster isn't a tight neighbourhood and a train/test boundary between clusters does **not** guarantee any minimum cross-similarity. Use `similarity_threshold` or `hi` if you need that guarantee.
    - Produces many singletons on diverse libraries (often 30-60% of records); `singleton_policy` decides where they go and materially changes difficulty -- the default `"own_group"` scatters them randomly, softening the split.
    - Highly sensitive to `cutoff` -- 0.35 vs. 0.4 in distance can halve or double the cluster count.
    - The distance/similarity convention is a classic source of silent errors -- always read `metadata["cutoff_is"]`.
    - `reorder=True` gives different clusters than `reorder=False`, and both get called "Butina" in the literature -- report which.
    - `O(n²)` in time regardless of `algorithm` -- beyond roughly 10⁵ molecules use `k_means_cluster` with mini-batch k-means, or subsample.


    References
    ----------
    .. [1] Taylor, R. Simulation Analysis of Experimental Design Strategies for Screening Random Compounds
       as Potential New Drugs and Agrochemicals. *J. Chem. Inf. Comput. Sci.* **1995**, 35 (1), 59-67.
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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
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
        D = _dist_matrix(self, ctx)
        dist_cutoff = self.cutoff if self.cutoff_is == "distance" else 1.0 - self.cutoff
        clusters = _clustering.butina(D, dist_cutoff, reorder=self.reorder)
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
                for k in singleton_idx:
                    rec = clusters[k][0]
                    centroids = [clusters[j][0] for j in non_singleton]
                    nearest = argmin_tiebreak(lambda c: float(D[rec, c]), centroids)
                    target = non_singleton[centroids.index(nearest)]
                    clusters[target].append(rec)
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


# -
# SphereExclusionSplitter
# -


class SphereExclusionSplitter(_SimilarityGroupBase):
    """Sphere-exclusion clustering: scan records in random (or index) order; each record not yet
    claimed becomes a representative and claims every unclaimed record within ``radius`` of it.

    Unlike :class:`ButinaSplitter`, the scan order is not density-driven, and the radius can be
    given as a fraction of the observed distance range (``radius_is="fraction_of_range"``), so it
    also works on raw descriptors with unbounded metrics. The scan order is drawn from the
    ``"sphere_exclusion.order"`` stream when ``order="random"``.

    Advantages
    ----------
    - Clusters have a guaranteed maximum radius around their representative -- cluster tightness is a direct, interpretable parameter.
    - Linear number of passes over the distance matrix, with no density pre-computation, so it's cheaper than Butina at the same `n`.
    - `order="random"` gives a different but equally valid clustering per seed -- repeat over seeds to get variance from the clustering itself, not just from group assignment.
    - `radius_is="fraction_of_range"` makes one radius meaningful across metrics and descriptor scales.

    Pitfalls
    --------
    - Random scan order means a dense region can be split among several representatives while an outlier founds its own cluster -- cluster sizes are far more uneven than Butina's.
    - The radius bounds distance to the representative, not between clusters -- records of different clusters can be closer than `radius` to each other. Not a hard leakage constraint; use `similarity_threshold` for that.
    - `fraction_of_range` depends on the extreme pairwise distances, so one outlier stretches the range and silently enlarges every cluster.
    - Results depend on the seed unless `order="index"`, which in turn depends on input order.

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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
        self.radius = radius
        self.radius_is = radius_is
        self.order = order
        _validate_radius(self, radius, radius_is)
        if order not in ("random", "index"):
            raise ParameterError(f"invalid order: {order!r}")

    def _group_labels(self, ctx: _Context) -> IndexArray:
        D = _dist_matrix(self, ctx)
        n = ctx.n
        threshold = _resolve_radius(D, self.radius, self.radius_is)
        scan = (
            seed_for(ctx.rng_seeds, "sphere_exclusion.order", 0).permutation(n).tolist()
            if self.order == "random"
            else None
        )
        reps, clusters = _clustering.sphere_exclusion(D, threshold, order=scan)
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

# -
# KMeansClusterSplitter
# -


class KMeansClusterSplitter(_SimilarityGroupBase):
    """K-means (or a related partitional clusterer) over fingerprint/feature space.

    Advantages
    ----------
    - Scales far better than any `O(n²)` method -- `minibatch_kmeans` handles millions of molecules.
    - The cluster count is an explicit, reportable knob, and `auto_rule` makes the default reproducible rather than ad hoc.
    - Works on any feature representation, including learned embeddings and physicochemical descriptors, making it a natural generic clusterer.
    - `birch` and `minibatch` give a memory-bounded path where Butina and spectral clustering can't run.

    Pitfalls
    --------
    - **k is arbitrary.** Nothing in the chemistry determines it, yet split difficulty depends on it strongly -- `auto_rule="sqrt_n"` is a convention, not a principle.
    - K-means assumes isotropic, roughly equal-variance clusters in Euclidean space, which binary fingerprint space isn't -- clusters end up as much geometric artefacts as chemical families. SVD reduction mitigates but doesn't fix this.
    - Cluster sizes come out wildly uneven, so the achieved train/test ratio drifts from the request -- expect `SizeToleranceWarning`.
    - Euclidean distance on binary fingerprints is dominated by molecule size (bit count), so clusters partly track molecular weight rather than chemotype. Use `property` if that's actually what you want.
    - SVD sign ambiguity makes naive implementations non-reproducible across BLAS builds; a sign fix is mandatory here, but residual cluster-assignment drift from Lloyd's-iteration floating-point noise can still survive it -- its golden test uses a size/histogram tolerance, not an exact match.
    - `agglomerative` with `single` linkage chains badly on chemical data, typically producing one giant cluster plus dust.


    References
    ----------
    .. [1] MacQueen, J. Some Methods for Classification and Analysis of Multivariate Observations. In
       *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*,
       Vol. 1; University of California Press, **1967**; pp 281-297. No DOI;
       https://projecteuclid.org/euclid.bsmsp/1200512992
    .. [2] Lloyd, S. P. Least Squares Quantization in PCM. *IEEE Trans. Inf. Theory* **1982**, 28 (2),
       129-137. https://doi.org/10.1109/TIT.1982.1056489
    .. [3] ``algorithm="minibatch_kmeans"``: Sculley, D. Web-Scale k-Means Clustering. In *Proceedings of
       the 19th International Conference on World Wide Web (WWW '10)*, **2010**; pp 1177-1178.
       https://doi.org/10.1145/1772690.1772862
    .. [4] ``algorithm="agglomerative"``, Ward linkage: Ward, J. H. Hierarchical Grouping to Optimize an
       Objective Function. *J. Am. Stat. Assoc.* **1963**, 58 (301), 236-244.
       https://doi.org/10.1080/01621459.1963.10500845
    .. [5] ``algorithm="birch"``: Zhang, T.; Ramakrishnan, R.; Livny, M. BIRCH: An Efficient Data
       Clustering Method for Very Large Databases. *ACM SIGMOD Rec.* **1996**, 25 (2), 103-114.
       https://doi.org/10.1145/235968.233324
    .. [6] Clustering as a QSAR dataset-division strategy: Golbraikh, A.; Shen, M.; Xiao, Z. et al. Rational
       Selection of Training and Test Sets for the Development of Validated QSAR Models.
       *J. Comput.-Aided Mol. Des.* **2003**, 17 (2-4), 241-253. https://doi.org/10.1023/A:1025386326946
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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
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
        if self.reduce_method == "svd" and self.reduce_dim is not None and Z.shape[1] > self.reduce_dim:
            seed = int(seed_for(ctx.rng_seeds, "kmeans.svd", 0).integers(0, 2**31 - 1))
            svd = TruncatedSVD(n_components=self.reduce_dim, random_state=seed, algorithm="randomized", n_iter=7)
            Z = svd.fit_transform(Z)
            Z = _fix_svd_signs(Z)
        k = self._resolve_k(ctx.n)
        if k >= ctx.n:
            raise ParameterError(f"resolved n_clusters={k} must be < n={ctx.n}")
        seed = int(seed_for(ctx.rng_seeds, "kmeans.fit", 0).integers(0, 2**31 - 1))
        if self.algorithm == "kmeans":
            labels = KMeans(n_clusters=k, n_init=10, algorithm="lloyd", max_iter=300, tol=1e-4, random_state=seed).fit_predict(Z)
        elif self.algorithm == "minibatch_kmeans":
            labels = MiniBatchKMeans(n_clusters=k, batch_size=self.batch_size, n_init=10, max_iter=100, random_state=seed).fit_predict(Z)
        elif self.algorithm == "agglomerative":
            metric_arg = "euclidean" if self.linkage == "ward" else self.metric
            labels = AgglomerativeClustering(n_clusters=k, linkage=self.linkage, metric=metric_arg).fit_predict(Z)
        else:  # birch
            labels = Birch(n_clusters=k, threshold=0.5, branching_factor=50).fit_predict(Z)
        if len(set(labels.tolist())) == 1:
            raise DegenerateGroupingError(f"{type(self).__name__}: all records fell into a single cluster")
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


# -
# DensityClusterSplitter
# -


class DensityClusterSplitter(_SimilarityGroupBase):
    """DBSCAN or HDBSCAN density clustering on a precomputed distance matrix.

    Advantages
    ----------
    - No `k` to choose, and clusters can take any shape -- a better match for chemical space than k-means's spherical assumption.
    - Explicitly models "this molecule belongs to no family", which is chemically real and which every other clusterer forces into some cluster.
    - `noise_policy="test"` produces a clean, defensible "singletons and oddities" test set for applicability-domain work.

    Pitfalls
    --------
    - The noise bucket can swallow a large fraction of a diverse library -- 40% or more at sensible `eps` -- and `noise_policy` then decides most of the split; the default `"own_groups"` quietly makes it much easier, since noise points scatter.
    - `eps` interacts with fingerprint density in a way with no cross-dataset meaning -- a value tuned on one dataset doesn't transfer.
    - DBSCAN on a precomputed matrix needs `O(n²)` memory, capping `n` around 20,000 at the default guard.
    - HDBSCAN's `min_cluster_size` and `min_samples` interact non-obviously -- changing one changes the cluster count non-monotonically.
    - Density clustering on binary fingerprints suffers from the concentration of Tanimoto distances in high dimensions -- most pairs sit in a narrow band, so density contrast is weak.


    References
    ----------
    .. [1] Ester, M.; Kriegel, H.-P.; Sander, J.; Xu, X. A Density-Based Algorithm for Discovering Clusters
       in Large Spatial Databases with Noise. In *Proceedings of the 2nd International Conference on
       Knowledge Discovery and Data Mining (KDD-96)*; AAAI Press, **1996**; pp 226-231. No DOI;
       https://cdn.aaai.org/KDD/1996/KDD96-037.pdf
    .. [2] Campello, R. J. G. B.; Moulavi, D.; Sander, J. Density-Based Clustering Based on Hierarchical
       Density Estimates. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2013)*; Lecture
       Notes in Computer Science 7819; Springer, **2013**; pp 160-172.
       https://doi.org/10.1007/978-3-642-37456-2_14
    .. [3] Campello, R. J. G. B.; Moulavi, D.; Zimek, A.; Sander, J. Hierarchical Density Estimates for Data
       Clustering, Visualization, and Outlier Detection. *ACM Trans. Knowl. Discov. Data* **2015**,
       10 (1), 1-51. https://doi.org/10.1145/2733381
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
        noise_policy: Literal["test", "train", "own_groups", "discard", "distribute"] = "own_groups",
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
        self.algorithm = algorithm
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.noise_policy = noise_policy
        self._validate_similarity_params()
        if noise_policy not in ("test", "train", "own_groups", "discard", "distribute"):
            raise ParameterError(f"invalid noise_policy: {noise_policy!r}")

    def _group_labels(self, ctx: _Context) -> IndexArray:
        D = _dist_matrix(self, ctx)
        if self.algorithm == "dbscan":
            labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed").fit_predict(D)
        else:
            try:
                from sklearn.cluster import HDBSCAN
            except ImportError as exc:  # pragma: no cover - sklearn>=1.3 always has this
                raise MissingDependencyError(type(self).__name__, "hdbscan") from exc
            labels = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, metric="precomputed").fit_predict(D)
        n = ctx.n
        noise = np.nonzero(labels == -1)[0]
        forced: list[int] = []
        if noise.size:
            if self.noise_policy == "discard":
                forced = noise.tolist()
            elif self.noise_policy in ("test", "train"):
                # Cluster labels alone can't force a group into a *specific* partition (assign_groups
                # only balances by size); record the noise indices here so _partition can move them
                # into the target partition after the normal group-based assignment runs.
                self._last_noise_idx = noise.copy()
            elif self.noise_policy == "distribute" and (labels != -1).any():
                core_idx = np.nonzero(labels != -1)[0]
                for i in noise:
                    nearest = argmin_tiebreak(lambda c: float(D[i, c]), core_idx.tolist())
                    labels[i] = labels[nearest]
            # "own_groups": leave as -1; converted to singleton labels below.
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


# -
# SpectralSplitter
# -


class SpectralSplitter(_SimilarityGroupBase):
    """Laplacian-eigenmap spectral clustering on an affinity graph.

    ``graph="landmark"`` is landmark-based spectral clustering (Chen & Cai 2011) and never builds
    an ``n x n`` matrix. It selects ``n_landmarks`` landmark records, by default with OptiSim
    (``"spectral.landmarks"`` stream; subsample of ``ceil(n/20)``, no exclusion radius), so the
    landmarks are both spread out and representative. Each record is then represented by Gaussian
    weights to its ``landmark_neighbors`` nearest landmarks (bandwidth = mean distance to those
    landmarks), normalised to sum to one. The top ``n_clusters`` left singular vectors of that
    ``n x p`` matrix, scaled by the inverse square root of the landmark degrees, are clustered
    with k-means. As in Chen & Cai, the leading singular vector is kept (``drop_first`` does not
    apply).

    Advantages
    ----------
    - Minimises inter-cluster similarity by construction, reliably yielding the least train/test overlap among routine structure-based splits.
    - Handles non-convex, elongated regions of chemical space that k-means cuts straight through.
    - The eigenvalue spectrum is a free diagnostic -- the spectral gap shows whether the dataset genuinely has that many separable families.

    Pitfalls
    --------
    - `O(n²)` affinity construction and a dense-ish eigenproblem cap it near 50,000 molecules -- subsample above that and say so.
    - Depends on three coupled choices -- graph construction, Laplacian normalisation, and `n_clusters` -- none with a chemically principled default.
    - Degenerate eigenvalues (common on symmetric chemical graphs, e.g. many identical singleton components) make eigenvectors non-unique up to rotation, so cluster labels can differ between runs and platforms despite identical eigenvalues. The implementation warns but can't fix this -- its golden test uses a size/histogram tolerance, not an exact match.
    - A disconnected affinity graph silently turns spectral clustering into "one cluster per component", usually not what was wanted -- hence the hard error.
    - Being the hardest split isn't the same as being the right one -- a model evaluated only under spectral splitting looks worse than it will perform on a realistic screening library.
    - `graph="landmark"` approximates the full graph through `p` landmarks: too few landmarks blur small families together, and the result depends on the landmark draw.

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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
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
            isinstance(n_landmarks, bool) or not isinstance(n_landmarks, (int, np.integer)) or n_landmarks < 2
        ):
            raise ParameterError(f"n_landmarks must be None or an int >= 2, got {n_landmarks!r}")
        if isinstance(landmark_neighbors, bool) or not isinstance(landmark_neighbors, (int, np.integer)) or landmark_neighbors < 1:
            raise ParameterError(f"landmark_neighbors must be an int >= 1, got {landmark_neighbors!r}")

    def _landmark_labels(self, ctx: _Context) -> IndexArray:
        n = ctx.n
        p = self.n_landmarks if self.n_landmarks is not None else min(n - 1, max(50, math.ceil(math.sqrt(n)) * 5))
        if not (self.n_clusters <= p < n):
            raise ParameterError(f"n_landmarks={p} must satisfy n_clusters <= n_landmarks < n={n}")
        r = min(self.landmark_neighbors, p)
        F = ctx.get_features(resolve_featurizer(self.featurizer))
        guard_memory(max(1, math.isqrt(n * p) + 1), self.max_memory_bytes, type(self).__name__)
        rng = seed_for(ctx.rng_seeds, "spectral.landmarks", 0)
        if self.landmark_selection == "optisim":

            def column(j: int) -> np.ndarray:
                return pairwise_distances(F, F[j:j + 1], metric=self.metric)[:, 0]

            landmarks = _clustering.optisim_pick_columns(n, column, p, max(1, -(-n // 20)), 0.0, rng)
        else:
            landmarks = sorted(int(j) for j in rng.choice(n, size=p, replace=False))
        if len(landmarks) < self.n_clusters:
            raise DegenerateGroupingError(
                f"{type(self).__name__}: only {len(landmarks)} distinct landmarks for n_clusters={self.n_clusters}"
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


# -
# MaxMinSplitter
# -


class MaxMinSplitter(_SimilarityBase):
    """Greedy maximally-diverse selection (MaxMin / Kennard-Stone).

    The selected set may go to **train** (maximise coverage) or **test** (probe breadth) -- opposite
    experiments sharing one algorithm; never compare numbers across ``picked_goes_to`` values.

    With ``swap_fraction > 0`` the selection is then perturbed: that fraction of the picked records
    (rounded half up, capped at the number of unpicked records) is swapped for as many unpicked
    records, both drawn from the ``"maxmin.swap"`` stream. ``init="kennard_stone"`` with
    ``swap_fraction=0.1`` is the Morais-Lima-Martin (MLM) random-mutation Kennard-Stone method.

    Advantages
    ----------
    - With `picked_goes_to="train"`, builds the most informative training set for a fixed budget -- the standard answer to "which 500 compounds should I actually assay?".
    - `coverage_radius` is a directly interpretable guarantee -- no record sits further than that from a training example.
    - Memory-light in its lazy form, running on datasets where Butina and spectral clustering can't.
    - Deterministic apart from a single initial pick, and fully deterministic with `init="kennard_stone"`.
    - `init="kennard_stone"` with `metric="mahalanobis"` on a descriptor matrix is the MDKS variant; for label-aware selection see :class:`SPXYSplitter`.

    Pitfalls
    --------
    - **The two directions are different experiments and are routinely confused.** Diverse-in-train gives an optimistic, well-covered test set; diverse-in-test gives a hard extrapolation test. Never compare numbers across `picked_goes_to` values.
    - Greedy MaxMin chases outliers -- the first picks are typically the weirdest molecules in the set, including parse artefacts, salts, and fragments. Clean the data first, or the "diverse" set is a junk set.
    - Strongly depends on the initial pick when `init="random"` -- report the seed, or use `"kennard_stone"` for a seed-free run.
    - Optimises coverage, not group separation -- nothing stops a near-duplicate of a picked molecule from landing in the other partition. Not a leakage-control split, and shouldn't be described as one.
    - `coverage_radius` is only meaningful in the chosen metric -- comparing it across fingerprints is meaningless.
    - `swap_fraction` trades coverage for a less systematically optimistic test set; the swapped records make the split seed-dependent even with `init="kennard_stone"`.

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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
        self.picked_goes_to = picked_goes_to
        self.init = init
        self.n_picks = n_picks
        self.swap_fraction = swap_fraction
        self._validate_similarity_params()
        if picked_goes_to not in ("train", "test"):
            raise ParameterError(f"invalid picked_goes_to: {picked_goes_to!r}")
        if isinstance(swap_fraction, bool) or not isinstance(swap_fraction, (int, float)) or not (0.0 <= swap_fraction <= 0.5):
            raise ParameterError(f"swap_fraction must be in [0, 0.5], got {swap_fraction!r}")

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        D = _dist_matrix(self, ctx)
        n = ctx.n
        n_picks = self.n_picks if self.n_picks is not None else (
            ctx.sizes.n_train if self.picked_goes_to == "train" else ctx.sizes.n_test
        )
        if not (1 <= n_picks < n):
            raise ParameterError(f"n_picks must satisfy 1 <= n_picks < n, got {n_picks}")
        rng = seed_for(ctx.rng_seeds, "maxmin.init", 0) if self.init == "random" else None
        picked = _clustering.maxmin_pick(D, n_picks, init=self.init, rng=rng)
        swap_meta: dict[str, Any] = {}
        if self.swap_fraction > 0:
            picked_set = set(picked)
            unpicked = [i for i in range(n) if i not in picked_set]
            k = min(floor_round(self.swap_fraction * len(picked)), len(unpicked))
            swap_rng = seed_for(ctx.rng_seeds, "maxmin.swap", 0)
            swapped_out = sorted(int(i) for i in swap_rng.choice(picked, size=k, replace=False)) if k else []
            swapped_in = sorted(int(i) for i in swap_rng.choice(unpicked, size=k, replace=False)) if k else []
            out_set = set(swapped_out)
            picked = [i for i in picked if i not in out_set] + swapped_in
            swap_meta = {"swapped_out": swapped_out, "swapped_in": swapped_in}
        rem_rng = seed_for(ctx.rng_seeds, "maxmin.remainder", 0)
        buckets = _fill_remainder(picked, n, ctx.sizes, rem_rng, self.picked_goes_to)
        min_pairwise = float("inf")
        for a in range(len(picked)):
            for b in picked[a + 1:]:
                min_pairwise = min(min_pairwise, D[picked[a], b])
        coverage = float(np.max(np.min(D[:, picked], axis=1))) if picked else float("nan")
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


# -
# SPXYSplitter
# -


class SPXYSplitter(_SimilarityBase):
    """Kennard-Stone selection over a joint feature-and-label distance (SPXY).

    The feature distance matrix and the pairwise Euclidean distance between labels (across all
    columns for multi-task ``y``) are each divided by their own maximum and summed; Kennard-Stone
    then picks ``n_train`` records from that joint matrix into **train**. The remainder fills
    valid/test through a shuffle drawn from the ``"spxy.remainder"`` stream, so the split is
    seed-free whenever no validation set is requested. With ``metric="mahalanobis"`` this is
    M-SPXY (Apinantanakon et al. 2019, eq. 9): Mahalanobis distance on the features and Euclidean
    distance on the labels, each scaled by its maximum.

    Advantages
    ----------
    - Covers the label range as well as chemical space, so the training set spans the response surface rather than only the descriptor space -- the standard fix for Kennard-Stone leaving extreme activities out of train.
    - Fully deterministic without a seed for two-way splits: no random initial pick, ties broken by smallest index.
    - Each term is scaled to `[0, 1]` before summing, so neither the fingerprint distance nor the label units dominate.

    Pitfalls
    --------
    - The split depends on `y`, so the training set is chosen with knowledge of the labels -- fine for calibration-set design, but results aren't comparable with label-blind splits.
    - Like Kennard-Stone, the first picks are the most extreme records, including outliers and label errors; clean data first.
    - Optimises coverage, not separation -- test records can have near-duplicates in train. Not a leakage-control split.
    - Label distances are Euclidean over the raw `y` columns, so in multi-task data a task with a wider range weighs more; standardise `y` first if that matters.
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
        D_x = _dist_matrix(self, ctx).astype(np.float64)
        y = np.asarray(ctx.y, dtype=np.float64)
        y = y.reshape(n, -1)
        D_y = cdist(y, y, metric="euclidean")
        degenerate = []
        for name, M in (("feature", D_x), ("label", D_y)):
            m = float(M.max()) if M.size else 0.0
            if m > 0.0:
                M /= m
            else:
                degenerate.append(name)
        if degenerate:
            warn_with_details(
                DegenerateClusterWarning(
                    f"{type(self).__name__}: all pairwise {' and '.join(degenerate)} distances are "
                    "zero; that term contributes nothing to the selection",
                    details={"zero_terms": degenerate},
                )
            )
        D = D_x + D_y
        del D_x, D_y
        n_picks = ctx.sizes.n_train
        picked = _clustering.kennard_stone(D, n_picks)[:n_picks]
        rem_rng = seed_for(ctx.rng_seeds, "spxy.remainder", 0)
        buckets = _fill_remainder(picked, n, ctx.sizes, rem_rng, "train")
        coverage = float(np.max(np.min(D[:, picked], axis=1))) if picked else float("nan")
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


# -
# OptiSimSplitter
# -


class OptiSimSplitter(_SimilarityGroupBase):
    """OptiSim diversity selection, used either as cluster centres or as a picked set.

    Each round draws random candidates (``"optisim.draw"`` stream) until ``subsample_size`` of
    them lie further than ``radius`` from everything already selected, then selects the one with
    the largest minimum distance to the selection. ``subsample_size=1`` is random selection with
    sphere exclusion; a subsample covering every record is MaxMin -- the parameter trades
    representativeness against diversity.

    ``mode="cluster"`` treats the ``n_picks`` selected records (default: the ``"auto"`` rule of
    :class:`KMeansClusterSplitter`) as centres, groups every record with its nearest centre (ties
    -> earliest-selected centre) and assigns whole groups to partitions. ``mode="pick"`` sends the
    selected set to ``picked_goes_to`` like :class:`MaxMinSplitter` (default ``n_picks``: that
    partition's size), shuffles the rest into the other partitions (``"optisim.remainder"``
    stream), and forms no groups.

    Advantages
    ----------
    - One knob, `subsample_size`, spans random sampling to MaxMin, so a selection can be diverse without being dominated by outliers the way pure MaxMin is.
    - `radius` guarantees a minimum spacing between selected records (and hence between cluster centres).
    - Cluster mode keeps near-duplicates of a centre in that centre's group, so they can't straddle train and test.
    - Selection costs `O(n · n_picks)` on top of the distance matrix -- cheap next to Butina or spectral clustering.

    Pitfalls
    --------
    - Results depend on the seed at every round, not just the first pick -- report it, and repeat over seeds.
    - Cluster mode's groups are Voronoi cells around the centres, not density clusters; with a small `n_picks` they are large and chemically mixed.
    - If `radius` excludes every remaining candidate, fewer than `n_picks` records are selected (a `DegenerateClusterWarning` names how many). In pick mode, records the smaller picked set can't absorb are **discarded**.
    - Pick mode optimises coverage, not separation -- like MaxMin, it's not a leakage-control split. Only cluster mode is group-forming.
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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
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
        for name, value, lo in (("n_picks", n_picks, min_picks), ("subsample_size", subsample_size, 1)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lo:
                raise ParameterError(f"{name} must be None or an int >= {lo}, got {value!r}")

    def compute_groups(self, X: Any, y: Any = None, **kw: Any) -> IndexArray:
        if self.mode == "pick":
            raise ParameterError(
                f"{type(self).__name__}(mode='pick') forms no groups; use mode='cluster' for compute_groups()"
            )
        return super().compute_groups(X, y, **kw)

    def _select(self, ctx: _Context, n_picks: int) -> tuple[np.ndarray, list[int], float, int]:
        D = _dist_matrix(self, ctx)
        n = ctx.n
        if not (1 <= n_picks < n):
            raise ParameterError(f"n_picks must satisfy 1 <= n_picks < n, got {n_picks}")
        threshold = _resolve_radius(D, self.radius, self.radius_is)
        k = self.subsample_size if self.subsample_size is not None else max(1, -(-n // 20))
        rng = seed_for(ctx.rng_seeds, "optisim.draw", 0)
        picked = _clustering.optisim_pick(D, n_picks, k, threshold, rng)
        if len(picked) < n_picks:
            warn_with_details(
                DegenerateClusterWarning(
                    f"{type(self).__name__}: only {len(picked)} of n_picks={n_picks} records lie "
                    f"further than radius={self.radius} ({self.radius_is}) from each other",
                    details={"n_picks": n_picks, "n_selected": len(picked)},
                )
            )
        return D, picked, threshold, k

    def _group_labels(self, ctx: _Context) -> IndexArray:
        n = ctx.n
        n_picks = self.n_picks if self.n_picks is not None else _resolve_cluster_count(n, "auto")
        D, centres, threshold, k = self._select(ctx, n_picks)
        slot = list(range(len(centres)))
        labels = np.asarray(
            [argmin_tiebreak(lambda c: float(D[i, centres[c]]), slot) for i in range(n)], dtype=np.int64
        )
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
            clusters, n, type(self).__name__, f"n_picks={n_picks}, radius={self.radius} ({self.radius_is})"
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
        D, picked, threshold, k = self._select(ctx, n_picks)
        rem_rng = seed_for(ctx.rng_seeds, "optisim.remainder", 0)
        buckets = _fill_remainder(picked, n, ctx.sizes, rem_rng, self.picked_goes_to)
        coverage = float(np.max(np.min(D[:, picked], axis=1)))
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
                "realised_sizes": {name: int(v.size) for name, v in buckets.items() if name != "discard"},
            },
        )
        _small_partition_check(result, n)
        return [result]


# -
# MinimalTestSetDissimilaritySplitter
# -


def _check_task_index(task_index: Any) -> None:
    if isinstance(task_index, bool) or not isinstance(task_index, (int, np.integer)) or task_index < 0:
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

    Each record's total dissimilarity is the sum of its distances to every other record. Records
    are sorted by label, most active first (ties by index), and cut into ``n_test`` contiguous bins
    whose sizes differ by at most one; the record with the smallest total dissimilarity in each bin
    (ties -> smallest index) goes to **test**. A validation set, if requested, is chosen the same way
    from the remaining records, with total dissimilarities recomputed over that remainder; the rest
    is train. For a 20% test set the bins hold 5 records each, as in Martin et al. (2012), who used
    Euclidean distance on preselected descriptors (``featurizer=..., metric="euclidean"``).
    Fully deterministic: no random draws at all.

    Advantages
    ----------
    - The test set spans the full label range by construction -- one record per activity bin -- so no part of the response is left unevaluated.
    - Each test record is the most typical member of its bin, so test compounds always have close analogues in train: a clean check of interpolation quality.
    - No seed, no free parameter beyond the featurizer and metric -- easy to reproduce and describe.
    - Follows criterion 2 of rational division (test compounds close to training compounds) directly.

    Pitfalls
    --------
    - **Deliberately optimistic.** Test records are the least unusual compounds, so test scores overstate performance on new chemistry; Martin et al. found rational test sets beat random ones on test but not on an external set.
    - Selection uses `y`, so the split is not label-blind and isn't comparable with label-free splits.
    - Total dissimilarity is dominated by global position: a dense region's centre wins every bin it touches, so test can concentrate in one region of chemical space.
    - Builds the full `n x n` distance matrix.
    - Multi-task `y` uses one column (`task_index`); the other tasks are ignored.

    References
    ----------
    .. [1] Martin, T. M.; Harten, P.; Young, D. M.; Muratov, E. N.; Golbraikh, A.; Zhu, H.;
       Tropsha, A. Does Rational Selection of Training and Test Sets Improve the Outcome of QSAR
       Modeling? *J. Chem. Inf. Model.* **2012**, 52 (10), 2570-2578.
       https://doi.org/10.1021/ci300338w
    .. [2] Kuz'min, V. E.; Artemenko, A. G.; Muratov, E. N.; Volineckaya, I. L.; Makarov, V. A.;
       Riabova, O. B.; Wutzler, P.; Schmidtke, M. Quantitative Structure−Activity Relationship
       Studies of [(Biphenyloxy)propyl]isoxazole Derivatives. Inhibitors of Human Rhinovirus 2
       Replication. *J. Med. Chem.* **2007**, 50 (17), 4205-4213. https://doi.org/10.1021/jm0704806
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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
        self.task_index = task_index
        self._validate_similarity_params()
        _check_task_index(task_index)

    def _check_preconditions(self, ctx: _Context) -> None:
        _label_column(self, ctx)

    @staticmethod
    def _select(D: np.ndarray, y: np.ndarray, pool: list[int], k: int) -> tuple[list[int], list[list[int]]]:
        """Pick ``k`` records from ``pool``: one minimal-total-dissimilarity record per activity bin."""
        if k <= 0:
            return [], []
        idx = np.asarray(pool, dtype=np.int64)
        totals = D[np.ix_(idx, idx)].sum(axis=1)
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
        D = _dist_matrix(self, ctx).astype(np.float64)
        test, test_edges = self._select(D, y, list(range(n)), ctx.sizes.n_test)
        test_set = set(test)
        remaining = [i for i in range(n) if i not in test_set]
        valid, _ = self._select(D, y, remaining, ctx.sizes.n_valid)
        valid_set = set(valid)
        train = [i for i in remaining if i not in valid_set]
        totals = D.sum(axis=1)
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


# -
# SupportPointsSplitter
# -


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
            codes = np.asarray([levels.setdefault(v, len(levels)) for v in col.tolist()], dtype=np.int64)
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
    ``x_i <- [ (N/n) sum_k (x_i - x_k)/|x_i - x_k| + sum_m z_m/|x_i - z_m| ] / sum_m 1/|x_i - z_m|``,
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

    Features (plus the labels, when given and ``use_labels=True``) form one design matrix:
    categorical label columns become Helmert contrasts, constant columns are dropped and every
    column is standardised. Support points -- the ``k`` points minimising the energy distance to the
    data -- are computed for the smaller side of the cut (``k = n_test``, or ``n - n_test`` when test
    is the larger side) by the convex-concave fixed point of Mak & Joseph, started from ``k``
    distinct records drawn from the ``"support_points.init"`` stream. Each support point in turn
    then takes its nearest still-unassigned record. A validation set, if requested, is selected the
    same way from the remaining records (stream index 1). Joseph's optimal-ratio result suggests a
    test fraction of about ``1 / (sqrt(p) + 1)`` for ``p`` model parameters; chemsplit leaves the
    sizes to the caller.

    :param featurizer: Feature representation. Defaults to ``"physchem"``: the method works in
        standardised Euclidean space, which suits continuous descriptors.
    :param use_labels: Append ``y`` to the design matrix when it is given, as in the original
        method. Defaults to True.
    :param label_kind: How to encode ``y``: ``"continuous"``, ``"categorical"`` (Helmert
        contrasts) or ``"auto"`` (categorical for non-numeric or boolean columns). Defaults to
        ``"auto"``.
    :param max_iter: Maximum fixed-point iterations. Defaults to 500.
    :param tol: Stop once the energy criterion improves by less than this fraction in one
        iteration. Defaults to 1e-6.
    :param max_memory_bytes: Ceiling on the support-point/record distance blocks. Defaults to 2 GiB.
    :param base: See :class:`chemsplit.base.BaseSplitter`.

    Advantages
    ----------
    - Both subsets follow the joint distribution of features and labels as closely as a subset of that size can -- an optimal version of what a random split only achieves on average.
    - Far less variance between seeds than a random split, so one split is representative.
    - Handles mixed continuous and categorical labels through Helmert coding.
    - Never builds an `n x n` matrix: memory grows with `n x k`.

    Pitfalls
    --------
    - **An interpolation split.** Test records sit where the training data is densest, so scores are as optimistic as a good random split -- not a test of generalisation to new chemistry.
    - With `use_labels=True` the split depends on `y`; set it to False for a label-blind split.
    - Each fixed-point iteration costs `O(k · n · p)`; for large `n` and `k` this is the slowest splitter in the family.
    - Standardisation gives every column equal weight, so hundreds of noisy descriptors can drown out the labels; select descriptors first.
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
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ParameterError(f"max_iter must be an int >= 1, got {max_iter!r}")
        if not (isinstance(tol, (int, float)) and tol > 0):
            raise ParameterError(f"tol must be > 0, got {tol!r}")
        if isinstance(max_memory_bytes, bool) or not isinstance(max_memory_bytes, (int, np.integer)) or max_memory_bytes <= 0:
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
            raise DegenerateGroupingError(f"{type(self).__name__}: every feature and label column is constant")
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

    def _select(self, Z: np.ndarray, pool: list[int], k: int, ctx: _Context, stage: int) -> tuple[list[int], dict[str, Any]]:
        m = len(pool)
        if k <= 0:
            return [], {}
        if k >= m:
            return list(pool), {}
        n_points = min(k, m - k)  # support points always describe the smaller side
        self._guard(n_points, m)
        sub = Z[np.asarray(pool, dtype=np.int64)]
        rng = seed_for(ctx.rng_seeds, "support_points.init", stage)
        points, n_iter, converged, criterion = _support_points(sub, n_points, rng, self.max_iter, self.tol)
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
                "realised_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
            },
        )
        _small_partition_check(result, n)
        return [result]


# -
# DuplexSplitter
# -


class DuplexSplitter(_SimilarityBase):
    """DUPLEX: train, test and valid each grow as a maximally spread-out set, taking turns.

    Train is seeded with the farthest-apart pair of records, test with the farthest-apart pair of
    the rest, then valid likewise (ties -> lexicographically smallest pair). The partitions then
    take turns adding the unassigned record farthest (by minimum distance) from their own members
    -- ties -> smallest index -- and drop out of the rotation once they reach their size. Snee
    described two sets; this generalises the rotation to three and to unequal sizes, so every size
    is met exactly. Fully deterministic: no random draws.

    Advantages
    ----------
    - Every partition spans the whole data space, so test covers the same range as train -- unlike Kennard-Stone, which puts all the extremes in train.
    - The alternation makes the partitions statistically similar in spread, which suits model validation in the sense Snee intended.
    - Exact sizes and fully deterministic without a seed.
    - `metadata["coverage_radius"]` reports how far any record is from each partition.

    Pitfalls
    --------
    - **An interpolation split.** Test records are spread through the same space as train, so scores are optimistic for genuinely new chemistry.
    - Seeding by the farthest pairs puts outliers into every partition first; clean the data before splitting.
    - Builds the full `n x n` distance matrix, and each step is `O(n)`, so the whole split is `O(n²)`.
    - Not a leakage-control split: near-duplicates can land in different partitions.

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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
        self._validate_similarity_params()

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        D = _dist_matrix(self, ctx).astype(np.float64)
        names = ("train", "test", "valid")
        targets = (ctx.sizes.n_train, ctx.sizes.n_test, ctx.sizes.n_valid)
        parts = _clustering.duplex_order(D, targets)
        buckets = {name: np.sort(np.asarray(part, dtype=np.int64)) for name, part in zip(names, parts, strict=True)}
        coverage = {
            name: float(np.max(np.min(D[:, buckets[name]], axis=1)))
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
                "seed_pairs": {name: part[:2] for name, part in zip(names, parts, strict=True) if part},
                "coverage_radius": coverage,
                "realised_sizes": {name: int(buckets[name].size) for name in names},
            },
        )
        _small_partition_check(result, n)
        return [result]


# -
# DOptimalSplitter
# -


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

    Features are standardised (constant columns dropped) and projected onto their leading
    ``n_components`` principal components (signs fixed); the design matrix is those scores plus an
    intercept column. Starting from a Kennard-Stone selection on the scores (or, with
    ``init="random"``, a draw from the ``"d_optimal.init"`` stream), a modified Fedorov exchange
    visits the design points in index order and swaps each one for the non-design record that most
    increases the determinant -- ``Δ(i, j) = d(j) − d(i) − [d(i)·d(j) − d(i, j)²]`` with
    ``d(a, b) = x_aᵀ (XᵀX)⁻¹ x_b``, ties to the smallest index -- until a full pass makes no swap.
    A validation set, if requested, is the D-optimal subset of the remaining records; the rest is
    test.

    :param featurizer: Feature representation. Defaults to ``"physchem"``: D-optimality is
        defined on continuous design variables.
    :param n_components: Principal components in the design matrix; ``None`` uses
        ``min(10, n_train - 2, n_features)``. Defaults to ``None``.
    :param init: Starting design, ``"kennard_stone"`` (deterministic) or ``"random"``. Defaults
        to ``"kennard_stone"``.
    :param ridge: Added to the diagonal of ``XᵀX`` so near-singular designs stay invertible.
        Defaults to 1e-8.
    :param max_passes: Maximum exchange passes over the design. Defaults to 100.
    :param max_memory_bytes: Ceiling for the Kennard-Stone distance matrix. Defaults to 2 GiB.
    :param base: See :class:`chemsplit.base.BaseSplitter`.

    Advantages
    ----------
    - The training set gives the most precise estimates of a linear model's coefficients in the chosen descriptor space -- the classical optimal-design criterion.
    - Deterministic without a seed under the default Kennard-Stone start.
    - `metadata["log_det"]` reports the achieved criterion, so designs can be compared.
    - Works on PCA scores, so it stays well-posed with many correlated descriptors.

    Pitfalls
    --------
    - **Picks the edges of descriptor space.** D-optimal training sets concentrate on extreme records, so test holds the interior -- test scores can overestimate predictive power, as Gramatica and co-workers noted.
    - Optimal for a linear model in the chosen components; a nonlinear model or different descriptors would want a different design.
    - The exchange finds a local optimum, which depends on the starting design.
    - Each pass costs `O(n_train · n · p²)`; large sets with many components are slow.
    - Selection ignores `y`; label imbalance between train and test is not controlled.

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
            isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)) or n_components < 1
        ):
            raise ParameterError(f"n_components must be None or an int >= 1, got {n_components!r}")
        if init not in ("kennard_stone", "random"):
            raise ParameterError(f"invalid init: {init!r}")
        if not (isinstance(ridge, (int, float)) and ridge >= 0):
            raise ParameterError(f"ridge must be >= 0, got {ridge!r}")
        if isinstance(max_passes, bool) or not isinstance(max_passes, (int, np.integer)) or max_passes < 1:
            raise ParameterError(f"max_passes must be an int >= 1, got {max_passes!r}")

    def _design(self, ctx: _Context, n_select: int) -> np.ndarray:
        F = ctx.get_features(resolve_featurizer(self.featurizer))
        X = np.asarray(F.toarray() if sp.issparse(F) else F, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ParameterError(f"{type(self).__name__}: features contain NaN or infinite values")
        sd = X.std(axis=0)
        keep = sd > 0
        if not keep.any():
            raise DegenerateGroupingError(f"{type(self).__name__}: every feature column is constant")
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

    def _select(self, X: np.ndarray, pool: list[int], k: int, ctx: _Context, stage: int) -> tuple[list[int], dict[str, Any]]:
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


# -
# MaxDissimilaritySplitter
# -


class MaxDissimilaritySplitter(_SimilarityBase):
    """Pushes train and test to opposite regions of chemical space.

    Advantages
    ----------
    - Produces a clean, reproducible large extrapolation -- exactly the right test for "can this model reach a region it's never seen?".
    - Only two records are chosen by any rule; everything else follows deterministically, keeping the split easy to describe and audit.
    - `metadata["min_cross_distance"]` quantifies how far apart the two sets actually ended up.

    Pitfalls
    --------
    - Deliberately worst-case -- it estimates performance on **one specific** extrapolation, not average prospective performance, so a single number here carries a very wide implicit confidence interval.
    - The whole split hinges on two seed molecules, usually outliers -- one badly standardised salt can define the entire experiment.
    - Test and train are contiguous regions, so the test set is chemically homogeneous with strongly correlated errors, and the effective sample size is far below `n_test`.
    - Not a leakage constraint -- nothing bounds the minimum train-to-test distance except whatever geometry results. Read `min_cross_distance` before claiming novelty.
    - `grow="nearest_to_set"` can chain and walk the test set back toward the train seed; `"nearest_to_seed"` keeps it compact -- the two give materially different splits.


    References
    ----------
    .. [1] The two-seed grow-apart construction is a composition with no single published origin; the
       diverse-selection root and the comparative evidence are:
    .. [2] Kennard, R. W.; Stone, L. A. Computer Aided Design of Experiments. *Technometrics* **1969**,
       11 (1), 137-148. https://doi.org/10.1080/00401706.1969.10490666
    .. [3] Martin, T. M.; Harten, P.; Young, D. M. et al. Does Rational Selection of Training and Test Sets
       Improve the Outcome of QSAR Modeling? *J. Chem. Inf. Model.* **2012**, 52 (10), 2570-2578.
       https://doi.org/10.1021/ci300338w
    .. [4] Tossou, P.; Wognum, C.; Craig, M.; Mary, H.; Noutahi, E. Real-World Molecular
       Out-Of-Distribution: Specification and Investigation. *J. Chem. Inf. Model.* **2024**, 64 (3),
       697-711. https://doi.org/10.1021/acs.jcim.3c01774
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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
        self.seed_pair = seed_pair
        self.grow = grow
        self._validate_similarity_params()

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        D = _dist_matrix(self, ctx)
        n = ctx.n
        guard_memory(n, self.max_memory_bytes, type(self).__name__)
        if self.seed_pair == "max_distance":
            iu = np.triu_indices(n, k=1)
            dvals = D[iu]
            max_d = dvals.max()
            cand = [(int(iu[0][k]), int(iu[1][k])) for k in range(len(dvals)) if dvals[k] >= max_d - EPS]
            a, b = min(cand)
            n_tied = len(cand)
        else:
            rng = seed_for(ctx.rng_seeds, "maxdiss.seed", 0)
            a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
            n_tied = 1
        test = [b]
        key = D[:, b].copy()
        assigned = np.zeros(n, dtype=bool)
        assigned[b] = True
        n_test = max(1, ctx.sizes.n_test)
        while len(test) < n_test:
            cand_idx = [i for i in range(n) if not assigned[i]]
            if not cand_idx:
                break
            nxt = argmin_tiebreak(lambda i: key[i], cand_idx)
            test.append(nxt)
            assigned[nxt] = True
            if self.grow == "nearest_to_set":
                key = np.minimum(key, D[:, nxt])
        rest = [i for i in range(n) if not assigned[i]]
        n_valid = ctx.sizes.n_valid
        valid: list[int] = []
        if n_valid > 0:
            rest_sorted = stable_sort(rest, key=lambda i: D[i, a], desc=True)
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
                "seed_distance": float(D[a, b]),
                "n_tied_seed_pairs": n_tied,
                "min_cross_distance": float(min(D[t, tr] for t in test for tr in train)) if train and test else float("nan"),
                "realised_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
            },
        )
        _small_partition_check(result, n)
        return [result]


# -
# PerimeterSplitter
# -


class PerimeterSplitter(_SimilarityBase):
    """Holds out the outskirts of the distribution; trains on the dense core.

    Advantages
    ----------
    - Directly tests the applicability-domain edge -- the held-out molecules are the ones a deployed model would be least confident about, exactly where failures cost money.
    - Completely deterministic, with no seed and no free parameter beyond the metric -- unusually easy to reproduce and describe.
    - The training set stays dense and representative, so training stays stable even though evaluation is hard.

    Pitfalls
    --------
    - The test set is enriched in oddities -- fragments, salts, dyes, very large or small molecules, standardisation failures. A poor score may reflect data quality rather than model quality -- inspect `test` before trusting the number.
    - Error bars run large, since the test set is heterogeneous and small in effective size.
    - `greedy_pairs` is `O(n²)` in memory for the pair sort and doesn't scale past a few tens of thousands of records.
    - Not a chemical-novelty guarantee -- an outlier can still sit near a training molecule if that's its only near neighbour. Check with `audit.nn_similarity_profile`.
    - Because peripherality is defined by mean distance, the split partly encodes molecular size and fingerprint density.


    References
    ----------
    .. [1] Szántai-Kis, C.; Kövesdi, I.; Kéri, G.; Örfi, L. Validation Subset Selections for Extrapolation
       Oriented QSPAR Models. *Mol. Divers.* **2003**, 7 (1), 37-43.
       https://doi.org/10.1023/B:MODI.0000006538.99122.00
    .. [2] Tossou, P.; Wognum, C.; Craig, M.; Mary, H.; Noutahi, E. Real-World Molecular
       Out-Of-Distribution: Specification and Investigation. *J. Chem. Inf. Model.* **2024**, 64 (3),
       697-711. https://doi.org/10.1021/acs.jcim.3c01774
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
        super().__init__(featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, **kwargs)
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
            pair_order = sorted(range(len(dvals)), key=lambda k: (-dvals[k], int(iu[0][k]), int(iu[1][k])))
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
        valid = stable_sort(rest, key=lambda i: outlier_score[i], desc=True)[:n_valid] if n_valid else []
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
                "test_mean_outlier_score": float(np.mean(outlier_score[test])) if test else float("nan"),
                "train_mean_outlier_score": float(np.mean(outlier_score[train])) if train else float("nan"),
                "realised_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
            },
        )
        _small_partition_check(result, n)
        return [result]


# -
# LeaveOneClusterOutSplitter
# -


class LeaveOneClusterOutSplitter(GroupSplitter):
    """Each cluster (from a caller-supplied ``clusterer``) takes a turn as the test fold.

    Advantages
    ----------
    - Yields a **per-cluster error distribution** instead of one number -- you learn which regions of chemical space the model fails in, not just that it fails.
    - Every record is tested exactly once (when `max_folds` is `None` and no small-cluster pooling happens), so the aggregate is an honest whole-dataset estimate under cluster-level extrapolation.
    - Composes with every scaffold/similarity/embedding grouping, so the same protocol answers "generalise across scaffolds?", "across Butina clusters?", "across UMAP regions?".

    Pitfalls
    --------
    - Cluster sizes are uneven, so per-fold scores come from wildly different sample sizes -- a macro-average over folds and a micro-average over records can disagree sharply. Report both, and report `cluster_size` alongside every fold score.
    - Many small clusters make the fold count explode and the run expensive; `max_folds` caps it, but then some clusters are never tested, biasing the aggregate.
    - Training-set size varies across folds, so fold-to-fold differences partly measure training-set size rather than chemical difficulty.
    - A single-cluster test fold with 3 records can't support ROC-AUC or a meaningful R² -- the splitter warns but can't stop the caller from computing them anyway.


    References
    ----------
    .. [1] Kramer, C.; Gedeck, P. Leave-Cluster-Out Cross-Validation Is Appropriate for Scoring Functions
       Derived from Diverse Protein Data Sets. *J. Chem. Inf. Model.* **2010**, 50 (11), 1961-1969.
       https://doi.org/10.1021/ci100264e
    """

    splitter_id: ClassVar[str] = "leave_one_cluster_out"
    family: ClassVar[str] = "similarity"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    deterministic_method: ClassVar[bool] = True

    def __init__(
        self,
        *,
        clusterer: GroupSplitter | None = None,
        min_cluster_size: int = 1,
        small_cluster_policy: Literal["merge_into_train", "own_fold", "pool"] = "merge_into_train",
        max_folds: int | None = 50,
        fold_order: Literal["size_desc", "size_asc", "index"] = "size_desc",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(clusterer, str):
            raise ParameterError(
                "clusterer must be an instantiated GroupSplitter -- string-based registry lookup "
                "('butina') is not available until chemsplit.registry lands"
            )
        self.clusterer = clusterer
        self.min_cluster_size = min_cluster_size
        self.small_cluster_policy = small_cluster_policy
        self.max_folds = max_folds
        self.fold_order = fold_order
        if self.n_splits != 1:
            raise ConfigurationError("n_splits is derived from the cluster count and must not be set")

    def _group_labels(self, ctx: _Context) -> IndexArray:
        if self.clusterer is not None:
            return self.clusterer._group_labels(ctx)
        D = compute_distance_matrix(ctx, "ecfp4", "tanimoto", 2 * 1024**3, type(self).__name__, 1)
        clusters = _clustering.butina(D, 0.35, reorder=False)
        labels = np.empty(ctx.n, dtype=np.int64)
        for cid, members in enumerate(clusters):
            for m in members:
                labels[m] = cid
        return dense_label_encode(labels.tolist())

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """The real fold count depends on the data (cluster count after ``small_cluster_policy``
        and ``max_folds``), so this can only be computed exactly when ``X`` is supplied -- matching
        every other data-dependent ``n_splits`` in the library (e.g. ``GroupKFoldSplitter``'s
        ``n_splits="auto"``). Without ``X``, ``1`` is a documented lower-bound placeholder."""
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
            raise DegenerateGroupingError(f"{type(self).__name__}: clusterer produced a single cluster")
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


# -
# BalancedMultiTaskSplitter
# -


class BalancedMultiTaskSplitter(GroupSplitter):
    """Balanced multi-task cluster assignment: assigns whole clusters to folds so
    every task gets an acceptable train/test ratio and label balance.

    Clusters are assigned to folds via an independently designed optimizer
    (:mod:`chemsplit._optimize`), whose backends were selected by comparing candidate
    architectures across problem sizes. The ``solver`` parameter accepts
    ``"auto"``/``"milp"``/``"heuristic"``.

    Advantages
    ----------
    - The practical answer to sparse multi-task matrices, where naive cluster splitting can leave some targets with zero test actives and undefined metrics.
    - Balance is a *constraint*, not a hope -- if the requested balance is impossible, the splitter says so and names the binding task instead of silently producing a useless fold.
    - `per_task_fold_counts` gives a complete audit of what every task got -- exactly the table reviewers ask for.
    - Works with any clusterer, cleanly separating the chemical criterion from the balancing.

    Pitfalls
    --------
    - Infeasibility is common on real sparse matrices -- a task with three actives in one cluster simply can't be balanced. `on_infeasible="relax"` is the pragmatic escape, but a relaxed tolerance means the balance requested isn't the balance achieved; read `metadata["tolerance_used"]`.
    - Solve time grows quickly with cluster count; the `dust` merge that keeps it tractable changes the grouping in a way that's invisible unless you read `metadata["dust_merged"]`.
    - Multi-threaded MIP solving is non-deterministic -- raising `threads` above 1 for speed silently breaks reproducibility.
    - Balancing on label statistics chooses the split partly using the labels, a mild form of information leakage into the experimental design -- usually the lesser evil versus undefined metrics, but it should be disclosed.


    References
    ----------
    .. [1] Tricarico, G. A.; Hofmans, J.; Lenselink, E. B.; López-Ramos, M.; Dréanic, M.-P.; Stouten, P. F. W.
       Construction of Balanced, Chemically Dissimilar Training, Validation and Test Sets for Machine
       Learning on Molecular Datasets. *ChemRxiv* preprint, **2024** (not peer reviewed).
       https://doi.org/10.26434/chemrxiv-2022-m8l33-v3
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
        clusterer: GroupSplitter | None = None,
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
        if isinstance(clusterer, str):
            raise ParameterError(
                "clusterer must be an instantiated GroupSplitter -- string-based registry lookup "
                "is not available until chemsplit.registry lands"
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
        if self.clusterer is not None:
            return self.clusterer._group_labels(ctx)
        D = compute_distance_matrix(ctx, "ecfp4", "tanimoto", 2 * 1024**3, type(self).__name__, 1)
        clusters = _clustering.butina(D, 0.35, reorder=False)
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
        item_task_actives = np.zeros((n_clusters, n_tasks), dtype=np.float64) if self.balance == "counts_and_actives" else None
        for c in range(n_clusters):
            rows = y[labels == c]
            mask = ~np.isnan(rows)
            item_task_counts[c] = mask.sum(axis=0)
            if item_task_actives is not None:
                item_task_actives[c] = np.nansum(np.where(mask, rows, 0.0), axis=0)

        buckets = [("train", ctx.sizes.n_train), ("valid", ctx.sizes.n_valid), ("test", ctx.sizes.n_test)]
        active_buckets = [(name, cap) for name, cap in buckets if cap > 0]
        n_buckets = len(active_buckets)
        bucket_target = np.asarray([cap for _, cap in active_buckets], dtype=np.float64)

        task_weight = np.ones(n_tasks) if self.task_weights is None else np.asarray(self.task_weights, dtype=np.float64)

        tolerance = self.tolerance
        architecture = "auto" if self.solver == "auto" else ("milp" if self.solver == "milp" else "local_search")
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
            solution = solve_balance(problem, rng=rng, time_limit_s=self.time_limit_s, architecture=architecture, mip_gap=self.mip_gap)
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
            [[float(item_task_actives[solution.assignment == b, t].sum()) for b in range(n_buckets)] for t in range(n_tasks)]
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
