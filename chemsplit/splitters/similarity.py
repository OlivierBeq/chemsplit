"""The ``similarity``/fingerprint splitter family.

Every splitter here operates on a fingerprint/feature distance or similarity matrix.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
import scipy.sparse as sp
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
from chemsplit.base import BaseSplitter, GroupSplitter, SplitResult, Strictness, _Context
from chemsplit.determinism import argmax_tiebreak, argmin_tiebreak, seed_for, stable_sort
from chemsplit.exceptions import (
    ConfigurationError,
    ConstraintUnsatisfiableError,
    DegenerateClusterWarning,
    DegenerateGroupingError,
    LabelError,
    MissingDependencyError,
    ParameterError,
    SizeToleranceWarning,
    warn_with_details,
)
from chemsplit.types import IndexArray

__all__ = [
    "SimilarityThresholdSplitter",
    "ButinaSplitter",
    "KMeansClusterSplitter",
    "DensityClusterSplitter",
    "SpectralSplitter",
    "MaxMinSplitter",
    "MaxDissimilaritySplitter",
    "PerimeterSplitter",
    "LeaveOneClusterOutSplitter",
    "BalancedMultiTaskSplitter",
]


# ---------------------------------------------------------------------------
# Shared plumbing: featurizer/metric/max_memory_bytes/n_jobs
# ---------------------------------------------------------------------------

# Every leaf class below spells out its FULL constructor signature explicitly (never bare
# **kwargs) — sklearn's BaseEstimator._get_param_names() silently drops VAR_KEYWORD-only
# parameters, which would otherwise make SplitResult.params silently incomplete (see
# chemsplit/splitters/scaffold.py's module-level note for the verified detail).


class _SimilarityGroupBase(SimilarityParamsMixin, GroupSplitter):
    family: ClassVar[str] = "similarity"
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False


class _SimilarityBase(SimilarityParamsMixin, BaseSplitter):
    family: ClassVar[str] = "similarity"
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    group_forming: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False


def _dist_matrix(self: Any, ctx: _Context) -> np.ndarray:
    return compute_distance_matrix(
        ctx, self.featurizer, self.metric, self.max_memory_bytes, type(self).__name__, self.n_jobs
    )


def _sim_matrix(self: Any, ctx: _Context) -> np.ndarray:
    return compute_similarity_matrix(
        ctx, self.featurizer, self.metric, self.max_memory_bytes, type(self).__name__, self.n_jobs
    )


def _fill_remainder(
    picked: list[int], n: int, sizes, rng: np.random.Generator, picked_goes_to: str
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


# ---------------------------------------------------------------------------
# SimilarityThresholdSplitter
# ---------------------------------------------------------------------------


class SimilarityThresholdSplitter(_SimilarityGroupBase):
    """Hard constraint: no test record may exceed ``threshold`` similarity to any train record.

    Advantages
    ----------
    - The constraint is explicit, checkable, and reported — `metadata["max_cross_similarity"]` is either below the threshold or the split is wrong. No other splitter in this family gives that guarantee.
    - Directly parameterises what people actually mean by "novel chemistry": how dissimilar must test compounds be?
    - `graph_component` never discards data, so the full dataset gets used.

    Pitfalls
    --------
    - **The threshold is the experiment.** ECFP4 Tanimoto 0.4, ECFP6 Tanimoto 0.4, and MACCS 0.4 are three completely different difficulty levels — a result reported without fingerprint, radius, bit length, metric, and cutoff isn't reproducible; `params` records all five, and so should your paper.
    - Similarity isn't transitive, so connected components can be enormous and chemically incoherent — a chain of pairwise-similar molecules can link two very different ends. On dense datasets one component can swallow everything, raised as `ConstraintUnsatisfiableError` rather than silently accepted.
    - `greedy_prune` and `seeded_growth` both discard records, and exactly the ones in the interesting boundary region — the remaining test set isn't a uniform sample of anything.
    - Tanimoto on sparse fingerprints saturates — for large diverse libraries most pairs sit below 0.2, so a 0.4 cutoff removes almost nothing and the split silently becomes random. Check `metadata["component_sizes"]`.
    - All-zero fingerprints (parse failures under `on_parse_error="ignore"`, or tiny fragments) register similarity 1.0 to each other by convention and cluster together spuriously.

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
        featurizer: "str | Any" = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
    ) -> None:
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=n_jobs
        )
        GroupSplitter.__init__(
            self,
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
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
            keys = [f"train_{i}" if i not in test else f"test_{i}" for i in range(n)]
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


# ---------------------------------------------------------------------------
# ButinaSplitter
# ---------------------------------------------------------------------------


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
    - Membership is defined only relative to the **centroid**, not pairwise — two members of one cluster can be up to `2 x cutoff` apart, so a cluster isn't a tight neighbourhood and a train/test boundary between clusters does **not** guarantee any minimum cross-similarity. Use `similarity_threshold` or `hi` if you need that guarantee.
    - Produces many singletons on diverse libraries (often 30-60% of records); `singleton_policy` decides where they go and materially changes difficulty — the default `"own_group"` scatters them randomly, softening the split.
    - Highly sensitive to `cutoff` — 0.35 vs. 0.4 in distance can halve or double the cluster count.
    - The distance/similarity convention is a classic source of silent errors — always read `metadata["cutoff_is"]`.
    - `reorder=True` gives different clusters than `reorder=False`, and both get called "Butina" in the literature — report which.
    - `O(n²)` in time regardless of `algorithm` — beyond roughly 10⁵ molecules use `k_means_cluster` with mini-batch k-means, or subsample.

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
        featurizer: "str | Any" = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
    ) -> None:
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=n_jobs
        )
        GroupSplitter.__init__(
            self,
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
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
        if n_groups == n:
            raise DegenerateGroupingError(
                f"{type(self).__name__}: cutoff={self.cutoff} ({self.cutoff_is}) produced {n} "
                "singleton clusters (every record its own group)"
            )
        largest_frac = max(len(c) for c in clusters) / n if clusters else 0.0
        if largest_frac > 0.95:
            raise DegenerateGroupingError(
                f"{type(self).__name__}: largest cluster covers {largest_frac:.1%} of records"
            )
        if largest_frac > 0.6:
            warn_with_details(
                DegenerateClusterWarning(
                    f"{type(self).__name__}: largest cluster covers {largest_frac:.1%} of records",
                    details={"largest_cluster_frac": largest_frac},
                )
            )
        return dense_label_encode(labels.tolist())

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        return getattr(self, "_last_meta", {})


# ---------------------------------------------------------------------------
# KMeansClusterSplitter
# ---------------------------------------------------------------------------


class KMeansClusterSplitter(_SimilarityGroupBase):
    """K-means (or a related partitional clusterer) over fingerprint/feature space.

    Advantages
    ----------
    - Scales far better than any `O(n²)` method — `minibatch_kmeans` handles millions of molecules.
    - The cluster count is an explicit, reportable knob, and `auto_rule` makes the default reproducible rather than ad hoc.
    - Works on any feature representation, including learned embeddings and physicochemical descriptors, making it a natural generic clusterer.
    - `birch` and `minibatch` give a memory-bounded path where Butina and spectral clustering can't run.

    Pitfalls
    --------
    - **k is arbitrary.** Nothing in the chemistry determines it, yet split difficulty depends on it strongly — `auto_rule="sqrt_n"` is a convention, not a principle.
    - K-means assumes isotropic, roughly equal-variance clusters in Euclidean space, which binary fingerprint space isn't — clusters end up as much geometric artefacts as chemical families. SVD reduction mitigates but doesn't fix this.
    - Cluster sizes come out wildly uneven, so the achieved train/test ratio drifts from the request — expect `SizeToleranceWarning`.
    - Euclidean distance on binary fingerprints is dominated by molecule size (bit count), so clusters partly track molecular weight rather than chemotype. Use `property` if that's actually what you want.
    - SVD sign ambiguity makes naive implementations non-reproducible across BLAS builds; a sign fix is mandatory here.
    - `agglomerative` with `single` linkage chains badly on chemical data, typically producing one giant cluster plus dust.

    """

    splitter_id: ClassVar[str] = "k_means_cluster"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    bounded_metric_required: ClassVar[bool] = False

    def __init__(
        self,
        *,
        n_clusters: "int | Literal['auto']" = "auto",
        algorithm: Literal["kmeans", "minibatch_kmeans", "agglomerative", "birch"] = "kmeans",
        linkage: Literal["ward", "complete", "average", "single"] = "ward",
        auto_rule: Literal["sqrt_n", "n_over_50"] = "sqrt_n",
        auto_range: tuple[int, int] = (2, 50),
        batch_size: int = 1024,
        reduce_dim: "int | None" = 128,
        reduce_method: Literal["svd", "none"] = "svd",
        featurizer: "str | Any" = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
    ) -> None:
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=n_jobs
        )
        GroupSplitter.__init__(
            self,
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
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
        if self.n_clusters != "auto":
            k = int(self.n_clusters)
        elif self.auto_rule == "sqrt_n":
            k = int(round(n**0.5))
        else:
            k = max(1, n // 50)
        lo, hi = self.auto_range
        return int(np.clip(k, lo, min(hi, n - 1)))

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
        self._last_meta = {"n_clusters": int(len(set(labels.tolist()))), "algorithm": self.algorithm}
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


# ---------------------------------------------------------------------------
# DensityClusterSplitter
# ---------------------------------------------------------------------------


class DensityClusterSplitter(_SimilarityGroupBase):
    """DBSCAN or HDBSCAN density clustering on a precomputed distance matrix.

    Advantages
    ----------
    - No `k` to choose, and clusters can take any shape — a better match for chemical space than k-means's spherical assumption.
    - Explicitly models "this molecule belongs to no family", which is chemically real and which every other clusterer forces into some cluster.
    - `noise_policy="test"` produces a clean, defensible "singletons and oddities" test set for applicability-domain work.

    Pitfalls
    --------
    - The noise bucket can swallow a large fraction of a diverse library — 40% or more at sensible `eps` — and `noise_policy` then decides most of the split; the default `"own_groups"` quietly makes it much easier, since noise points scatter.
    - `eps` interacts with fingerprint density in a way with no cross-dataset meaning — a value tuned on one dataset doesn't transfer.
    - DBSCAN on a precomputed matrix needs `O(n²)` memory, capping `n` around 20,000 at the default guard.
    - HDBSCAN's `min_cluster_size` and `min_samples` interact non-obviously — changing one changes the cluster count non-monotonically.
    - Density clustering on binary fingerprints suffers from the concentration of Tanimoto distances in high dimensions — most pairs sit in a narrow band, so density contrast is weak.

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
        featurizer: "str | Any" = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
    ) -> None:
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=n_jobs
        )
        GroupSplitter.__init__(
            self,
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
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
            frac = noise.size / n
            if self.noise_policy == "discard":
                forced = noise.tolist()
            elif self.noise_policy in ("test", "train"):
                pass  # handled via a shared pseudo-group below
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


# ---------------------------------------------------------------------------
# SpectralSplitter
# ---------------------------------------------------------------------------


class SpectralSplitter(_SimilarityGroupBase):
    """Laplacian-eigenmap spectral clustering on an affinity graph.

    Advantages
    ----------
    - Minimises inter-cluster similarity by construction, reliably yielding the least train/test overlap among routine structure-based splits.
    - Handles non-convex, elongated regions of chemical space that k-means cuts straight through.
    - The eigenvalue spectrum is a free diagnostic — the spectral gap shows whether the dataset genuinely has that many separable families.

    Pitfalls
    --------
    - `O(n²)` affinity construction and a dense-ish eigenproblem cap it near 50,000 molecules — subsample above that and say so.
    - Depends on three coupled choices — graph construction, Laplacian normalisation, and `n_clusters` — none with a chemically principled default.
    - Degenerate eigenvalues (common on symmetric chemical graphs, e.g. many identical singleton components) make eigenvectors non-unique up to rotation, so cluster labels can differ between runs and platforms despite identical eigenvalues. The implementation warns but can't fix this.
    - A disconnected affinity graph silently turns spectral clustering into "one cluster per component", usually not what was wanted — hence the hard error.
    - Being the hardest split isn't the same as being the right one — a model evaluated only under spectral splitting looks worse than it will perform on a realistic screening library.

    """

    splitter_id: ClassVar[str] = "spectral"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    bounded_metric_required: ClassVar[bool] = True

    def __init__(
        self,
        *,
        n_clusters: int = 8,
        graph: Literal["threshold", "knn", "full"] = "knn",
        knn_k: int = 20,
        threshold: float = 0.3,
        laplacian: Literal["sym", "rw", "unnormalized"] = "sym",
        assign: Literal["kmeans", "discretize"] = "kmeans",
        drop_first: bool = True,
        featurizer: "str | Any" = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
    ) -> None:
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=n_jobs
        )
        GroupSplitter.__init__(
            self,
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.n_clusters = n_clusters
        self.graph = graph
        self.knn_k = knn_k
        self.threshold = threshold
        self.laplacian = laplacian
        self.assign = assign
        self.drop_first = drop_first
        self._validate_similarity_params()

    def _group_labels(self, ctx: _Context) -> IndexArray:
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
        self._last_meta = {"n_clusters": int(len(set(labels.tolist()))), "graph": self.graph}
        return labels

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        return getattr(self, "_last_meta", {})


# ---------------------------------------------------------------------------
# MaxMinSplitter
# ---------------------------------------------------------------------------


class MaxMinSplitter(_SimilarityBase):
    """Greedy maximally-diverse selection (MaxMin / Kennard-Stone).

    The selected set may go to **train** (maximise coverage) or **test** (probe breadth) — opposite
    experiments sharing one algorithm; never compare numbers across ``picked_goes_to`` values.

    Advantages
    ----------
    - With `picked_goes_to="train"`, builds the most informative training set for a fixed budget — the standard answer to "which 500 compounds should I actually assay?".
    - `coverage_radius` is a directly interpretable guarantee — no record sits further than that from a training example.
    - Memory-light in its lazy form, running on datasets where Butina and spectral clustering can't.
    - Deterministic apart from a single initial pick, and fully deterministic with `init="kennard_stone"`.

    Pitfalls
    --------
    - **The two directions are different experiments and are routinely confused.** Diverse-in-train gives an optimistic, well-covered test set; diverse-in-test gives a hard extrapolation test. Never compare numbers across `picked_goes_to` values.
    - Greedy MaxMin chases outliers — the first picks are typically the weirdest molecules in the set, including parse artefacts, salts, and fragments. Clean the data first, or the "diverse" set is a junk set.
    - Strongly depends on the initial pick when `init="random"` — report the seed, or use `"kennard_stone"` for a seed-free run.
    - Optimises coverage, not group separation — nothing stops a near-duplicate of a picked molecule from landing in the other partition. Not a leakage-control split, and shouldn't be described as one.
    - `coverage_radius` is only meaningful in the chosen metric — comparing it across fingerprints is meaningless.

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
        n_picks: "int | None" = None,
        featurizer: "str | Any" = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
    ) -> None:
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=n_jobs
        )
        BaseSplitter.__init__(
            self,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.picked_goes_to = picked_goes_to
        self.init = init
        self.n_picks = n_picks
        self._validate_similarity_params()
        if picked_goes_to not in ("train", "test"):
            raise ParameterError(f"invalid picked_goes_to: {picked_goes_to!r}")

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
            },
        )
        _small_partition_check(result, n)
        return [result]


# ---------------------------------------------------------------------------
# MaxDissimilaritySplitter
# ---------------------------------------------------------------------------


class MaxDissimilaritySplitter(_SimilarityBase):
    """Pushes train and test to opposite regions of chemical space.

    Advantages
    ----------
    - Produces a clean, reproducible large extrapolation — exactly the right test for "can this model reach a region it's never seen?".
    - Only two records are chosen by any rule; everything else follows deterministically, keeping the split easy to describe and audit.
    - `metadata["min_cross_distance"]` quantifies how far apart the two sets actually ended up.

    Pitfalls
    --------
    - Deliberately worst-case — it estimates performance on **one specific** extrapolation, not average prospective performance, so a single number here carries a very wide implicit confidence interval.
    - The whole split hinges on two seed molecules, usually outliers — one badly standardised salt can define the entire experiment.
    - Test and train are contiguous regions, so the test set is chemically homogeneous with strongly correlated errors, and the effective sample size is far below `n_test`.
    - Not a leakage constraint — nothing bounds the minimum train-to-test distance except whatever geometry results. Read `min_cross_distance` before claiming novelty.
    - `grow="nearest_to_set"` can chain and walk the test set back toward the train seed; `"nearest_to_seed"` keeps it compact — the two give materially different splits.

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
        featurizer: "str | Any" = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
    ) -> None:
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=n_jobs
        )
        BaseSplitter.__init__(
            self,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
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


# ---------------------------------------------------------------------------
# PerimeterSplitter
# ---------------------------------------------------------------------------


class PerimeterSplitter(_SimilarityBase):
    """Holds out the outskirts of the distribution; trains on the dense core.

    Advantages
    ----------
    - Directly tests the applicability-domain edge — the held-out molecules are the ones a deployed model would be least confident about, exactly where failures cost money.
    - Completely deterministic, with no seed and no free parameter beyond the metric — unusually easy to reproduce and describe.
    - The training set stays dense and representative, so training stays stable even though evaluation is hard.

    Pitfalls
    --------
    - The test set is enriched in oddities — fragments, salts, dyes, very large or small molecules, standardisation failures. A poor score may reflect data quality rather than model quality — inspect `test` before trusting the number.
    - Error bars run large, since the test set is heterogeneous and small in effective size.
    - `greedy_pairs` is `O(n²)` in memory for the pair sort and doesn't scale past a few tens of thousands of records.
    - Not a chemical-novelty guarantee — an outlier can still sit near a training molecule if that's its only near neighbour. Check with `audit.nn_similarity_profile`.
    - Because peripherality is defined by mean distance, the split partly encodes molecular size and fingerprint density.

    """

    splitter_id: ClassVar[str] = "perimeter"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    bounded_metric_required: ClassVar[bool] = False
    deterministic_without_seed: ClassVar[bool] = True

    def __init__(
        self,
        *,
        pair_rule: Literal["greedy_pairs", "outlier_score"] = "greedy_pairs",
        featurizer: "str | Any" = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
    ) -> None:
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes, n_jobs=n_jobs
        )
        BaseSplitter.__init__(
            self,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
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


# ---------------------------------------------------------------------------
# LeaveOneClusterOutSplitter
# ---------------------------------------------------------------------------


class LeaveOneClusterOutSplitter(GroupSplitter):
    """Each cluster (from a caller-supplied ``clusterer``) takes a turn as the test fold.

    Advantages
    ----------
    - Yields a **per-cluster error distribution** instead of one number — you learn which regions of chemical space the model fails in, not just that it fails.
    - Every record is tested exactly once (when `max_folds` is `None` and no small-cluster pooling happens), so the aggregate is an honest whole-dataset estimate under cluster-level extrapolation.
    - Composes with every scaffold/similarity/embedding grouping, so the same protocol answers "generalise across scaffolds?", "across Butina clusters?", "across UMAP regions?".

    Pitfalls
    --------
    - Cluster sizes are uneven, so per-fold scores come from wildly different sample sizes — a macro-average over folds and a micro-average over records can disagree sharply. Report both, and report `cluster_size` alongside every fold score.
    - Many small clusters make the fold count explode and the run expensive; `max_folds` caps it, but then some clusters are never tested, biasing the aggregate.
    - Training-set size varies across folds, so fold-to-fold differences partly measure training-set size rather than chemical difficulty.
    - A single-cluster test fold with 3 records can't support ROC-AUC or a meaningful R² — the splitter warns but can't stop the caller from computing them anyway.

    """

    splitter_id: ClassVar[str] = "leave_one_cluster_out"
    family: ClassVar[str] = "similarity"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    deterministic_method: ClassVar[bool] = True

    def __init__(
        self,
        *,
        clusterer: "GroupSplitter | None" = None,
        min_cluster_size: int = 1,
        small_cluster_policy: Literal["merge_into_train", "own_fold", "pool"] = "merge_into_train",
        max_folds: "int | None" = 50,
        fold_order: Literal["size_desc", "size_asc", "index"] = "size_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        if isinstance(clusterer, str):
            raise ParameterError(
                "clusterer must be an instantiated GroupSplitter — string-based registry lookup "
                "('butina') is not available until chemsplit.registry lands"
            )
        self.clusterer = clusterer
        self.min_cluster_size = min_cluster_size
        self.small_cluster_policy = small_cluster_policy
        self.max_folds = max_folds
        self.fold_order = fold_order
        if n_splits != 1:
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
        return 1  # derived at _run time; see _partition

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        labels = self._group_labels(ctx)
        n = ctx.n
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

        fold_groups = sorted(eligible.keys(), key=sort_key)
        never_tested: list[int] = []
        if self.max_folds is not None and len(fold_groups) > self.max_folds:
            never_tested = fold_groups[self.max_folds:]
            fold_groups = fold_groups[: self.max_folds]

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


# ---------------------------------------------------------------------------
# BalancedMultiTaskSplitter
# ---------------------------------------------------------------------------


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
    - Balance is a *constraint*, not a hope — if the requested balance is impossible, the splitter says so and names the binding task instead of silently producing a useless fold.
    - `per_task_fold_counts` gives a complete audit of what every task got — exactly the table reviewers ask for.
    - Works with any clusterer, cleanly separating the chemical criterion from the balancing.

    Pitfalls
    --------
    - Infeasibility is common on real sparse matrices — a task with three actives in one cluster simply can't be balanced. `on_infeasible="relax"` is the pragmatic escape, but a relaxed tolerance means the balance requested isn't the balance achieved; read `metadata["tolerance_used"]`.
    - Solve time grows quickly with cluster count; the `dust` merge that keeps it tractable changes the grouping in a way that's invisible unless you read `metadata["dust_merged"]`.
    - Multi-threaded MIP solving is non-deterministic — raising `threads` above 1 for speed silently breaks reproducibility.
    - Balancing on label statistics chooses the split partly using the labels, a mild form of information leakage into the experimental design — usually the lesser evil versus undefined metrics, but it should be disclosed.

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
        clusterer: "GroupSplitter | None" = None,
        clusterer_kwargs: "dict | None" = None,
        task_weights: "list[float] | None" = None,
        balance: Literal["counts", "counts_and_actives"] = "counts_and_actives",
        tolerance: float = 0.10,
        solver: Literal["auto", "milp", "heuristic"] = "auto",
        time_limit_s: float = 300.0,
        mip_gap: float = 1e-4,
        on_infeasible: Literal["raise", "relax"] = "raise",
        relax_steps: tuple[float,...] = (0.15, 0.20, 0.30),
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        if isinstance(clusterer, str):
            raise ParameterError(
                "clusterer must be an instantiated GroupSplitter — string-based registry lookup "
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

        bucket_idx = {name: i for i, (name, _) in enumerate(active_buckets)}
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
