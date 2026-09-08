"""The ``embedding``/projection splitter family."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import numpy as np

from chemsplit._fp_similarity import SimilarityParamsMixin, compute_distance_matrix
from chemsplit._unionfind import dense_label_encode
from chemsplit.base import GroupSplitter, Strictness, _Context
from chemsplit.determinism import argmax_tiebreak, seed_for
from chemsplit.exceptions import (
    CircularityWarning,
    DegenerateGroupingError,
    MissingDependencyError,
    ParameterError,
    warn_with_details,
)
from chemsplit.exceptions import DeterminismWarning as _DeterminismWarning
from chemsplit.types import IndexArray

__all__ = ["LatentSpaceSplitter", "ProjectionSplitter", "UMAPClusterSplitter"]


# ---------------------------------------------------------------------------
# Shared n_clusters/cluster_algorithm/auto_rule/auto_range block. Local to this module
# (UMAPClusterSplitter and ProjectionSplitter use it directly; LatentSpaceSplitter clusters in a
# caller-fixed space but still exposes the same constructor block, so it reuses this mixin too).
# ---------------------------------------------------------------------------


class _ClusterCountMixin:
    """``n_clusters``/``cluster_algorithm``/``auto_rule``/``auto_range`` + cluster-count
    resolution and the actual clustering call, shared by all three embedding-family splitters."""

    def __init__(
        self,
        *,
        n_clusters: int | Literal['auto'] = "auto",
        cluster_algorithm: Literal["kmeans", "agglomerative", "hdbscan"] = "agglomerative",
        auto_rule: Literal["sqrt_n", "n_over_50", "silhouette"] = "sqrt_n",
        auto_range: tuple[int, int] = (2, 50),
    ) -> None:
        self.n_clusters = n_clusters
        self.cluster_algorithm = cluster_algorithm
        self.auto_rule = auto_rule
        self.auto_range = auto_range

    def _validate_cluster_count_params(self) -> None:
        if self.n_clusters != "auto":
            if isinstance(self.n_clusters, bool) or not isinstance(self.n_clusters, (int, np.integer)):
                raise ParameterError(f"n_clusters must be 'auto' or an int, got {self.n_clusters!r}")
            if self.n_clusters < 2:
                raise ParameterError(f"n_clusters must be >= 2, got {self.n_clusters!r}")
        if self.cluster_algorithm not in ("kmeans", "agglomerative", "hdbscan"):
            raise ParameterError(f"invalid cluster_algorithm: {self.cluster_algorithm!r}")
        if self.auto_rule not in ("sqrt_n", "n_over_50", "silhouette"):
            raise ParameterError(f"invalid auto_rule: {self.auto_rule!r}")
        lo, hi = self.auto_range
        if not (isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer)) and 1 <= lo <= hi):
            raise ParameterError(f"auto_range must be (lo, hi) with 1 <= lo <= hi, got {self.auto_range!r}")

    def _resolve_n_clusters(self, n: int, Z: np.ndarray, rng: np.random.Generator) -> int:
        lo, hi = self.auto_range
        hi = min(hi, n - 1) if n > 1 else lo
        if self.n_clusters != "auto":
            k = int(self.n_clusters)
            return max(2, min(k, n - 1))
        if self.auto_rule == "sqrt_n":
            k = int(round(float(np.sqrt(n))))
        elif self.auto_rule == "n_over_50":
            k = int(round(n / 50))
        else:  # "silhouette": maximise mean silhouette over auto_range on a subsample
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            sub_n = min(n, 5000)
            if sub_n < n:
                sub_idx = seed_for(self._silhouette_bundle, "kmeans.silhouette_subsample", 0).choice(
                    n, size=sub_n, replace=False
                )
                Zs = Z[sub_idx]
            else:
                Zs = Z
            best_k, best_score = lo, -np.inf
            for k in range(lo, hi + 1):
                if k >= Zs.shape[0]:
                    break
                labels = KMeans(n_clusters=k, n_init=10, algorithm="lloyd", max_iter=300, tol=1e-4,
                                 random_state=0).fit_predict(Zs)
                if len(set(labels.tolist())) < 2:
                    continue
                score = silhouette_score(Zs, labels)
                if score > best_score:
                    best_score, best_k = score, k
            return best_k
        return max(lo, min(k, hi))

    def _cluster(self, Z: np.ndarray, k: int, seed: int) -> np.ndarray:
        from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans

        if self.cluster_algorithm == "kmeans":
            labels = KMeans(
                n_clusters=k, n_init=10, algorithm="lloyd", max_iter=300, tol=1e-4,
                random_state=seed,
            ).fit_predict(Z)
        elif self.cluster_algorithm == "agglomerative":
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean").fit_predict(Z)
        else:  # hdbscan
            labels = HDBSCAN(min_cluster_size=max(2, Z.shape[0] // (2 * k))).fit_predict(Z)
            noise = labels == -1
            if noise.any():
                next_id = int(labels.max()) + 1 if (labels != -1).any() else 0
                labels = labels.copy()
                for i in np.nonzero(noise)[0]:
                    labels[i] = next_id
                    next_id += 1
        return np.asarray(dense_label_encode(labels.tolist()), dtype=np.int64)


def _fix_sign(Z: np.ndarray) -> np.ndarray:
    """Deterministic sign fix for SVD/eigendecomposition components:
    per column, negate if the largest-|value| entry is negative."""
    Z = np.asarray(Z, dtype=np.float64).copy()
    for c in range(Z.shape[1]):
        col = Z[:, c]
        abs_col = np.abs(col)
        idx = argmax_tiebreak(lambda k: abs_col[k], range(len(col)))
        if col[idx] < 0:
            Z[:, c] = -col
    return Z


# ---------------------------------------------------------------------------
# UMAPClusterSplitter
# ---------------------------------------------------------------------------


class UMAPClusterSplitter(_ClusterCountMixin, SimilarityParamsMixin, GroupSplitter):
    """UMAP embedding followed by clustering (``umap_cluster``).

    Embeds the featurized records into a low-dimensional UMAP space, then clusters that embedding
    and treats each cluster as an atomic group.

    :param n_components: Embedding dimension. 2 is conventional and lossy; 5-10 preserves more
        structure and is recommended when the embedding is used for splitting rather than
        visualisation. Defaults to 2.
    :param n_neighbors: UMAP local-neighbourhood size. Governs the local/global trade-off and
        materially changes the resulting split. Defaults to 15.
    :param min_dist: Minimum embedded distance. Defaults to 0.1.
    :param umap_metric: UMAP's own metric on the input features. ``"jaccard"`` is correct for
        binary fingerprints (equals Tanimoto distance on binary vectors); ``"euclidean"`` on raw
        bit vectors is a common and documented mistake. Defaults to ``"jaccard"``.
    :param n_epochs: ``None`` uses UMAP's own default (500 for n<10000, else 200). Pinning it
        makes runs comparable across dataset sizes. Defaults to ``None``.
    :ivar splitter_id: ``"umap_cluster"``.

    Advantages
    ----------
    - Typically produces the widest train/test gap among routinely used splits — a strong stress test.
    - The embedding is directly plottable, so train and test regions are visible on one figure — something no fingerprint-space split offers.
    - Non-linear, so it separates chemical families that PCA blends together.
    - Scales to far larger datasets than spectral clustering.

    Pitfalls
    --------
    - **UMAP doesn't preserve global distances.** Inter-cluster distances in the embedding aren't meaningful, so "these clusters are far apart in UMAP" isn't evidence of dissimilarity. Verify with `audit.nn_similarity_profile` in fingerprint space.
    - The split shifts substantially with `n_neighbors`, `min_dist`, `n_components`, `densmap`, and the seed — all recorded in `params` and all must be reported.
    - Reproducibility depends on library versions, not just the seed: a `numba` or `pynndescent` upgrade can change the embedding and therefore the split, which is why `metadata` records the version triple — its golden test uses a size/histogram tolerance, not an exact match.
    - `n_jobs > 1` silently breaks UMAP reproducibility, so this implementation refuses it at a real speed cost.
    - Choosing the cluster count is guesswork, as in `k_means_cluster`, compounded here by the embedding's own hyperparameters.
    - `n_components=2` suits pictures, not splitting — it discards a lot of structure. Prefer 5-10 when the embedding is meant to define groups.

    Notes
    -----
    Determinism ``purpose`` strings: ``"umap.fit"``, ``"kmeans.fit"``, ``"group.assign"``.
    """

    splitter_id: ClassVar[str] = "umap_cluster"
    family: ClassVar[str] = "embedding"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    group_forming: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    extras: ClassVar[tuple[str,...]] = ("umap",)
    deterministic_without_seed: ClassVar[bool] = False
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        umap_metric: str = "jaccard",
        densmap: bool = False,
        n_epochs: int | None = None,
        n_clusters: int | Literal['auto'] = "auto",
        cluster_algorithm: Literal["kmeans", "agglomerative", "hdbscan"] = "agglomerative",
        auto_rule: Literal["sqrt_n", "n_over_50", "silhouette"] = "sqrt_n",
        auto_range: tuple[int, int] = (2, 50),
        featurizer: Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        **base: Any,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.umap_metric = umap_metric
        self.densmap = densmap
        self.n_epochs = n_epochs
        _ClusterCountMixin.__init__(
            self, n_clusters=n_clusters, cluster_algorithm=cluster_algorithm,
            auto_rule=auto_rule, auto_range=auto_range,
        )
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes,
            n_jobs=n_jobs,
        )
        GroupSplitter.__init__(
            self, size_tolerance=size_tolerance, group_assignment=group_assignment,
            n_jobs=n_jobs, **base,
        )
        if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ParameterError(f"n_components must be >= 1, got {n_components!r}")
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 2:
            raise ParameterError(f"n_neighbors must be >= 2, got {n_neighbors!r}")
        if not (0.0 <= min_dist < 1.0):
            raise ParameterError(f"min_dist must satisfy 0 <= min_dist < 1, got {min_dist!r}")
        self._validate_cluster_count_params()

    def _group_labels(self, ctx: _Context) -> IndexArray:
        try:
            import umap
        except ImportError as exc:
            raise MissingDependencyError(type(self).__name__, "umap") from exc

        if ctx.n <= self.n_neighbors:
            raise ParameterError(
                f"UMAPClusterSplitter requires n > n_neighbors ({self.n_neighbors}), got n={ctx.n}"
            )

        from chemsplit.featurizers import get_featurizer

        featurizer = get_featurizer(self.featurizer)
        F = ctx.get_features(featurizer)

        n_jobs = self.n_jobs
        if n_jobs is not None and n_jobs > 1:
            warn_with_details(
                _DeterminismWarning(
                    "UMAPClusterSplitter: n_jobs>1 is non-reproducible in umap-learn even with "
                    "random_state set; overriding to n_jobs=1.",
                    details={"requested_n_jobs": n_jobs},
                )
            )

        seed = int(seed_for(ctx.rng_seeds, "umap.fit", 0).integers(0, 2**31 - 1))
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.umap_metric,
            densmap=self.densmap,
            n_epochs=self.n_epochs,
            random_state=seed,
            n_jobs=1,
            transform_seed=seed,
            verbose=False,
        )
        F_dense = F.toarray() if hasattr(F, "toarray") else np.asarray(F)
        Z = np.asarray(reducer.fit_transform(F_dense), dtype=np.float64)

        self._silhouette_bundle = ctx.rng_seeds
        k = self._resolve_n_clusters(ctx.n, Z, seed_for(ctx.rng_seeds, "kmeans.fit", 0))
        labels = self._cluster(Z, k, int(seed_for(ctx.rng_seeds, "kmeans.fit", 0).integers(0, 2**31 - 1)))

        self._last_embedding_shape = list(Z.shape)
        self._last_labels = labels
        return labels

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        import numba
        import pynndescent
        import umap as umap_module

        sizes = np.bincount(labels).tolist()
        return {
            "n_clusters": int(labels.max()) + 1 if labels.size else 0,
            "cluster_sizes": sizes,
            "embedding_shape": getattr(self, "_last_embedding_shape", None),
            "umap_versions": {
                "umap": getattr(umap_module, "__version__", "unknown"),
                "numba": numba.__version__,
                "pynndescent": pynndescent.__version__,
            },
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "nondeterministic_method": True,  # numba JIT, not bit-exact cross-platform
        }


# ---------------------------------------------------------------------------
# ProjectionSplitter
# ---------------------------------------------------------------------------


class ProjectionSplitter(_ClusterCountMixin, SimilarityParamsMixin, GroupSplitter):
    """Linear or manifold projection followed by clustering, an axis cut, or a grid
    (``projection``).

    :param method: Projection method. Defaults to ``"pca"``.
    :param n_components: Projected dimension. Defaults to 2.
    :param mode: ``"cluster"`` clusters the projection as in ``umap_cluster``. ``"axis_cut"``
        sorts by component ``axis`` and cuts into contiguous blocks sized by the size targets.
        ``"grid"`` bins each of the first ``n_components`` axes into ``grid_bins``
        equal-frequency bins; the group is the cell tuple. Defaults to ``"cluster"``.
    :param axis: Component index used by ``mode="axis_cut"``. Defaults to 0.
    :param grid_bins: Bins per axis for ``mode="grid"``. Defaults to 4.
    :param tsne_perplexity: t-SNE perplexity (only used when ``method="tsne"``). Defaults to 30.0.
    :param kernel: Kernel for ``method="kernel_pca"``. Defaults to ``"rbf"``.

    Advantages
    ----------
    - PCA is linear, cheap, and interpretable — the loading vector shows *which* descriptors define the split direction, unlike any non-linear method.
    - `mode="axis_cut"` gives a clean, reportable one-dimensional extrapolation along the dataset's main axis of variation.
    - `explained_variance_ratio` states plainly how much of the data the projection actually captured — an honesty check UMAP and t-SNE lack.

    Pitfalls
    --------
    - **t-SNE inter-cluster distances are meaningless.** Treating t-SNE geometry as a split criterion attributes chemical significance to an artefact of its cost function; it's common practice, but the docstring must say it's unsound as a distance.
    - PCA on binary fingerprints puts most variance in bit frequency, which correlates with molecule size, so the first component often just tracks "how big is the molecule".
    - `mode="grid"` produces exponentially many groups, most empty or singletons.
    - Two components typically explain only a small fraction of fingerprint variance — check `explained_variance_ratio` before trusting the geometry.
    - t-SNE and MDS aren't byte-reproducible across BLAS builds even when seeded, so their golden tests use a tolerance, not an exact match.

    Notes
    -----
    ``metadata["nondeterministic_method"]`` is ``True`` for ``method`` in ``{"tsne", "mds"}``
    (irreducible BLAS-level non-determinism). SVD/PCA/KernelPCA components are deterministically
    sign-fixed.
    """

    splitter_id: ClassVar[str] = "projection"
    family: ClassVar[str] = "embedding"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False

    def __init__(
        self,
        *,
        method: Literal["pca", "svd", "tsne", "mds", "kernel_pca"] = "pca",
        n_components: int = 2,
        mode: Literal["cluster", "axis_cut", "grid"] = "cluster",
        axis: int = 0,
        grid_bins: int = 4,
        tsne_perplexity: float = 30.0,
        kernel: Literal["linear", "rbf", "cosine"] = "rbf",
        n_clusters: int | Literal['auto'] = "auto",
        cluster_algorithm: Literal["kmeans", "agglomerative", "hdbscan"] = "agglomerative",
        auto_rule: Literal["sqrt_n", "n_over_50", "silhouette"] = "sqrt_n",
        auto_range: tuple[int, int] = (2, 50),
        featurizer: Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        **base: Any,
    ) -> None:
        self.method = method
        self.n_components = n_components
        self.mode = mode
        self.axis = axis
        self.grid_bins = grid_bins
        self.tsne_perplexity = tsne_perplexity
        self.kernel = kernel
        _ClusterCountMixin.__init__(
            self, n_clusters=n_clusters, cluster_algorithm=cluster_algorithm,
            auto_rule=auto_rule, auto_range=auto_range,
        )
        SimilarityParamsMixin.__init__(
            self, featurizer=featurizer, metric=metric, max_memory_bytes=max_memory_bytes,
            n_jobs=n_jobs,
        )
        GroupSplitter.__init__(
            self, size_tolerance=size_tolerance, group_assignment=group_assignment,
            n_jobs=n_jobs, **base,
        )
        if method not in ("pca", "svd", "tsne", "mds", "kernel_pca"):
            raise ParameterError(f"invalid method: {method!r}")
        if mode not in ("cluster", "axis_cut", "grid"):
            raise ParameterError(f"invalid mode: {mode!r}")
        if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ParameterError(f"n_components must be >= 1, got {n_components!r}")
        if isinstance(grid_bins, bool) or not isinstance(grid_bins, (int, np.integer)) or grid_bins < 2:
            raise ParameterError(f"grid_bins must be >= 2, got {grid_bins!r}")
        self._validate_cluster_count_params()
        self._nondeterministic_method = method in ("tsne", "mds")

    def _group_labels(self, ctx: _Context) -> IndexArray:
        from chemsplit.featurizers import get_featurizer

        featurizer = get_featurizer(self.featurizer)
        F = ctx.get_features(featurizer)
        n = ctx.n
        if self.n_components >= min(n, F.shape[1] if hasattr(F, "shape") else n):
            raise ParameterError(
                f"n_components ({self.n_components}) must be < min(n, n_features)"
            )
        if self.method == "tsne" and self.tsne_perplexity >= n / 3:
            raise ParameterError(
                f"tsne_perplexity ({self.tsne_perplexity}) must be < n/3 ({n / 3:.1f})"
            )

        seed = int(seed_for(ctx.rng_seeds, "projection.fit", 0).integers(0, 2**31 - 1))
        explained_variance_ratio: list[float] | None = None
        kl_divergence: float | None = None

        if self.method in ("pca", "svd"):
            from sklearn.decomposition import TruncatedSVD

            svd = TruncatedSVD(n_components=self.n_components, random_state=seed, algorithm="randomized", n_iter=7)
            F_in = F
            if self.method == "pca" and hasattr(F, "toarray"):
                F_in = F.toarray()
            Z = svd.fit_transform(F_in)
            Z = _fix_sign(Z)
            explained_variance_ratio = svd.explained_variance_ratio_.tolist()
        elif self.method == "tsne":
            from sklearn.manifold import TSNE

            F_dense = F.toarray() if hasattr(F, "toarray") else np.asarray(F)
            tsne = TSNE(
                n_components=self.n_components, perplexity=self.tsne_perplexity, init="pca",
                random_state=seed,
                method="barnes_hut" if self.n_components < 4 else "exact",
            )
            Z = tsne.fit_transform(F_dense)
            kl_divergence = float(tsne.kl_divergence_)
        elif self.method == "mds":
            from sklearn.manifold import MDS

            D = compute_distance_matrix(
                ctx, featurizer, self.metric, self.max_memory_bytes, type(self).__name__, self.n_jobs,
            )
            mds = MDS(
                n_components=self.n_components, dissimilarity="precomputed", random_state=seed,
                normalized_stress="auto",
            )
            Z = mds.fit_transform(D)
        else:  # kernel_pca
            from sklearn.decomposition import KernelPCA

            F_dense = F.toarray() if hasattr(F, "toarray") else np.asarray(F)
            kpca = KernelPCA(n_components=self.n_components, kernel=self.kernel, random_state=seed)
            Z = kpca.fit_transform(F_dense)
            Z = _fix_sign(Z)

        Z = np.asarray(Z, dtype=np.float64)

        if self.mode == "cluster":
            self._silhouette_bundle = ctx.rng_seeds
            k = self._resolve_n_clusters(n, Z, seed_for(ctx.rng_seeds, "kmeans.fit", 0))
            labels = self._cluster(Z, k, int(seed_for(ctx.rng_seeds, "kmeans.fit", 0).integers(0, 2**31 - 1)))
        elif self.mode == "axis_cut":
            order = np.argsort(Z[:, self.axis], kind="stable")
            n_train, n_valid, n_test = ctx.sizes.n_train, ctx.sizes.n_valid, ctx.sizes.n_test
            cuts = np.cumsum([n_train, n_valid, n_test])
            block = np.zeros(n, dtype=np.int64)
            block[order[: cuts[0]]] = 0
            block[order[cuts[0]:cuts[1]]] = 1
            block[order[cuts[1]:cuts[2]]] = 2
            if cuts[2] < n:
                block[order[cuts[2]:]] = 3
            labels = np.asarray(dense_label_encode(block.tolist()), dtype=np.int64)
        else:  # grid
            bins_per_axis = []
            for c in range(self.n_components):
                edges = np.quantile(Z[:, c], np.linspace(0, 1, self.grid_bins + 1))
                edges = np.unique(edges)
                bins_per_axis.append(np.clip(np.digitize(Z[:, c], edges[1:-1]), 0, len(edges) - 2))
            cell = list(zip(*bins_per_axis, strict=True))
            labels = np.asarray(dense_label_encode(cell), dtype=np.int64)
            if len(set(cell)) > n:
                raise DegenerateGroupingError(
                    f"ProjectionSplitter(mode='grid'): {len(set(cell))} grid cells for {n} "
                    "records -- reduce grid_bins or n_components"
                )

        self._last_explained_variance_ratio = explained_variance_ratio
        self._last_kl_divergence = kl_divergence
        return labels

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        sizes = np.bincount(labels).tolist() if labels.size else []
        return {
            "method": self.method,
            "mode": self.mode,
            "n_components": self.n_components,
            "explained_variance_ratio": getattr(self, "_last_explained_variance_ratio", None),
            "kl_divergence": getattr(self, "_last_kl_divergence", None),
            "n_clusters": len(sizes),
            "nondeterministic_method": self._nondeterministic_method,
        }


# ---------------------------------------------------------------------------
# LatentSpaceSplitter
# ---------------------------------------------------------------------------


class LatentSpaceSplitter(_ClusterCountMixin, GroupSplitter):
    """Clusters in an embedding supplied by the caller (``latent_space``) -- a ChemBERTa/GNN/VAE
    representation, not one chemsplit computes itself.

    :param embedding: An ``(n, d)`` array, or a callable applied to ``X``. If ``None``, ``X``
        itself must already be a feature matrix (``accepts=("features",)``).
    :param normalize: Row normalisation applied to the embedding before clustering. Defaults to
        ``"l2"``.
    :param embedding_metric: Distance used by the clusterer. Unbounded metrics are allowed here.
        Defaults to ``"cosine"``.
    :param independence_declared: The caller asserts that the encoder producing the embedding is
        not the model being evaluated. If ``False``, :class:`~chemsplit.exceptions.CircularityWarning`
        is emitted at every call -- the flag exists to force the user to think about this, not to
        verify it. Defaults to ``False``.

    Advantages
    ----------
    - Splits in the space the downstream model actually perceives — arguably the most relevant notion of "similar" for that model.
    - Works for modalities without fingerprints: reaction embeddings, 3-D conformer embeddings, multimodal representations.
    - `embedding_hash` ties the split to a specific representation, making it traceable.

    Pitfalls
    --------
    - **Circularity.** If the encoder defining the split is also the model under evaluation, or was pretrained on the same data, the split is inadvertently tuned to be easy or hard for that model, and cross-model comparison becomes meaningless. Use an independent encoder — the `independence_declared` flag forces you to think about this, but doesn't verify it.
    - Pretrained encoders usually saw enormous public corpora overlapping your test set, so novelty relative to your training set isn't novelty relative to the encoder's pretraining set.
    - Latent geometry drifts with checkpoint, tokenizer, and pooling choices, so a split reproduces only against a pinned encoder artefact.
    - Cosine distance in a latent space has no chemical units, so fingerprint-space cutoff intuitions don't transfer.

    """

    splitter_id: ClassVar[str] = "latent_space"
    family: ClassVar[str] = "embedding"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("features",)
    extras: ClassVar[tuple[str,...]] = ()

    def __init__(
        self,
        *,
        embedding: np.ndarray | Callable[[Any], np.ndarray] | None = None,
        normalize: Literal["none", "l2", "zscore"] = "l2",
        embedding_metric: str = "cosine",
        independence_declared: bool = False,
        n_clusters: int | Literal['auto'] = "auto",
        cluster_algorithm: Literal["kmeans", "agglomerative", "hdbscan"] = "agglomerative",
        auto_rule: Literal["sqrt_n", "n_over_50", "silhouette"] = "sqrt_n",
        auto_range: tuple[int, int] = (2, 50),
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        **base: Any,
    ) -> None:
        self.embedding = embedding
        self.normalize = normalize
        self.embedding_metric = embedding_metric
        self.independence_declared = independence_declared
        _ClusterCountMixin.__init__(
            self, n_clusters=n_clusters, cluster_algorithm=cluster_algorithm,
            auto_rule=auto_rule, auto_range=auto_range,
        )
        GroupSplitter.__init__(self, size_tolerance=size_tolerance, group_assignment=group_assignment, **base)
        if normalize not in ("none", "l2", "zscore"):
            raise ParameterError(f"invalid normalize: {normalize!r}")
        self._validate_cluster_count_params()

    def _group_labels(self, ctx: _Context) -> IndexArray:
        if not self.independence_declared:
            warn_with_details(
                CircularityWarning(
                    "LatentSpaceSplitter: independence_declared=False -- if the encoder that "
                    "produced this embedding is also (or shares data with) the model under "
                    "evaluation, this split's difficulty is not meaningful for that model.",
                    details={"independence_declared": False},
                )
            )

        if self.embedding is None:
            Z = np.asarray(ctx.get_features(), dtype=np.float64)
        elif callable(self.embedding):
            # accepts=("features",) guarantees X arrived as ctx.raw_features (no mols to fall
            # back to -- LatentSpaceSplitter never featurizes molecules itself).
            Z = np.asarray(self.embedding(ctx.raw_features), dtype=np.float64)
        else:
            Z = np.asarray(self.embedding, dtype=np.float64)

        if Z.shape[0] != ctx.n:
            from chemsplit.exceptions import InputKindError

            raise InputKindError(f"embedding has {Z.shape[0]} rows, expected n={ctx.n}")
        if not np.all(np.isfinite(Z)):
            from chemsplit.exceptions import InputKindError

            raise InputKindError("LatentSpaceSplitter: embedding contains non-finite values")

        if self.normalize == "l2":
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            Z = Z / norms
        elif self.normalize == "zscore":
            mean = Z.mean(axis=0, keepdims=True)
            std = Z.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            Z = (Z - mean) / std

        self._silhouette_bundle = ctx.rng_seeds
        k = self._resolve_n_clusters(ctx.n, Z, seed_for(ctx.rng_seeds, "kmeans.fit", 0))
        labels = self._cluster(Z, k, int(seed_for(ctx.rng_seeds, "kmeans.fit", 0).integers(0, 2**31 - 1)))

        self._last_embedding_hash = hashlib.blake2b(
            np.ascontiguousarray(Z, dtype=np.float32).tobytes(), digest_size=8
        ).hexdigest()
        self._last_embedding_dim = int(Z.shape[1])
        return labels

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        sizes = np.bincount(labels).tolist() if labels.size else []
        return {
            "embedding_hash": getattr(self, "_last_embedding_hash", ""),
            "embedding_dim": getattr(self, "_last_embedding_dim", 0),
            "normalize": self.normalize,
            "independence_declared": self.independence_declared,
            "n_clusters": len(sizes),
            "cluster_sizes": sizes,
        }
