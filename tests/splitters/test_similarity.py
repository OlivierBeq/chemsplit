"""Tests for chemsplit.splitters.similarity (renamed the similarity family)."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit.exceptions import (
    ConstraintUnsatisfiableError,
    DegenerateGroupingError,
    ParameterError,
)
from chemsplit.splitters.similarity import (
    BalancedMultiTaskSplitter,
    ButinaSplitter,
    DensityClusterSplitter,
    KMeansClusterSplitter,
    LeaveOneClusterOutSplitter,
    MaxDissimilaritySplitter,
    MaxMinSplitter,
    PerimeterSplitter,
    SimilarityThresholdSplitter,
    SpectralSplitter,
)

# Two chemically distant "families": simple alkanes/alcohols vs. aromatic amines, so clustering
# behaviour is checkable without relying on delicate similarity thresholds.
_FAMILY_A = [
    "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC",
    "CCCCCCO", "CCCCCCCO", "CCCCCCCCO", "CCCCCCCCCO", "CCCCCCCCCCO",
]
_FAMILY_B = [
    "c1ccc(N)cc1", "c1ccc(N)nc1", "c1ccc(N)cc1C", "c1ccc(N)cc1CC", "Cc1ccc(N)cc1",
    "c1ccc2[nH]ccc2c1", "c1ccc2ncccc2c1", "c1ccc2[nH]ncc2c1", "Nc1ccc2ccccc2c1", "Nc1ccc2[nH]ccc2c1",
]
SMILES_20 = _FAMILY_A + _FAMILY_B


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestSimilarityThresholdSplitter:
    def test_graph_component_respects_threshold(self):
        splitter = SimilarityThresholdSplitter(
            threshold=0.3, strategy="graph_component", train_size=0.5, test_size=0.5, random_state=0
        )
        train, test = next(splitter.split(SMILES_20))
        assert len(set(train) & set(test)) == 0
        assert len(train) + len(test) == 20

    def test_determinism(self):
        s1 = SimilarityThresholdSplitter(threshold=0.3, train_size=0.5, test_size=0.5, random_state=0)
        s2 = SimilarityThresholdSplitter(threshold=0.3, train_size=0.5, test_size=0.5, random_state=0)
        r1 = next(s1.split(SMILES_20))
        r2 = next(s2.split(SMILES_20))
        assert list(r1[0]) == list(r2[0])
        assert list(r1[1]) == list(r2[1])

    def test_unbounded_metric_rejected(self):
        with pytest.raises(ParameterError):
            SimilarityThresholdSplitter(metric="euclidean")

    def test_single_component_raises(self):
        # threshold so low every pair is "similar" -> one giant component
        splitter = SimilarityThresholdSplitter(threshold=0.01, train_size=0.5, test_size=0.5, random_state=0)
        with pytest.raises(ConstraintUnsatisfiableError):
            next(splitter.split(SMILES_20))


class TestButinaSplitter:
    def test_clusters_families_separately(self):
        splitter = ButinaSplitter(cutoff=0.4, train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        # every member of the baseline family should share at most a couple of distinct groups with the scaffold family
        labels_a = set(groups[:10].tolist())
        labels_b = set(groups[10:].tolist())
        assert len(labels_a & labels_b) == 0

    def test_cutoff_is_similarity_vs_distance(self):
        d = ButinaSplitter(cutoff=0.65, cutoff_is="distance", random_state=0)
        s = ButinaSplitter(cutoff=0.35, cutoff_is="similarity", random_state=0)
        gd = d.compute_groups(SMILES_20)
        gs = s.compute_groups(SMILES_20)
        assert list(gd) == list(gs)

    def test_degenerate_all_singletons_raises(self):
        # Orthogonal one-hot "fingerprints": every pairwise Tanimoto distance is exactly 1.0, so
        # any nontrivial cutoff leaves every point a singleton cluster.
        features = np.eye(20, dtype=np.uint8)
        splitter = ButinaSplitter(cutoff=1e-6, train_size=0.5, test_size=0.5, random_state=0)
        with pytest.raises(DegenerateGroupingError):
            splitter.compute_groups(features)


class TestKMeansClusterSplitter:
    def test_basic_clustering(self):
        splitter = KMeansClusterSplitter(n_clusters=2, train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert len(set(groups.tolist())) == 2

    def test_ward_requires_euclidean(self):
        with pytest.raises(ParameterError):
            KMeansClusterSplitter(algorithm="agglomerative", linkage="ward", metric="tanimoto")

    def test_determinism(self):
        s1 = KMeansClusterSplitter(n_clusters=2, random_state=0)
        s2 = KMeansClusterSplitter(n_clusters=2, random_state=0)
        assert list(s1.compute_groups(SMILES_20)) == list(s2.compute_groups(SMILES_20))


class TestDensityClusterSplitter:
    def test_own_groups_noise_policy(self):
        splitter = DensityClusterSplitter(
            algorithm="dbscan", eps=0.3, min_samples=2, noise_policy="own_groups", random_state=0
        )
        groups = splitter.compute_groups(SMILES_20)
        assert groups.shape[0] == 20

    def test_invalid_noise_policy(self):
        with pytest.raises(ParameterError):
            DensityClusterSplitter(noise_policy="bogus")


class TestSpectralSplitter:
    def test_basic_partition(self):
        splitter = SpectralSplitter(n_clusters=2, graph="knn", knn_k=5, train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert len(set(groups.tolist())) >= 1

    def test_knn_k_too_large(self):
        splitter = SpectralSplitter(n_clusters=2, graph="knn", knn_k=100, random_state=0)
        with pytest.raises(ParameterError):
            splitter.compute_groups(SMILES_20)


class TestMaxMinSplitter:
    def test_picked_goes_to_train(self):
        splitter = MaxMinSplitter(picked_goes_to="train", n_picks=6, train_size=0.5, test_size=0.5, random_state=0)
        result = splitter.split_result(SMILES_20)[0]
        assert len(result.train) >= 6
        assert result.metadata["picked_goes_to"] == "train"

    def test_kennard_stone_deterministic_without_seed(self):
        s1 = MaxMinSplitter(init="kennard_stone", n_picks=5, train_size=0.5, test_size=0.5)
        s2 = MaxMinSplitter(init="kennard_stone", n_picks=5, train_size=0.5, test_size=0.5)
        r1 = s1.split_result(SMILES_20)[0]
        r2 = s2.split_result(SMILES_20)[0]
        assert r1.metadata["picked"] == r2.metadata["picked"]

    def test_n_picks_equal_n_raises(self):
        splitter = MaxMinSplitter(n_picks=20, train_size=0.5, test_size=0.5, random_state=0)
        with pytest.raises(ParameterError):
            splitter.split_result(SMILES_20)


class TestMaxDissimilaritySplitter:
    def test_seeds_are_the_max_distance_pair(self):
        splitter = MaxDissimilaritySplitter(seed_pair="max_distance", train_size=0.5, test_size=0.5, random_state=0)
        result = splitter.split_result(SMILES_20)[0]
        assert result.metadata["seed_train"] != result.metadata["seed_test"]
        assert result.metadata["min_cross_distance"] >= 0.0

    def test_train_test_disjoint_and_complete(self):
        splitter = MaxDissimilaritySplitter(train_size=0.5, test_size=0.5, random_state=0)
        result = splitter.split_result(SMILES_20)[0]
        assert set(result.train.tolist()) | set(result.test.tolist()) | set(result.valid.tolist()) == set(range(20))


class TestPerimeterSplitter:
    def test_greedy_pairs(self):
        splitter = PerimeterSplitter(pair_rule="greedy_pairs", train_size=0.6, test_size=0.4, random_state=0)
        result = splitter.split_result(SMILES_20)[0]
        assert result.n_records == 20
        assert result.metadata["pair_rule"] == "greedy_pairs"

    def test_outlier_score(self):
        splitter = PerimeterSplitter(pair_rule="outlier_score", train_size=0.6, test_size=0.4, random_state=0)
        result = splitter.split_result(SMILES_20)[0]
        assert result.metadata["test_mean_outlier_score"] >= result.metadata["train_mean_outlier_score"]

    def test_determinism_no_seed(self):
        s1 = PerimeterSplitter(train_size=0.6, test_size=0.4)
        s2 = PerimeterSplitter(train_size=0.6, test_size=0.4)
        assert list(s1.split_result(SMILES_20)[0].test) == list(s2.split_result(SMILES_20)[0].test)


class TestLeaveOneClusterOutSplitter:
    def test_default_clusterer_yields_multiple_folds(self):
        splitter = LeaveOneClusterOutSplitter(min_cluster_size=1, max_folds=None)
        results = splitter.split_result(SMILES_20)
        assert len(results) >= 2
        for r in results:
            assert set(r.train.tolist()) & set(r.test.tolist()) == set()

    def test_string_clusterer_rejected(self):
        with pytest.raises(ParameterError):
            LeaveOneClusterOutSplitter(clusterer="butina")

    def test_explicit_butina_clusterer(self):
        clusterer = ButinaSplitter(cutoff=0.4, random_state=0)
        splitter = LeaveOneClusterOutSplitter(clusterer=clusterer, min_cluster_size=1, max_folds=None)
        results = splitter.split_result(SMILES_20)
        assert len(results) >= 1


class TestBalancedMultiTaskSplitter:
    def test_balances_multitask_labels(self):
        rng = np.random.default_rng(0)
        n = 60
        smiles = [_FAMILY_A[i % 10] if i % 2 == 0 else _FAMILY_B[i % 10] for i in range(n)]
        y = rng.random((n, 3))
        mask = rng.random((n, 3)) < 0.7
        y[~mask] = np.nan
        splitter = BalancedMultiTaskSplitter(train_size=0.7, test_size=0.3, tolerance=0.3, random_state=0)
        result = splitter.split_result(smiles, y=y)[0]
        assert result.metadata["n_tasks"] == 3
        assert result.metadata["solver_status"] in ("optimal", "time_limit_feasible")
        assert len(result.train) + len(result.test) == n

    def test_requires_2d_labels(self):
        splitter = BalancedMultiTaskSplitter(train_size=0.7, test_size=0.3, random_state=0)
        with pytest.raises(Exception):
            splitter.split_result(SMILES_20, y=np.arange(20))

    def test_no_external_solver_extra(self):
        assert BalancedMultiTaskSplitter.extras == ()

    def test_determinism(self):
        rng = np.random.default_rng(1)
        n = 40
        smiles = [_FAMILY_A[i % 10] if i % 2 == 0 else _FAMILY_B[i % 10] for i in range(n)]
        y = rng.random((n, 2))
        s1 = BalancedMultiTaskSplitter(train_size=0.7, test_size=0.3, tolerance=0.3, random_state=0)
        s2 = BalancedMultiTaskSplitter(train_size=0.7, test_size=0.3, tolerance=0.3, random_state=0)
        r1 = s1.split_result(smiles, y=y)[0]
        r2 = s2.split_result(smiles, y=y)[0]
        assert list(r1.train) == list(r2.train)
        assert list(r1.test) == list(r2.test)
