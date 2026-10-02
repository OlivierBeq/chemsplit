"""Tests for chemsplit.splitters.similarity (renamed the similarity family)."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit.exceptions import (
    ConstraintUnsatisfiableError,
    DegenerateClusterWarning,
    DegenerateGroupingError,
    LabelError,
    ParameterError,
)
from chemsplit.splitters.similarity import (
    BalancedMultiTaskSplitter,
    ButinaSplitter,
    DensityClusterSplitter,
    DOptimalSplitter,
    DuplexSplitter,
    KMeansClusterSplitter,
    LeaveOneClusterOutSplitter,
    MaxDissimilaritySplitter,
    MaxMinSplitter,
    MinimalTestSetDissimilaritySplitter,
    OptiSimSplitter,
    PerimeterSplitter,
    SimilarityThresholdSplitter,
    SpectralSplitter,
    SphereExclusionSplitter,
    SPXYSplitter,
    SupportPointsSplitter,
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


class TestSphereExclusionSplitter:
    def test_separates_families(self):
        splitter = SphereExclusionSplitter(radius=0.6, train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert set(groups[:10].tolist()).isdisjoint(groups[10:].tolist())

    def test_members_within_radius_of_representative(self):
        from rdkit import Chem

        from chemsplit.featurizers import get_featurizer
        from chemsplit.metrics import pairwise_distances

        splitter = SphereExclusionSplitter(radius=0.6, train_size=0.5, test_size=0.5, random_state=3)
        result = splitter.split_result(SMILES_20)[0]
        mols = [Chem.MolFromSmiles(smi) for smi in SMILES_20]
        D = pairwise_distances(get_featurizer("ecfp4").transform(mols), metric="tanimoto")
        for rep in result.metadata["representatives"]:
            members = np.flatnonzero(result.groups == result.groups[rep])
            assert np.all(D[rep, members] <= 0.6 + 1e-6)
        # a representative is never within the radius of an earlier representative
        reps = result.metadata["representatives"]
        for k, rep in enumerate(reps):
            assert all(D[rep, prev] > 0.6 for prev in reps[:k])

    def test_index_order_is_seed_free(self):
        g1 = SphereExclusionSplitter(radius=0.6, order="index", random_state=0).compute_groups(SMILES_20)
        g2 = SphereExclusionSplitter(radius=0.6, order="index", random_state=1).compute_groups(SMILES_20)
        assert g1.tolist() == g2.tolist()

    def test_random_order_is_seeded(self):
        g1 = SphereExclusionSplitter(radius=0.6, random_state=5).compute_groups(SMILES_20)
        g2 = SphereExclusionSplitter(radius=0.6, random_state=5).compute_groups(SMILES_20)
        assert g1.tolist() == g2.tolist()

    def test_radius_is_similarity_matches_distance(self):
        gd = SphereExclusionSplitter(radius=0.6, radius_is="distance", random_state=0).compute_groups(SMILES_20)
        gs = SphereExclusionSplitter(radius=0.4, radius_is="similarity", random_state=0).compute_groups(SMILES_20)
        assert gd.tolist() == gs.tolist()

    def test_fraction_of_range_on_unscaled_euclidean_features(self):
        rng = _rng(4)
        X = np.r_[rng.normal(0.0, 1.0, (15, 3)), rng.normal(20.0, 1.0, (15, 3))] * [1.0, 100.0, 0.01]
        splitter = SphereExclusionSplitter(
            metric="euclidean", radius=0.2, radius_is="fraction_of_range",
            train_size=0.5, test_size=0.5, random_state=0,
        )
        groups = splitter.compute_groups(X)
        assert set(groups[:15].tolist()).isdisjoint(groups[15:].tolist())

    def test_all_singletons_raises(self):
        features = np.eye(20, dtype=np.uint8)
        with pytest.raises(DegenerateGroupingError):
            SphereExclusionSplitter(radius=1e-6, random_state=0).compute_groups(features)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"radius": 0.0},
            {"radius": 1.0},
            {"radius": 1.5, "radius_is": "fraction_of_range", "metric": "euclidean"},
            {"radius": -1.0, "metric": "euclidean"},
            {"radius": 0.5, "radius_is": "similarity", "metric": "euclidean"},
            {"radius_is": "bogus"},
            {"order": "bogus"},
        ],
    )
    def test_invalid_params_raise(self, kwargs):
        with pytest.raises(ParameterError):
            SphereExclusionSplitter(**kwargs)

    def test_unbounded_distance_radius_may_exceed_one(self):
        SphereExclusionSplitter(radius=5.0, metric="euclidean")


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


class TestSpectralLandmarkGraph:
    @staticmethod
    def _fixture():
        from chemsplit.datasets import make_two_clusters

        fx = make_two_clusters(n=120, seed=0)
        return fx.smiles, np.asarray(fx.groups_true)

    def test_optisim_landmarks_recover_families_without_full_matrix(self, monkeypatch):
        import chemsplit.splitters.similarity as sim

        def no_full_matrix(*args, **kwargs):
            raise AssertionError("graph='landmark' must not build the n x n matrix")

        monkeypatch.setattr(sim, "_sim_matrix", no_full_matrix)
        monkeypatch.setattr(sim, "_dist_matrix", no_full_matrix)
        smiles, truth = self._fixture()
        splitter = SpectralSplitter(graph="landmark", n_clusters=2, train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(smiles)
        assert all(len(set(truth[groups == g].tolist())) == 1 for g in set(groups.tolist()))

    def test_random_landmarks_run_and_report(self):
        smiles, _ = self._fixture()
        splitter = SpectralSplitter(
            graph="landmark", landmark_selection="random", n_landmarks=20, n_clusters=3,
            train_size=0.5, test_size=0.5, random_state=0,
        )
        result = splitter.split_result(smiles)[0]
        assert result.metadata["graph"] == "landmark"
        assert result.metadata["n_landmarks"] == 20

    def test_landmark_count_must_cover_clusters(self):
        smiles, _ = self._fixture()
        with pytest.raises(ParameterError):
            SpectralSplitter(graph="landmark", n_landmarks=2, n_clusters=3, random_state=0).split_result(smiles)

    @pytest.mark.parametrize(
        "kwargs",
        [{"graph": "bogus"}, {"landmark_selection": "kmeans"}, {"n_landmarks": 1}, {"landmark_neighbors": 0}],
    )
    def test_invalid_params(self, kwargs):
        with pytest.raises(ParameterError):
            SpectralSplitter(**kwargs)


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

    def test_swap_fraction_zero_matches_plain_kennard_stone(self):
        plain = MaxMinSplitter(init="kennard_stone", train_size=0.5, test_size=0.5, random_state=0).split_result(SMILES_20)[0]
        zero = MaxMinSplitter(init="kennard_stone", swap_fraction=0.0, train_size=0.5, test_size=0.5, random_state=0).split_result(SMILES_20)[0]
        assert plain.train.tolist() == zero.train.tolist()
        assert "swapped_in" not in zero.metadata

    def test_mlm_swaps_rounded_fraction_each_way(self):
        X = _rng(10).normal(size=(50, 3))
        ks = MaxMinSplitter(init="kennard_stone", metric="euclidean", train_size=0.8, test_size=0.2, random_state=0).split_result(X)[0]
        mlm = MaxMinSplitter(
            init="kennard_stone", metric="euclidean", swap_fraction=0.1, train_size=0.8, test_size=0.2, random_state=0
        ).split_result(X)[0]
        out, inn = mlm.metadata["swapped_out"], mlm.metadata["swapped_in"]
        assert len(out) == len(inn) == 4  # floor_round(0.1 * 40)
        assert set(out) <= set(ks.train.tolist()) and set(out).isdisjoint(mlm.train.tolist())
        assert set(inn) <= set(ks.test.tolist()) and set(inn) <= set(mlm.train.tolist())
        assert mlm.train.size == 40
        again = MaxMinSplitter(
            init="kennard_stone", metric="euclidean", swap_fraction=0.1, train_size=0.8, test_size=0.2, random_state=0
        ).split_result(X)[0]
        assert again.train.tolist() == mlm.train.tolist()

    def test_swap_capped_by_unpicked_records(self):
        X = _rng(11).normal(size=(20, 2))
        result = MaxMinSplitter(
            init="kennard_stone", metric="euclidean", swap_fraction=0.5, train_size=0.9, test_size=0.1, random_state=0
        ).split_result(X)[0]
        assert len(result.metadata["swapped_in"]) == 2

    @pytest.mark.parametrize("swap_fraction", [-0.1, 0.6, True, "0.1"])
    def test_invalid_swap_fraction(self, swap_fraction):
        with pytest.raises(ParameterError):
            MaxMinSplitter(swap_fraction=swap_fraction)


class TestSPXYSplitter:
    # 1-D points 0, 1, 3, 4 with record 1 the only label outlier. Scaled joint distances:
    # (0,1)=1.25, (0,2)=0.75, (0,3)=1.0, (1,2)=1.5, (1,3)=1.75, (2,3)=0.25, so the first
    # Kennard-Stone pair is (1, 3); on X alone it would be (0, 3).
    X_1D = np.array([[0.0], [1.0], [3.0], [4.0]])
    Y_1D = np.array([0.0, 3.0, 0.0, 0.0])

    def test_hand_computed_trace(self):
        splitter = SPXYSplitter(metric="euclidean", train_size=2, test_size=2)
        result = splitter.split_result(self.X_1D, self.Y_1D)[0]
        assert result.metadata["picked"] == [1, 3]
        assert result.train.tolist() == [1, 3]
        assert result.test.tolist() == [0, 2]
        ks = MaxMinSplitter(init="kennard_stone", metric="euclidean", train_size=2, test_size=2)
        assert ks.split_result(self.X_1D)[0].train.tolist() == [0, 3]

    def test_constant_y_reduces_to_kennard_stone(self):
        X = _rng(1).normal(size=(30, 4))
        with pytest.warns(DegenerateClusterWarning, match="label"):
            spxy = SPXYSplitter(metric="euclidean", train_size=0.7, test_size=0.3).split_result(X, np.ones(30))[0]
        ks = MaxMinSplitter(init="kennard_stone", metric="euclidean", train_size=0.7, test_size=0.3).split_result(X)[0]
        assert spxy.train.tolist() == ks.train.tolist()
        assert spxy.metadata["zero_distance_terms"] == ["label"]

    def test_deterministic_without_seed_for_two_way_split(self):
        r1 = SPXYSplitter(train_size=0.5, test_size=0.5).split_result(SMILES_20, np.arange(20.0))[0]
        r2 = SPXYSplitter(train_size=0.5, test_size=0.5).split_result(SMILES_20, np.arange(20.0))[0]
        assert r1.train.tolist() == r2.train.tolist()
        assert r1.test.tolist() == r2.test.tolist()

    def test_three_way_sizes(self):
        splitter = SPXYSplitter(train_size=0.6, valid_size=0.2, test_size=0.2, random_state=0)
        result = splitter.split_result(SMILES_20, np.arange(20.0))[0]
        assert (result.train.size, result.valid.size, result.test.size) == (12, 4, 4)

    def test_multitask_y(self):
        y = np.c_[np.arange(20.0), np.arange(20.0)[::-1] ** 2]
        result = SPXYSplitter(train_size=0.5, test_size=0.5).split_result(SMILES_20, y)[0]
        assert result.train.size == 10

    def test_mahalanobis_metric(self):
        X = _rng(2).normal(size=(30, 3)) @ np.array([[1.0, 0.9, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
        result = SPXYSplitter(metric="mahalanobis", train_size=0.7, test_size=0.3).split_result(X, X[:, 0])[0]
        assert result.train.size == 21

    def test_non_finite_y_raises(self):
        y = np.arange(20.0)
        y[3] = np.nan
        with pytest.raises(LabelError):
            SPXYSplitter(train_size=0.5, test_size=0.5).split_result(SMILES_20, y)

    def test_missing_y_raises(self):
        with pytest.raises(LabelError):
            SPXYSplitter(train_size=0.5, test_size=0.5).split_result(SMILES_20)


class TestOptiSimSplitter:
    def test_cluster_mode_separates_families(self):
        # a subsample covering every record makes the second centre the farthest record, i.e. the
        # other family, so both families always get their own centres
        splitter = OptiSimSplitter(n_picks=4, subsample_size=20, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert set(groups[:10].tolist()).isdisjoint(groups[10:].tolist())

    def test_cluster_mode_keeps_groups_whole(self):
        splitter = OptiSimSplitter(n_picks=5, subsample_size=20, train_size=0.5, test_size=0.5, random_state=0)
        result = splitter.split_result(SMILES_20)[0]
        assert set(result.groups[result.train].tolist()).isdisjoint(result.groups[result.test].tolist())
        centres = result.metadata["centres"]
        assert len(centres) == 5
        # every centre heads its own group
        assert len({int(result.groups[c]) for c in centres}) == 5

    def test_cluster_mode_default_n_picks_is_kmeans_auto_rule(self):
        result = OptiSimSplitter(train_size=0.5, test_size=0.5, random_state=0).split_result(SMILES_20)[0]
        assert result.metadata["n_picks"] == 4  # round(sqrt(20))

    @pytest.mark.parametrize("dest", ["train", "test"])
    def test_pick_mode_fills_destination(self, dest):
        splitter = OptiSimSplitter(
            mode="pick", picked_goes_to=dest, subsample_size=3, train_size=0.6, test_size=0.4, random_state=0
        )
        result = splitter.split_result(SMILES_20)[0]
        assert result.groups is None
        picked = result.metadata["picked"]
        assert set(picked) == set(getattr(result, dest).tolist())
        assert (result.train.size, result.test.size) == (12, 8)

    def test_pick_mode_compute_groups_raises(self):
        with pytest.raises(ParameterError, match="forms no groups"):
            OptiSimSplitter(mode="pick", random_state=0).compute_groups(SMILES_20)

    def test_same_seed_same_split(self):
        r1 = OptiSimSplitter(subsample_size=2, train_size=0.5, test_size=0.5, random_state=3).split_result(SMILES_20)[0]
        r2 = OptiSimSplitter(subsample_size=2, train_size=0.5, test_size=0.5, random_state=3).split_result(SMILES_20)[0]
        assert r1.train.tolist() == r2.train.tolist()
        assert r1.metadata["centres"] == r2.metadata["centres"]

    def test_radius_too_large_warns_and_pick_mode_discards(self):
        splitter = OptiSimSplitter(
            mode="pick", radius=0.95, subsample_size=20, train_size=0.5, test_size=0.5, random_state=0
        )
        with pytest.warns(DegenerateClusterWarning, match="only"):
            result = splitter.split_result(SMILES_20)[0]
        assert result.metadata["n_selected"] < 10
        assert result.train.size == result.metadata["n_selected"]
        assert result.discard.size == 10 - result.metadata["n_selected"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "bogus"},
            {"picked_goes_to": "valid"},
            {"n_picks": 1},
            {"n_picks": 0, "mode": "pick"},
            {"n_picks": True},
            {"subsample_size": 0},
            {"radius": 1.5},
        ],
    )
    def test_invalid_params_raise(self, kwargs):
        with pytest.raises(ParameterError):
            OptiSimSplitter(**kwargs)


class TestMinimalTestSetDissimilaritySplitter:
    # Two 1-D blobs (x = 0..4 and 10..14), labels descending with index, so the two activity bins
    # of a 20% test set are records 0-4 and 5-9. Distance sums: record 4 -> 10 + 40 = 50 is the
    # smallest in bin 1, record 5 -> 10 + 40 = 50 the smallest in bin 2.
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [10.0], [11.0], [12.0], [13.0], [14.0]])
    Y = np.arange(10.0)[::-1].copy()

    def _splitter(self, **kw):
        kw.setdefault("train_size", 8)
        kw.setdefault("test_size", 2)
        return MinimalTestSetDissimilaritySplitter(metric="euclidean", **kw)

    def test_hand_computed_selection(self):
        result = self._splitter().split_result(self.X, self.Y)[0]
        assert result.test.tolist() == [4, 5]
        assert result.metadata["bin_edges"] == [[0, 5], [5, 10]]
        assert result.metadata["test_total_dissimilarity"] == [50.0, 50.0]

    def test_one_record_per_activity_bin(self):
        y = _rng(3).normal(size=40)
        X = _rng(4).normal(size=(40, 3))
        result = MinimalTestSetDissimilaritySplitter(metric="euclidean", train_size=0.8, test_size=0.2).split_result(X, y)[0]
        ranked = sorted(range(40), key=lambda i: -y[i])
        bins = [set(ranked[k * 5:(k + 1) * 5]) for k in range(8)]
        assert result.test.size == 8
        assert all(len(b & set(result.test.tolist())) == 1 for b in bins)

    def test_uneven_bins_differ_by_at_most_one(self):
        y = np.arange(23.0)
        X = _rng(5).normal(size=(23, 2))
        result = self._splitter(train_size=18, test_size=5).split_result(X, y)[0]
        sizes = [stop - start for start, stop in result.metadata["bin_edges"]]
        assert sorted(sizes) == [4, 4, 5, 5, 5]

    def test_deterministic_without_seed(self):
        r1 = MinimalTestSetDissimilaritySplitter(train_size=0.8, test_size=0.2).split_result(SMILES_20, np.arange(20.0))[0]
        r2 = MinimalTestSetDissimilaritySplitter(train_size=0.8, test_size=0.2).split_result(SMILES_20, np.arange(20.0))[0]
        assert r1.test.tolist() == r2.test.tolist()

    def test_three_way_sizes(self):
        splitter = MinimalTestSetDissimilaritySplitter(train_size=0.6, valid_size=0.2, test_size=0.2)
        result = splitter.split_result(SMILES_20, np.arange(20.0))[0]
        assert (result.train.size, result.valid.size, result.test.size) == (12, 4, 4)

    def test_task_index_selects_column(self):
        y = np.c_[np.zeros(10), self.Y]
        result = self._splitter(task_index=1).split_result(self.X, y)[0]
        assert result.test.tolist() == [4, 5]
        with pytest.raises(ParameterError):
            self._splitter(task_index=2).split_result(self.X, y)

    def test_non_finite_or_missing_y_raises(self):
        y = self.Y.copy()
        y[0] = np.nan
        with pytest.raises(LabelError):
            self._splitter().split_result(self.X, y)
        with pytest.raises(LabelError):
            self._splitter().split_result(self.X)

    @pytest.mark.parametrize("task_index", [-1, 1.5, True])
    def test_invalid_task_index(self, task_index):
        with pytest.raises(ParameterError):
            MinimalTestSetDissimilaritySplitter(task_index=task_index)


class TestSupportPointsSplitter:
    @staticmethod
    def _blobs(n_each=50):
        return np.r_[_rng(0).normal(0.0, 1.0, (n_each, 2)), _rng(1).normal(10.0, 1.0, (n_each, 2))]

    def test_half_split_takes_half_of_each_blob(self):
        X = self._blobs()
        result = SupportPointsSplitter(train_size=0.5, test_size=0.5, random_state=0).split_result(X)[0]
        assert int((result.test < 50).sum()) == 25
        assert int((result.test >= 50).sum()) == 25

    def test_proportional_to_blob_sizes(self):
        X = np.r_[_rng(0).normal(0.0, 1.0, (80, 2)), _rng(1).normal(10.0, 1.0, (20, 2))]
        result = SupportPointsSplitter(train_size=0.75, test_size=0.25, random_state=0).split_result(X)[0]
        assert abs(int((result.test < 80).sum()) - 20) <= 1

    def test_larger_test_side_selects_train_by_support_points(self):
        X = self._blobs()
        result = SupportPointsSplitter(train_size=0.2, test_size=0.8, random_state=0).split_result(X)[0]
        assert (result.train.size, result.test.size) == (20, 80)
        assert int((result.train < 50).sum()) == 10

    def test_labels_change_the_selection(self):
        X = self._blobs()
        y = np.r_[np.zeros(50), np.linspace(0.0, 100.0, 50)]
        with_y = SupportPointsSplitter(train_size=0.8, test_size=0.2, random_state=0).split_result(X, y)[0]
        without = SupportPointsSplitter(use_labels=False, train_size=0.8, test_size=0.2, random_state=0).split_result(X, y)[0]
        assert with_y.metadata["used_label_columns"] == 1
        assert without.metadata["used_label_columns"] == 0
        assert with_y.test.tolist() != without.test.tolist()

    def test_converges_and_is_seeded(self):
        X = self._blobs()
        r1 = SupportPointsSplitter(train_size=0.7, test_size=0.3, random_state=4).split_result(X)[0]
        r2 = SupportPointsSplitter(train_size=0.7, test_size=0.3, random_state=4).split_result(X)[0]
        assert r1.metadata["test_selection"]["converged"]
        assert r1.test.tolist() == r2.test.tolist()

    def test_three_way_sizes_with_smiles(self):
        splitter = SupportPointsSplitter(train_size=0.6, valid_size=0.2, test_size=0.2, random_state=0)
        result = splitter.split_result(SMILES_20, np.arange(20.0))[0]
        assert (result.train.size, result.valid.size, result.test.size) == (12, 4, 4)

    def test_helmert_coding_of_categorical_labels(self):
        from chemsplit.splitters.similarity import _encode_label_columns

        coded = _encode_label_columns(np.array(["a", "b", "c", "a"]), "auto", "test")
        assert coded.tolist() == [[-1.0, -1.0], [1.0, -1.0], [0.0, 2.0], [-1.0, -1.0]]
        result = SupportPointsSplitter(train_size=0.5, test_size=0.5, random_state=0).split_result(
            self._blobs(10), np.array(["x", "y", "z", "w"] * 5)
        )[0]
        assert result.metadata["used_label_columns"] == 3

    def test_non_finite_labels_raise(self):
        y = np.arange(100.0)
        y[3] = np.inf
        with pytest.raises(LabelError):
            SupportPointsSplitter(random_state=0).split_result(self._blobs(), y)

    def test_memory_guard(self):
        from chemsplit.exceptions import ScalabilityError

        with pytest.raises(ScalabilityError):
            SupportPointsSplitter(max_memory_bytes=1000, random_state=0).split_result(self._blobs())

    @pytest.mark.parametrize(
        "kwargs",
        [{"label_kind": "bogus"}, {"max_iter": 0}, {"tol": 0.0}, {"use_labels": 1}, {"max_memory_bytes": 0}],
    )
    def test_invalid_params_raise(self, kwargs):
        with pytest.raises(ParameterError):
            SupportPointsSplitter(**kwargs)


class TestDuplexSplitter:
    def test_exact_three_way_sizes(self):
        splitter = DuplexSplitter(train_size=0.6, valid_size=0.2, test_size=0.2)
        result = splitter.split_result(SMILES_20)[0]
        assert (result.train.size, result.valid.size, result.test.size) == (12, 4, 4)

    def test_deterministic_without_seed(self):
        r1 = DuplexSplitter(train_size=0.5, test_size=0.5).split_result(SMILES_20)[0]
        r2 = DuplexSplitter(train_size=0.5, test_size=0.5, random_state=99).split_result(SMILES_20)[0]
        assert r1.test.tolist() == r2.test.tolist()

    def test_test_set_spans_both_families(self):
        result = DuplexSplitter(train_size=0.5, test_size=0.5).split_result(SMILES_20)[0]
        assert 0 < int((result.test < 10).sum()) < 10

    def test_test_set_covers_data_better_than_random(self):
        from chemsplit.splitters.baseline import RandomSplitter

        X = _rng(6).normal(size=(80, 2))
        duplex = DuplexSplitter(metric="euclidean", train_size=0.75, test_size=0.25).split_result(X)[0]
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        random_radius = []
        for seed in range(10):
            test = RandomSplitter(train_size=0.75, test_size=0.25, random_state=seed).split_result(X)[0].test
            random_radius.append(float(np.max(np.min(D[:, test], axis=1))))
        assert duplex.metadata["coverage_radius"]["test"] < min(random_radius)


class TestDOptimalSplitter:
    GRID = np.array([[x, y] for x in range(5) for y in range(5)], dtype=float)

    def test_grid_corners_are_d_optimal(self):
        result = DOptimalSplitter(n_components=2, train_size=4, test_size=21).split_result(self.GRID)[0]
        assert sorted(map(tuple, self.GRID[result.train].tolist())) == [(0, 0), (0, 4), (4, 0), (4, 4)]

    def test_log_det_beats_start_and_random_subsets(self):
        from chemsplit.splitters.similarity import _d_optimal_exchange

        X = _rng(7).normal(size=(60, 4))
        result = DOptimalSplitter(train_size=0.5, test_size=0.5, n_components=4).split_result(X)[0]
        achieved = result.metadata["train_design"]["log_det"]
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        U, S, _ = np.linalg.svd(Xs, full_matrices=False)
        design = np.hstack([np.ones((60, 1)), U[:, :4] * S[:4]])
        rng = _rng(8)
        for _ in range(20):
            rows = rng.choice(60, size=30, replace=False)
            _, random_log_det = np.linalg.slogdet(design[rows].T @ design[rows] + 1e-8 * np.eye(5))
            assert achieved >= random_log_det - 1e-9
        # the exchange never ends below its starting design
        start = list(range(30))
        _, start_log_det = np.linalg.slogdet(design[start].T @ design[start] + 1e-8 * np.eye(5))
        _, meta = _d_optimal_exchange(design, list(range(60)), start, 1e-8, 100)
        assert meta["log_det"] >= start_log_det
        assert meta["n_swaps"] > 0

    def test_deterministic_without_seed_and_random_init_is_seeded(self):
        X = _rng(9).normal(size=(40, 3))
        r1 = DOptimalSplitter(train_size=0.75, test_size=0.25).split_result(X)[0]
        r2 = DOptimalSplitter(train_size=0.75, test_size=0.25, random_state=5).split_result(X)[0]
        assert r1.train.tolist() == r2.train.tolist()
        r3 = DOptimalSplitter(init="random", train_size=0.75, test_size=0.25, random_state=5).split_result(X)[0]
        r4 = DOptimalSplitter(init="random", train_size=0.75, test_size=0.25, random_state=5).split_result(X)[0]
        assert r3.train.tolist() == r4.train.tolist()

    def test_three_way_sizes_with_smiles(self):
        result = DOptimalSplitter(train_size=0.6, valid_size=0.2, test_size=0.2, n_components=3).split_result(SMILES_20)[0]
        assert (result.train.size, result.valid.size, result.test.size) == (12, 4, 4)

    def test_too_many_components_raise(self):
        with pytest.raises(ParameterError):
            DOptimalSplitter(n_components=9, train_size=4, test_size=21).split_result(self.GRID)

    @pytest.mark.parametrize(
        "kwargs", [{"n_components": 0}, {"init": "bogus"}, {"ridge": -1.0}, {"max_passes": 0}]
    )
    def test_invalid_params_raise(self, kwargs):
        with pytest.raises(ParameterError):
            DOptimalSplitter(**kwargs)


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


# coverage additions


class TestSimilarityThresholdSplitterStrategies:
    def test_greedy_prune_strategy(self):
        splitter = SimilarityThresholdSplitter(
            threshold=0.3, strategy="greedy_prune", train_size=0.5, test_size=0.5,
            allow_discard=True, random_state=0,
        )
        result = splitter.split_result(SMILES_20)[0]
        assert result.n_records == 20

    def test_greedy_prune_disallow_discard_raises_when_needed(self):
        splitter = SimilarityThresholdSplitter(
            threshold=0.01, strategy="greedy_prune", train_size=0.5, test_size=0.5,
            allow_discard=False, max_discard_frac=0.0, random_state=0,
        )
        with pytest.raises(ConstraintUnsatisfiableError):
            splitter.split_result(SMILES_20)

    @pytest.mark.parametrize("seed_selection", ["random", "most_central", "most_peripheral"])
    def test_seeded_growth_strategy(self, seed_selection):
        splitter = SimilarityThresholdSplitter(
            threshold=0.3, strategy="seeded_growth", seed_selection=seed_selection,
            train_size=0.5, test_size=0.5, random_state=0,
        )
        result = splitter.split_result(SMILES_20)[0]
        assert result.n_records == 20

    def test_max_discard_frac_validation(self):
        with pytest.raises(ParameterError):
            SimilarityThresholdSplitter(max_discard_frac=1.5)


class TestButinaSplitterSingletonPolicies:
    def test_nearest_cluster_singleton_policy(self):
        splitter = ButinaSplitter(
            cutoff=0.15, singleton_policy="nearest_cluster", train_size=0.5, test_size=0.5, random_state=0,
        )
        groups = splitter.compute_groups(SMILES_20)
        assert groups.shape[0] == 20

    def test_shared_group_singleton_policy(self):
        splitter = ButinaSplitter(
            cutoff=0.15, singleton_policy="shared_group", train_size=0.5, test_size=0.5, random_state=0,
        )
        groups = splitter.compute_groups(SMILES_20)
        assert groups.shape[0] == 20

    def test_nearest_cluster_falls_back_when_no_non_singleton(self):
        # All-singleton case: nearest_cluster's fallback-to-own_group warning fires, and the
        # splitter then still (correctly) raises DegenerateGroupingError since every record ends
        # up its own group -- both are exercised together here.
        features = np.eye(10, dtype=np.uint8)
        splitter = ButinaSplitter(
            cutoff=0.5, singleton_policy="nearest_cluster", train_size=0.5, test_size=0.5, random_state=0,
        )
        with pytest.warns(Warning), pytest.raises(DegenerateGroupingError):
            splitter.compute_groups(features)

    def test_reorder_true(self):
        splitter = ButinaSplitter(cutoff=0.4, reorder=True, train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert groups.shape[0] == 20

    def test_sparse_algorithm(self):
        splitter = ButinaSplitter(cutoff=0.4, algorithm="sparse", train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert groups.shape[0] == 20

    def test_invalid_cutoff_is_and_singleton_policy(self):
        with pytest.raises(ParameterError):
            ButinaSplitter(cutoff_is="bogus")
        with pytest.raises(ParameterError):
            ButinaSplitter(singleton_policy="bogus")


class TestKMeansClusterSplitterMore:
    def test_minibatch_kmeans_algorithm(self):
        splitter = KMeansClusterSplitter(
            n_clusters=2, algorithm="minibatch_kmeans", train_size=0.5, test_size=0.5, random_state=0,
        )
        groups = splitter.compute_groups(SMILES_20)
        assert len(set(groups.tolist())) >= 1

    def test_birch_algorithm(self):
        splitter = KMeansClusterSplitter(n_clusters=2, algorithm="birch", train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert len(set(groups.tolist())) >= 1

    def test_agglomerative_ward_euclidean(self):
        splitter = KMeansClusterSplitter(
            n_clusters=2, algorithm="agglomerative", linkage="ward", metric="euclidean",
            train_size=0.5, test_size=0.5, random_state=0,
        )
        groups = splitter.compute_groups(SMILES_20)
        assert len(set(groups.tolist())) >= 1

    def test_auto_n_clusters_n_over_50(self):
        splitter = KMeansClusterSplitter(n_clusters="auto", auto_rule="n_over_50", train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert groups.shape[0] == 20

    def test_auto_n_clusters_silhouette(self):
        splitter = KMeansClusterSplitter(
            n_clusters="auto", auto_rule="silhouette", auto_range=(2, 4),
            train_size=0.5, test_size=0.5, random_state=0,
        )
        groups = splitter.compute_groups(SMILES_20)
        assert groups.shape[0] == 20


class TestDensityClusterSplitterNoisePolicies:
    def test_noise_policy_test_forces_noise_into_test(self):
        smiles = ["CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC", "c1ccccc1N"]
        splitter = DensityClusterSplitter(
            eps=0.15, min_samples=2, noise_policy="test", train_size=0.5, test_size=0.5, random_state=0,
        )
        result = splitter.split_result(smiles)[0]
        assert 5 in result.test.tolist()
        assert 5 not in result.train.tolist()

    def test_noise_policy_train_forces_noise_into_train(self):
        smiles = ["CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC", "c1ccccc1N"]
        splitter = DensityClusterSplitter(
            eps=0.15, min_samples=2, noise_policy="train", train_size=0.5, test_size=0.5, random_state=0,
        )
        result = splitter.split_result(smiles)[0]
        assert 5 in result.train.tolist()
        assert 5 not in result.test.tolist()

    def test_noise_policy_discard(self):
        smiles = ["CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC", "c1ccccc1N"]
        splitter = DensityClusterSplitter(
            eps=0.15, min_samples=2, noise_policy="discard", train_size=0.5, test_size=0.5, random_state=0,
        )
        result = splitter.split_result(smiles)[0]
        assert 5 in result.discard.tolist()

    def test_noise_policy_distribute(self):
        smiles = ["CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC", "c1ccccc1N"]
        splitter = DensityClusterSplitter(
            eps=0.15, min_samples=2, noise_policy="distribute", train_size=0.5, test_size=0.5, random_state=0,
        )
        result = splitter.split_result(smiles)[0]
        assert result.n_records == 6

    def test_hdbscan_algorithm(self):
        splitter = DensityClusterSplitter(
            algorithm="hdbscan", min_cluster_size=2, train_size=0.5, test_size=0.5, random_state=0,
        )
        groups = splitter.compute_groups(SMILES_20)
        assert groups.shape[0] == 20

    def test_all_noise_raises(self):
        features = np.eye(10, dtype=np.uint8) * 255
        splitter = DensityClusterSplitter(eps=1e-9, min_samples=5, noise_policy="own_groups", random_state=0)
        with pytest.raises((DegenerateGroupingError, ConstraintUnsatisfiableError)):
            splitter.compute_groups(features)


class TestSpectralSplitterMore:
    def test_threshold_graph(self):
        splitter = SpectralSplitter(n_clusters=2, graph="threshold", threshold=0.2, train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert len(set(groups.tolist())) >= 1

    def test_full_graph(self):
        splitter = SpectralSplitter(n_clusters=2, graph="full", train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert len(set(groups.tolist())) >= 1

    def test_rw_laplacian(self):
        splitter = SpectralSplitter(n_clusters=2, laplacian="rw", knn_k=5, train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert len(set(groups.tolist())) >= 1

    def test_unnormalized_laplacian_does_not_drop_first(self):
        splitter = SpectralSplitter(
            n_clusters=2, laplacian="unnormalized", drop_first=False, knn_k=5,
            train_size=0.5, test_size=0.5, random_state=0,
        )
        groups = splitter.compute_groups(SMILES_20)
        assert len(set(groups.tolist())) >= 1

    def test_discretize_assign(self):
        splitter = SpectralSplitter(n_clusters=2, assign="discretize", knn_k=5, train_size=0.5, test_size=0.5, random_state=0)
        groups = splitter.compute_groups(SMILES_20)
        assert len(set(groups.tolist())) >= 1


class TestLeaveOneClusterOutSplitterMore:
    def test_own_fold_small_cluster_policy(self):
        splitter = LeaveOneClusterOutSplitter(
            clusterer=ButinaSplitter(cutoff=0.15, random_state=0),
            small_cluster_policy="own_fold", max_folds=None,
        )
        results = splitter.split_result(SMILES_20)
        assert len(results) >= 1

    def test_pool_small_cluster_policy(self):
        splitter = LeaveOneClusterOutSplitter(
            clusterer=ButinaSplitter(cutoff=0.15, random_state=0),
            small_cluster_policy="pool", max_folds=None,
        )
        results = splitter.split_result(SMILES_20)
        assert len(results) >= 1

    def test_fold_order_size_asc_and_index(self):
        for order in ("size_asc", "index"):
            splitter = LeaveOneClusterOutSplitter(
                clusterer=ButinaSplitter(cutoff=0.15, random_state=0),
                fold_order=order, max_folds=None,
            )
            results = splitter.split_result(SMILES_20)
            assert len(results) >= 1

    def test_get_n_splits_with_x(self):
        splitter = LeaveOneClusterOutSplitter(
            clusterer=ButinaSplitter(cutoff=0.15, random_state=0), max_folds=None,
        )
        n = splitter.get_n_splits(SMILES_20)
        assert n == len(splitter.split_result(SMILES_20))

    def test_max_folds_caps_fold_count(self):
        splitter = LeaveOneClusterOutSplitter(
            clusterer=ButinaSplitter(cutoff=0.15, random_state=0), max_folds=1,
        )
        results = splitter.split_result(SMILES_20)
        assert len(results) == 1
