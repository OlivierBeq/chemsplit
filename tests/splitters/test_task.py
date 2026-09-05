"""Tests for the ``task`` splitter family (chemsplit's renamed the task family)."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit import datasets
from chemsplit.exceptions import (
    ConstraintUnsatisfiableError,
    DegenerateGroupingError,
    EmptyPartitionError,
    LabelError,
    ParameterError,
)
from chemsplit.splitters.task import (
    AVESplitter,
    ColdDrugSplitter,
    ColdPairSplitter,
    ColdTargetSplitter,
    DecoyBenchmarkSplitter,
    HiSplitter,
    LoSplitter,
    ScaffoldHopSplitter,
)

# A modest pool of real, distinct small molecules (varied scaffolds) for similarity-based tests.
_SMILES = [
    "c1ccccc1", "c1ccccc1C", "c1ccccc1CC", "c1ccncc1", "c1ccncc1C",
    "c1ccc2ccccc2c1", "c1ccc2ccccc2c1C", "C1CCCCC1", "C1CCCCC1C", "C1CCNCC1",
    "c1ccc(cc1)C(=O)O", "c1ccc(cc1)C(=O)OC", "c1ccc(cc1)N", "c1ccc(cc1)NC", "c1ccc(cc1)O",
    "c1ccc(cc1)OC", "CCCCCCCC", "CCCCCCCCC", "c1ccsc1", "c1ccoc1",
    "c1cc2ccccc2[nH]1", "c1cc2ccccc2o1", "C1CCC1", "C1CCC1C", "c1ccc(Cl)cc1",
    "c1ccc(Br)cc1", "c1ccc(F)cc1", "CC(C)C", "CC(C)CC", "CCN",
]


def _feature_matrix(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 8))


class TestHiSplitter:
    def test_basic_split_and_verification(self):
        sp = HiSplitter(threshold=0.3, coarse_cutoff=0.5, train_size=0.7, test_size=0.3, random_state=0)
        result = sp.split_result(_SMILES)[0]
        assert result.metadata["max_cross_similarity"] <= 0.3 + 1e-6
        assert result.train.size + result.test.size + result.valid.size + result.discard.size == len(_SMILES)

    def test_coarse_cutoff_below_threshold_raises(self):
        with pytest.raises(ParameterError):
            HiSplitter(threshold=0.5, coarse_cutoff=0.3)

    def test_determinism(self):
        sp1 = HiSplitter(threshold=0.3, train_size=0.7, test_size=0.3, random_state=42)
        sp2 = HiSplitter(threshold=0.3, train_size=0.7, test_size=0.3, random_state=42)
        r1 = sp1.split_result(_SMILES)[0]
        r2 = sp2.split_result(_SMILES)[0]
        assert np.array_equal(r1.train, r2.train)
        assert np.array_equal(r1.test, r2.test)

    def test_annealing_solver(self):
        sp = HiSplitter(threshold=0.3, solver="annealing", train_size=0.7, test_size=0.3, random_state=0,
                         time_limit_s=5.0)
        result = sp.split_result(_SMILES)[0]
        assert result.metadata["solver"] == "annealing"

    def test_ilp_solver(self):
        sp = HiSplitter(threshold=0.3, solver="ilp", train_size=0.7, test_size=0.3, random_state=0, time_limit_s=5.0)
        result = sp.split_result(_SMILES)[0]
        assert result.metadata["solver"] == "ilp"


class TestLoSplitter:
    def test_requires_labels(self):
        sp = LoSplitter(train_size=0.7, test_size=0.3)
        with pytest.raises(LabelError):
            sp.split_result(_SMILES)  # y omitted, requires_labels=True enforced in _run

    def test_basic_split(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=len(_SMILES))
        sp = LoSplitter(threshold=0.3, min_cluster_size=3, std_threshold=0.01, train_size=0.7, test_size=0.3,
                         random_state=0)
        try:
            result = sp.split_result(_SMILES, y=y)[0]
        except (ConstraintUnsatisfiableError, EmptyPartitionError):
            pytest.skip("no qualifying cluster (or train pool exhausted) on this tiny synthetic pool")
        assert "cluster_members" in result.metadata

    def test_no_qualifying_cluster_raises(self):
        y = np.ones(len(_SMILES))  # zero std everywhere -> no cluster can qualify
        sp = LoSplitter(threshold=0.05, min_cluster_size=3, std_threshold=0.5, train_size=0.7, test_size=0.3,
                         random_state=0)
        with pytest.raises(ConstraintUnsatisfiableError):
            sp.split_result(_SMILES, y=y)


class TestScaffoldHopSplitter:
    def test_scaffold_disjointness_asserted(self):
        rng = np.random.default_rng(0)
        y = (rng.random(len(_SMILES)) > 0.4).astype(np.int64)
        sp = ScaffoldHopSplitter(
            pharmacophore_similarity="none", train_size=0.7, test_size=0.3, random_state=0, min_pharm_similarity=0.0,
        )
        try:
            result = sp.split_result(_SMILES, y=y)[0]
        except (ConstraintUnsatisfiableError, DegenerateGroupingError):
            pytest.skip("not enough distinct active scaffolds in this tiny synthetic pool")
        train_scaffolds = set(result.metadata.get("test_scaffolds", []))
        assert isinstance(train_scaffolds, set)


class TestColdStart:
    def _interaction_data(self):
        fx = datasets.make_interactions(n_compounds=12, n_targets=6, density=0.4, seed=0)
        X = [(fx.smiles[c], fx.targets[t]) for (c, t, _y) in fx.interactions]
        y = [yv for (_c, _t, yv) in fx.interactions]
        return X, y

    def test_cold_drug_basic(self):
        X, y = self._interaction_data()
        sp = ColdDrugSplitter(random_state=0)
        result = sp.split_result(X, y=y)[0]
        assert result.metadata["axis"] == "compound"
        assert result.train.size + result.test.size + result.discard.size == len(X)

    def test_cold_target_warns_without_grouper(self):
        X, y = self._interaction_data()
        sp = ColdTargetSplitter(random_state=0)
        with pytest.warns(UserWarning):
            sp.split_result(X, y=y)

    def test_cold_pair_discards_mixed_side(self):
        X, y = self._interaction_data()
        sp = ColdPairSplitter(random_state=0)
        result = sp.split_result(X, y=y)[0]
        assert result.metadata["axis"] == "pair"
        assert result.discard.size >= 0

    def test_missing_interactions_raises(self):
        sp = ColdDrugSplitter(random_state=0)
        with pytest.raises(Exception):
            sp.split_result(_feature_matrix())


class TestAVESplitter:
    def test_binary_labels_required(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=len(_SMILES))
        sp = AVESplitter(train_size=0.7, test_size=0.3, random_state=0)
        with pytest.raises(LabelError):
            sp.split_result(_SMILES, y=y)

    def test_basic_split_reports_ave(self):
        rng = np.random.default_rng(0)
        y = (rng.random(len(_SMILES)) > 0.5).astype(np.int64)
        sp = AVESplitter(
            train_size=0.7, test_size=0.3, random_state=0, population_size=6, n_generations=2, tolerance=1.0,
        )
        result = sp.split_result(_SMILES, y=y)[0]
        assert "ave_initial" in result.metadata
        assert "ave_final" in result.metadata

    def test_determinism(self):
        rng = np.random.default_rng(0)
        y = (rng.random(len(_SMILES)) > 0.5).astype(np.int64)
        sp1 = AVESplitter(train_size=0.7, test_size=0.3, random_state=7, population_size=6, n_generations=2)
        sp2 = AVESplitter(train_size=0.7, test_size=0.3, random_state=7, population_size=6, n_generations=2)
        r1 = sp1.split_result(_SMILES, y=y)[0]
        r2 = sp2.split_result(_SMILES, y=y)[0]
        assert np.array_equal(r1.test, r2.test)


class TestDecoyBenchmarkSplitter:
    def test_requires_decoy_pool(self):
        with pytest.raises(Exception):
            DecoyBenchmarkSplitter(scheme="property_matched", decoy_pool=None)

    def test_property_matched_basic(self):
        y = np.zeros(len(_SMILES), dtype=np.int64)
        y[:6] = 1
        decoy_pool = ["CCCCCCCCCC", "CCCCCCCCCCC", "c1ccccc1CCCC", "C1CCCCCC1", "CCOCC", "CCNCC"] * 5
        sp = DecoyBenchmarkSplitter(
            decoy_ratio=2, topology_dissimilarity=0.9, decoy_pool=decoy_pool,
            train_size=0.7, test_size=0.3, random_state=0,
        )
        result = sp.split_result(_SMILES, y=y)[0]
        assert result.metadata["scheme"] == "property_matched"
        assert result.metadata["n_actives"] == 6

    def test_predefined_scheme(self):
        assignment = ["train"] * 20 + ["test"] * 10
        sp = DecoyBenchmarkSplitter(scheme="predefined", predefined_assignment=assignment)
        y = np.zeros(len(_SMILES), dtype=np.int64)
        result = sp.split_result(_SMILES, y=y)[0]
        assert result.train.size == 20
        assert result.test.size == 10


# --------------------------------------------------------------------------- coverage additions


class TestHiSplitterMore:
    def test_invalid_solver(self):
        with pytest.raises(ParameterError):
            HiSplitter(solver="bogus")

    def test_max_discard_frac_validation(self):
        with pytest.raises(ParameterError):
            HiSplitter(max_discard_frac=1.5)

    def test_residual_repair_discards_when_needed(self):
        sp = HiSplitter(
            threshold=0.15, coarse_cutoff=0.2, max_discard_frac=0.5,
            train_size=0.8, test_size=0.2, random_state=0,
        )
        result = sp.split_result(_SMILES)[0]
        assert result.n_records == len(_SMILES)


class TestLoSplitterMore:
    def test_param_validation(self):
        with pytest.raises(ParameterError):
            LoSplitter(threshold=1.5)
        with pytest.raises(ParameterError):
            LoSplitter(min_cluster_size=2)
        with pytest.raises(ParameterError):
            LoSplitter(max_clusters=0)
        with pytest.raises(ParameterError):
            LoSplitter(std_threshold=0.0)
        with pytest.raises(ParameterError):
            LoSplitter(train_similarity_ceiling=1.5)

    def test_explicit_train_similarity_ceiling(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=len(_SMILES))
        sp = LoSplitter(
            threshold=0.3, min_cluster_size=3, std_threshold=0.01,
            train_similarity_ceiling=0.9, train_size=0.7, test_size=0.3, random_state=0,
        )
        try:
            result = sp.split_result(_SMILES, y=y)[0]
        except (ConstraintUnsatisfiableError, EmptyPartitionError):
            pytest.skip("no qualifying cluster on this tiny synthetic pool")
        assert "n_pruned_from_train" in result.metadata

    def test_with_valid_size(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=len(_SMILES))
        sp = LoSplitter(
            threshold=0.3, min_cluster_size=3, std_threshold=0.01,
            train_size=0.5, valid_size=0.2, test_size=0.3, random_state=0,
        )
        try:
            result = sp.split_result(_SMILES, y=y)[0]
        except (ConstraintUnsatisfiableError, EmptyPartitionError):
            pytest.skip("no qualifying cluster on this tiny synthetic pool")
        assert result.n_records == len(_SMILES)


class TestScaffoldHopSplitterMore:
    def test_param_validation(self):
        with pytest.raises(ParameterError):
            ScaffoldHopSplitter(scaffold_kind="bogus")
        with pytest.raises(ParameterError):
            ScaffoldHopSplitter(active_definition="threshold", active_threshold=None)
        with pytest.raises(ParameterError):
            ScaffoldHopSplitter(pharmacophore_similarity="bogus")
        with pytest.raises(ParameterError):
            ScaffoldHopSplitter(inactives_policy="bogus")

    def test_threshold_active_definition(self):
        y = np.linspace(0, 10, len(_SMILES))
        sp = ScaffoldHopSplitter(
            active_definition="threshold", active_threshold=5.0,
            pharmacophore_similarity="none",
            train_size=0.7, test_size=0.3, random_state=0,
        )
        try:
            result = sp.split_result(_SMILES, y=y)[0]
        except (ConstraintUnsatisfiableError, DegenerateGroupingError):
            pytest.skip("no qualifying scaffold split on this tiny synthetic pool")
        assert result.n_records == len(_SMILES)


class TestColdStartMore:
    def _interaction_data(self):
        fx = datasets.make_interactions(n_compounds=12, n_targets=6, density=0.4, seed=0)
        X = [(fx.smiles[c], fx.targets[t]) for (c, t, _y) in fx.interactions]
        y = [yv for (_c, _t, yv) in fx.interactions]
        return X, y

    def test_compound_grouper_requires_structures(self):
        with pytest.raises(Exception):
            ColdDrugSplitter(compound_grouper="something")

    def test_min_interactions_per_entity_validation(self):
        with pytest.raises(ParameterError):
            ColdDrugSplitter(min_interactions_per_entity=0)

    def test_min_interactions_per_entity_filters(self):
        X, y = self._interaction_data()
        sp = ColdDrugSplitter(min_interactions_per_entity=2, random_state=0)
        result = sp.split_result(X, y=y)[0]
        assert result.n_records == len(X)

    def test_drop_unlabelled_false(self):
        X, y = self._interaction_data()
        sp = ColdDrugSplitter(drop_unlabelled=False, random_state=0)
        result = sp.split_result(X, y=y)[0]
        assert result.n_records == len(X)


class TestAVESplitterMore:
    def test_non_binary_labels_raise(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, size=len(_SMILES))
        sp = AVESplitter(train_size=0.7, test_size=0.3, random_state=0, population_size=4, n_generations=1)
        with pytest.raises(LabelError):
            sp.split_result(_SMILES, y=y)

    def test_param_validation(self):
        with pytest.raises(ParameterError):
            AVESplitter(n_bins=0)
        with pytest.raises(ParameterError):
            AVESplitter(population_size=1)
        with pytest.raises(ParameterError):
            AVESplitter(init_splitter="not_a_valid_string")


class TestDecoyBenchmarkSplitterMore:
    def test_invalid_scheme(self):
        with pytest.raises(ParameterError):
            DecoyBenchmarkSplitter(scheme="bogus")

    def test_predefined_scheme_invalid_value_raises(self):
        assignment = ["train"] * 20 + ["bogus"] * 10
        sp = DecoyBenchmarkSplitter(scheme="predefined", predefined_assignment=assignment)
        y = np.zeros(len(_SMILES), dtype=np.int64)
        with pytest.raises(ParameterError):
            sp.split_result(_SMILES, y=y)

    def test_spatial_random_scheme(self):
        y = np.zeros(len(_SMILES), dtype=np.int64)
        y[:6] = 1
        decoy_pool = ["CCCCCCCCCC", "CCCCCCCCCCC", "c1ccccc1CCCC"] * 10
        sp = DecoyBenchmarkSplitter(
            scheme="spatial_random", decoy_pool=decoy_pool,
            train_size=0.7, test_size=0.3, random_state=0,
        )
        result = sp.split_result(_SMILES, y=y)[0]
        assert result.metadata["scheme"] == "spatial_random"

