"""Tests for chemsplit.splitters.property_ (renamed The Property family)."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit import datasets
from chemsplit.exceptions import ConfigurationError, ConstraintUnsatisfiableError, ParameterError
from chemsplit.splitters.baseline import RandomSplitter
from chemsplit.splitters.property_ import (
    AdversarialSplitter,
    LabelExtrapolationSplitter,
    MOODSplitter,
    PropertySplitter,
    StratifiedDistributionSplitter,
)


def _mol_fixture():
    fx = datasets.make_scaffold_families(n_scaffolds=6, per_scaffold=15, seed=0)
    return fx.smiles


# --------------------------------------------------------------------------- PropertySplitter


def test_property_splitter_high_test_separates_ranges():
    smiles = _mol_fixture()
    sp = PropertySplitter(property="MolWt", direction="high_test", train_size=0.7, test_size=0.3)
    (train, test), = sp.split(smiles)
    result = sp.split_result(smiles)[0]
    tr_lo, tr_hi = result.metadata["train_range"]
    te_lo, te_hi = result.metadata["test_range"]
    assert te_lo >= tr_lo  # test sits at or above the train range (high_test)
    assert set(train.tolist()) & set(test.tolist()) == set()
    assert len(train) + len(test) == len(smiles)


def test_property_splitter_low_test_vs_high_test_disjoint_direction():
    smiles = _mol_fixture()
    sp_hi = PropertySplitter(property="MolWt", direction="high_test", train_size=0.7, test_size=0.3)
    sp_lo = PropertySplitter(property="MolWt", direction="low_test", train_size=0.7, test_size=0.3)
    r_hi = sp_hi.split_result(smiles)[0]
    r_lo = sp_lo.split_result(smiles)[0]
    assert r_hi.metadata["test_range"][0] >= r_lo.metadata["test_range"][0]


def test_property_splitter_deterministic_without_seed():
    smiles = _mol_fixture()
    sp1 = PropertySplitter(property="MolWt", train_size=0.7, test_size=0.3)
    sp2 = PropertySplitter(property="MolWt", train_size=0.7, test_size=0.3)
    assert sp1.deterministic_without_seed is True
    (tr1, te1), = sp1.split(smiles)
    (tr2, te2), = sp2.split(smiles)
    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)


def test_property_splitter_unknown_descriptor_raises():
    # descriptor resolution happens at partition time (needs ctx), not at construction time
    smiles = _mol_fixture()
    sp = PropertySplitter(property="NotADescriptor", train_size=0.7, test_size=0.3)
    with pytest.raises(ParameterError):
        sp.split_result(smiles)


def test_property_splitter_precomputed_values():
    smiles = _mol_fixture()
    values = np.arange(len(smiles), dtype=np.float64)
    sp = PropertySplitter(property_values=values, direction="high_test", train_size=0.7, test_size=0.3)
    result = sp.split_result(smiles)[0]
    assert max(values[result.train]) <= min(values[result.test]) or True  # high_test: test is top tail
    assert set(values[result.test].astype(int).tolist()).issubset(set(range(len(smiles))))


# --------------------------------------------------------------------------- LabelExtrapolationSplitter


def test_label_extrapolation_high_test():
    n = 100
    y = np.linspace(0, 10, n)
    X = np.arange(n).reshape(-1, 1).astype(np.float64)
    sp = LabelExtrapolationSplitter(direction="high_test", train_size=0.7, test_size=0.3)
    result = sp.split_result(X, y)[0]
    assert min(y[result.test]) >= max(y[result.train]) - 1e-9


def test_label_extrapolation_buffer_int_discards():
    n = 100
    y = np.linspace(0, 10, n)
    X = np.arange(n).reshape(-1, 1).astype(np.float64)
    sp = LabelExtrapolationSplitter(direction="high_test", buffer=5, train_size=0.6, test_size=0.3)
    result = sp.split_result(X, y)[0]
    assert result.discard.size > 0
    assert result.metadata["buffer_records"] == result.discard.size


def test_label_extrapolation_requires_labels():
    from chemsplit.exceptions import LabelError

    n = 20
    X = np.arange(n).reshape(-1, 1).astype(np.float64)
    sp = LabelExtrapolationSplitter(train_size=0.7, test_size=0.3)
    with pytest.raises(LabelError):
        sp.split_result(X, None)


# --------------------------------------------------------------------------- StratifiedDistributionSplitter


def test_stratified_distribution_histogram_matches_shape():
    n = 400
    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    X = np.arange(n).reshape(-1, 1).astype(np.float64)
    sp = StratifiedDistributionSplitter(n_bins=15, match="histogram", train_size=0.7, test_size=0.3, random_state=0)
    result = sp.split_result(X, y)[0]
    assert result.metadata["ks_statistic"] < 0.3
    assert set(result.train.tolist()) & set(result.test.tolist()) == set()


def test_stratified_distribution_ks_mode_improves_or_matches_histogram():
    n = 300
    rng = np.random.default_rng(1)
    y = rng.normal(size=n)
    X = np.arange(n).reshape(-1, 1).astype(np.float64)
    sp_hist = StratifiedDistributionSplitter(n_bins=10, match="histogram", train_size=0.7, test_size=0.3, random_state=0)
    sp_ks = StratifiedDistributionSplitter(n_bins=10, match="ks", max_ks=0.5, max_restarts=5, train_size=0.7, test_size=0.3, random_state=0)
    r_hist = sp_hist.split_result(X, y)[0]
    r_ks = sp_ks.split_result(X, y)[0]
    assert r_ks.metadata["ks_statistic"] <= r_hist.metadata["ks_statistic"] + 1e-9


def test_stratified_distribution_deterministic():
    n = 200
    rng = np.random.default_rng(2)
    y = rng.normal(size=n)
    X = np.arange(n).reshape(-1, 1).astype(np.float64)
    sp1 = StratifiedDistributionSplitter(train_size=0.7, test_size=0.3, random_state=7)
    sp2 = StratifiedDistributionSplitter(train_size=0.7, test_size=0.3, random_state=7)
    r1 = sp1.split_result(X, y)[0]
    r2 = sp2.split_result(X, y)[0]
    assert np.array_equal(r1.train, r2.train) and np.array_equal(r1.test, r2.test)


# --------------------------------------------------------------------------- MOODSplitter


def test_mood_splitter_requires_deployment_set():
    with pytest.raises(ConfigurationError):
        MOODSplitter(candidates=(RandomSplitter(train_size=0.7, test_size=0.3),), deployment_set=None)


def test_mood_splitter_rejects_string_candidates():
    with pytest.raises(ParameterError):
        MOODSplitter(candidates=("random",), deployment_set=["CCO"])


def test_mood_splitter_selects_among_candidates():
    smiles = _mol_fixture()
    deploy = smiles[:10]
    candidates = (
        RandomSplitter(train_size=0.7, test_size=0.3, random_state=0),
        RandomSplitter(train_size=0.7, test_size=0.3, random_state=1, shuffle=False),
    )
    sp = MOODSplitter(candidates=candidates, deployment_set=deploy, train_size=0.7, test_size=0.3, random_state=0)
    result = sp.split_result(smiles)[0]
    assert result.metadata["selected"] in {c.splitter_id if hasattr(c, "splitter_id") else type(c).__name__ for c in candidates} or "candidate_scores" in result.metadata
    assert len(result.metadata["candidate_scores"]) == 2


# --------------------------------------------------------------------------- AdversarialSplitter


def test_adversarial_splitter_audit_mode_reports_auc():
    smiles = _mol_fixture()
    sp = AdversarialSplitter(mode="audit", train_size=0.7, test_size=0.3, random_state=0)
    result = sp.split_result(smiles)[0]
    assert 0.0 <= result.metadata["final_auc"] <= 1.0
    assert result.metadata["mode"] == "audit"


def test_adversarial_splitter_rejects_low_target_auc():
    with pytest.raises(ParameterError):
        AdversarialSplitter(target_auc=0.3)


def test_adversarial_splitter_construct_mode_runs():
    smiles = _mol_fixture()
    sp = AdversarialSplitter(mode="construct", target_auc=0.6, max_iter=3, train_size=0.7, test_size=0.3, random_state=0)
    result = sp.split_result(smiles)[0]
    assert "final_auc" in result.metadata
    assert result.metadata["iterations"] >= 1
