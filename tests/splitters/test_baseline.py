"""Tests for chemsplit.splitters.baseline (renamed The Baseline family)."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit.exceptions import (
    ConfigurationError,
    EmptyPartitionError,
    LabelError,
    ParameterError,
)
from chemsplit.splitters.baseline import (
    KFoldSplitter,
    MonteCarloSplitter,
    PredefinedSplitter,
    RandomSplitter,
    StratifiedRandomSplitter,
)


def _X(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64).reshape(-1, 1)


# --------------------------------------------------------------------------- RandomSplitter


def test_random_splitter_basic_shapes():
    X = _X(20)
    sp = RandomSplitter(train_size=0.7, test_size=0.3, random_state=0)
    (train, test), = sp.split(X)
    assert len(train) + len(test) == 20
    assert set(train.tolist()) & set(test.tolist()) == set()


def test_random_splitter_determinism():
    X = _X(30)
    sp1 = RandomSplitter(random_state=42)
    sp2 = RandomSplitter(random_state=42)
    (tr1, te1), = sp1.split(X)
    (tr2, te2), = sp2.split(X)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(te1, te2)


def test_random_splitter_no_shuffle_is_index_split():
    X = _X(10)
    sp = RandomSplitter(shuffle=False, train_size=0.8, test_size=0.2)
    (train, test), = sp.split(X)
    assert train.tolist() == list(range(8))
    assert test.tolist() == list(range(8, 10))
    assert sp.deterministic_without_seed is True


def test_random_splitter_shuffle_param_type_checked():
    with pytest.raises(ParameterError):
        RandomSplitter(shuffle="yes")


def test_random_splitter_split_with_validation_yields_triple():
    X = _X(20)
    sp = RandomSplitter(train_size=0.6, valid_size=0.2, test_size=0.2, random_state=0)
    (train, valid, test), = sp.split_with_validation(X)
    assert len(train) + len(valid) + len(test) == 20


def test_random_splitter_valid_size_blocks_split():
    X = _X(20)
    sp = RandomSplitter(train_size=0.6, valid_size=0.2, test_size=0.2)
    with pytest.raises(ConfigurationError):
        next(sp.split(X))


def test_random_splitter_result_json_roundtrip():
    X = _X(20)
    sp = RandomSplitter(random_state=7)
    [result] = sp.split_result(X)
    s = result.to_json()
    from chemsplit.base import SplitResult

    back = SplitResult.from_json(s)
    assert np.array_equal(back.train, result.train)
    assert back.params["random_state"] == result.metadata["resolved_seed"]


def test_random_splitter_get_params_includes_base_params():
    sp = RandomSplitter(train_size=0.7, test_size=0.3, random_state=5)
    params = sp.get_params()
    assert params["shuffle"] is True
    assert params["train_size"] == 0.7
    assert params["test_size"] == 0.3
    assert params["random_state"] == 5


def test_random_splitter_generator_random_state_is_json_safe():
    X = _X(20)
    sp = RandomSplitter(random_state=np.random.default_rng(3))
    [result] = sp.split_result(X)
    # must not raise (I4 invariant already checked at construction); random_state resolved to int
    assert isinstance(result.params["random_state"], int)


# --------------------------------------------------------------------------- StratifiedRandomSplitter


def test_stratified_classification_balances_classes():
    rng = np.random.default_rng(0)
    n = 100
    y = np.array([0] * 80 + [1] * 20)
    X = _X(n)
    sp = StratifiedRandomSplitter(train_size=0.8, test_size=0.2, random_state=0)
    [result] = sp.split_result(X, y=y)
    train_pos = int(np.sum(y[result.train] == 1))
    test_pos = int(np.sum(y[result.test] == 1))
    assert train_pos + test_pos == 20
    assert test_pos >= 3  # roughly 20% of the 20 positives, largest-remainder apportioned


def test_stratified_requires_labels():
    X = _X(20)
    sp = StratifiedRandomSplitter()
    with pytest.raises(LabelError):
        sp.split_result(X, y=None)


def test_stratified_regression_quantile_binning():
    n = 200
    y = np.linspace(0, 100, n)
    X = _X(n)
    sp = StratifiedRandomSplitter(n_bins=5, train_size=0.8, test_size=0.2, random_state=1)
    [result] = sp.split_result(X, y=y)
    assert result.metadata["task_resolved"] == "auto"
    assert result.metadata["n_strata"] <= 5


def test_stratified_multitask_error_by_default():
    n = 20
    y = np.zeros((n, 3))
    X = _X(n)
    sp = StratifiedRandomSplitter()
    with pytest.raises(LabelError):
        sp.split_result(X, y=y)


def test_stratified_multitask_sum_labels():
    n = 40
    y = (np.random.default_rng(0).random((n, 3)) > 0.5).astype(float)
    X = _X(n)
    sp = StratifiedRandomSplitter(multitask="sum_labels", train_size=0.8, test_size=0.2, random_state=0)
    [result] = sp.split_result(X, y=y)
    assert len(result.train) + len(result.test) + len(result.valid) + len(result.discard) == n


def test_stratified_determinism():
    n = 60
    y = np.array([0, 1] * 30)
    X = _X(n)
    sp1 = StratifiedRandomSplitter(random_state=9)
    sp2 = StratifiedRandomSplitter(random_state=9)
    [r1] = sp1.split_result(X, y=y)
    [r2] = sp2.split_result(X, y=y)
    assert np.array_equal(r1.train, r2.train)
    assert np.array_equal(r1.test, r2.test)


# --------------------------------------------------------------------------- KFoldSplitter


def test_kfold_yields_n_splits_disjoint_test_sets():
    X = _X(20)
    sp = KFoldSplitter(n_splits=5, shuffle=True, random_state=0)
    folds = list(sp.split(X))
    assert len(folds) == 5
    all_test = np.concatenate([te for _, te in folds])
    assert sorted(all_test.tolist()) == list(range(20))  # every record tested exactly once


def test_kfold_rejects_size_params():
    with pytest.raises(ConfigurationError):
        KFoldSplitter(n_splits=5, train_size=0.8)


def test_kfold_shuffle_false_matches_contiguous_chunks():
    X = _X(10)
    sp = KFoldSplitter(n_splits=5, shuffle=False)
    folds = list(sp.split(X))
    test0 = folds[0][1].tolist()
    assert test0 == [0, 1]


def test_kfold_loo_sets_n_splits_to_n():
    X = _X(6)
    sp = KFoldSplitter(n_splits="loo", shuffle=False)
    assert sp.get_n_splits(X) == 6
    folds = list(sp.split(X))
    assert len(folds) == 6
    assert all(len(te) == 1 for _, te in folds)


def test_kfold_n_splits_exceeds_n_raises():
    X = _X(3)
    sp = KFoldSplitter(n_splits=5)
    with pytest.raises(ParameterError):
        list(sp.split(X))


# --------------------------------------------------------------------------- MonteCarloSplitter


def test_montecarlo_yields_n_splits_possibly_overlapping():
    X = _X(20)
    sp = MonteCarloSplitter(n_splits=5, train_size=0.7, test_size=0.3, random_state=0)
    folds = list(sp.split(X))
    assert len(folds) == 5
    # overlap across repeats is expected/allowed (not asserted away)


def test_montecarlo_determinism():
    X = _X(20)
    sp1 = MonteCarloSplitter(n_splits=3, random_state=1)
    sp2 = MonteCarloSplitter(n_splits=3, random_state=1)
    f1 = list(sp1.split(X))
    f2 = list(sp2.split(X))
    for (tr1, te1), (tr2, te2) in zip(f1, f2):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


# --------------------------------------------------------------------------- PredefinedSplitter


def test_predefined_sequence_assignment():
    X = _X(6)
    assignment = ["train", "train", "test", "test", "valid", "discard"]
    sp = PredefinedSplitter(assignment=assignment)
    [result] = sp.split_result(X)
    assert result.train.tolist() == [0, 1]
    assert result.test.tolist() == [2, 3]
    assert result.valid.tolist() == [4]
    assert result.discard.tolist() == [5]


def test_predefined_mapping_assignment_unlisted_goes_to_discard():
    X = _X(5)
    sp = PredefinedSplitter(assignment={"train": [0, 1], "test": [2]})
    [result] = sp.split_result(X)
    assert result.discard.tolist() == [3, 4]


def test_predefined_requires_exactly_one_of_assignment_fold_column():
    with pytest.raises(ParameterError):
        PredefinedSplitter()
    with pytest.raises(ParameterError):
        PredefinedSplitter(assignment=["train", "test"], fold_column=[0, 1])


def test_predefined_rejects_size_params():
    with pytest.raises(ConfigurationError):
        PredefinedSplitter(assignment=["train", "test"], train_size=0.5)


def test_predefined_invalid_partition_value_raises():
    X = _X(3)
    sp = PredefinedSplitter(assignment=["train", "bogus", "test"])
    with pytest.raises(ParameterError):
        sp.split_result(X)


def test_predefined_empty_train_raises():
    X = _X(3)
    sp = PredefinedSplitter(assignment=["discard", "discard", "test"])
    with pytest.raises(EmptyPartitionError):
        sp.split_result(X)


def test_predefined_fold_column():
    X = _X(6)
    fold_column = [0, 0, 1, 1, -1, -1]
    sp = PredefinedSplitter(fold_column=fold_column)
    assert sp.get_n_splits() == 2
    results = sp.split_result(X)
    assert len(results) == 2
    fold0 = results[0]
    assert fold0.test.tolist() == [0, 1]
    assert set(fold0.train.tolist()) == {2, 3, 4, 5}
