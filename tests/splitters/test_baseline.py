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
    for (tr1, te1), (tr2, te2) in zip(f1, f2, strict=True):
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


# --------------------------------------------------------------------------- coverage additions


def test_stratified_multitask_first():
    X = _X(40)
    y = np.column_stack([np.arange(40) % 4, np.arange(40) % 2])
    sp = StratifiedRandomSplitter(multitask="first", train_size=0.7, test_size=0.3, random_state=0)
    result = sp.split_result(X, y)[0]
    assert result.n_records == 40


def test_stratified_multitask_iterative():
    X = _X(40)
    y = np.column_stack([np.arange(40) % 4, np.arange(40) % 2]).astype(np.float64)
    sp = StratifiedRandomSplitter(multitask="iterative", train_size=0.7, test_size=0.3, random_state=0)
    result = sp.split_result(X, y)[0]
    assert result.n_records == 40


def test_stratified_multitask_invalid():
    with pytest.raises(ParameterError):
        StratifiedRandomSplitter(multitask="bogus")


def test_stratified_y_with_inf_raises():
    X = _X(10)
    y = np.array([1.0, 2.0, float("inf"), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    sp = StratifiedRandomSplitter(train_size=0.7, test_size=0.3, random_state=0)
    with pytest.raises(LabelError):
        sp.split_result(X, y)


def test_stratified_regression_uniform_binning():
    X = _X(50)
    y = np.random.default_rng(0).standard_normal(50)
    sp = StratifiedRandomSplitter(binning="uniform", train_size=0.7, test_size=0.3, random_state=0)
    result = sp.split_result(X, y)[0]
    assert result.n_records == 50


def test_stratified_regression_kmeans_binning():
    X = _X(50)
    y = np.random.default_rng(0).standard_normal(50)
    sp = StratifiedRandomSplitter(binning="kmeans", n_bins=4, train_size=0.7, test_size=0.3, random_state=0)
    result = sp.split_result(X, y)[0]
    assert result.n_records == 50


def test_stratified_binning_invalid():
    with pytest.raises(ParameterError):
        StratifiedRandomSplitter(binning="bogus")


def test_stratified_on_small_stratum_raise():
    X = _X(10)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # class 1 has only 1 member
    sp = StratifiedRandomSplitter(on_small_stratum="raise", train_size=0.7, test_size=0.3, random_state=0)
    with pytest.raises(LabelError):
        sp.split_result(X, y)


def test_stratified_on_small_stratum_ignore():
    X = _X(10)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    sp = StratifiedRandomSplitter(on_small_stratum="ignore", train_size=0.7, test_size=0.3, random_state=0)
    result = sp.split_result(X, y)[0]
    assert result.n_records == 10


def test_stratified_on_small_stratum_invalid():
    with pytest.raises(ParameterError):
        StratifiedRandomSplitter(on_small_stratum="bogus")


def test_stratified_task_invalid():
    with pytest.raises(ParameterError):
        StratifiedRandomSplitter(task="bogus")


def test_stratified_n_bins_and_min_per_stratum_validation():
    with pytest.raises(ParameterError):
        StratifiedRandomSplitter(n_bins=1)
    with pytest.raises(ParameterError):
        StratifiedRandomSplitter(min_per_stratum=0)


# --------------------------------------------------------------------------- KFoldSplitter


def test_kfold_stratified():
    X = _X(40)
    y = np.arange(40) % 4
    sp = KFoldSplitter(n_splits=4, stratify=True, random_state=0)
    results = sp.split_result(X, y)
    assert len(results) == 4
    for r in results:
        assert r.train.size + r.test.size == 40


def test_kfold_n_splits_type_validation():
    with pytest.raises(ParameterError):
        KFoldSplitter(n_splits="bogus")
    with pytest.raises(ParameterError):
        KFoldSplitter(n_splits=1)


def test_kfold_shuffle_and_stratify_type_validation():
    with pytest.raises(ParameterError):
        KFoldSplitter(shuffle="yes")
    with pytest.raises(ParameterError):
        KFoldSplitter(stratify="yes")


def test_kfold_stratify_kwargs_without_stratify_raises():
    with pytest.raises(ParameterError):
        KFoldSplitter(stratify=False, stratify_kwargs={"n_bins": 5})


def test_kfold_get_n_splits_loo_without_x():
    sp = KFoldSplitter(n_splits="loo")
    assert sp.get_n_splits() == 1


def test_kfold_get_n_splits_loo_with_x():
    X = _X(15)
    sp = KFoldSplitter(n_splits="loo")
    assert sp.get_n_splits(X) == 15


# --------------------------------------------------------------------------- MonteCarloSplitter


def test_montecarlo_stratified():
    X = _X(40)
    y = np.arange(40) % 4
    sp = MonteCarloSplitter(n_splits=3, stratify=True, train_size=0.7, test_size=0.3, random_state=0)
    results = sp.split_result(X, y)
    assert len(results) == 3


def test_montecarlo_n_splits_validation():
    with pytest.raises(ParameterError):
        MonteCarloSplitter(n_splits=0)


# --------------------------------------------------------------------------- PredefinedSplitter


def test_predefined_mapping_index_assigned_twice_raises():
    X = _X(5)
    sp = PredefinedSplitter(assignment={"train": [0, 1, 2], "test": [2, 3]})
    with pytest.raises(ParameterError):
        sp.split_result(X)


def test_predefined_mapping_unknown_partition_name_raises():
    X = _X(5)
    sp = PredefinedSplitter(assignment={"bogus": [0, 1]})
    with pytest.raises(ParameterError):
        sp.split_result(X)


def test_predefined_sequence_wrong_length_raises():
    X = _X(5)
    sp = PredefinedSplitter(assignment=["train", "test"])
    with pytest.raises(ParameterError):
        sp.split_result(X)


def test_predefined_fold_column_wrong_length_raises():
    X = _X(5)
    sp = PredefinedSplitter(fold_column=[0, 1])
    with pytest.raises(ParameterError):
        sp.split_result(X)


def test_predefined_fold_column_no_nonnegative_ids_raises():
    X = _X(5)
    sp = PredefinedSplitter(fold_column=[-1, -1, -1, -1, -1])
    with pytest.raises(ParameterError):
        sp.split_result(X)


def test_predefined_fold_column_empty_fold_raises():
    X = _X(4)
    sp = PredefinedSplitter(fold_column=[0, 0, 0, 0])
    # every record has fold id 0 -> that fold's "test" is everything, "train" is empty
    with pytest.raises(EmptyPartitionError):
        sp.split_result(X)
