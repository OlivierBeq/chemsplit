import numpy as np
import pytest

from chemsplit._multilabel import iterative_stratification, label_keys
from chemsplit.exceptions import LabelError


def _labels(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 5)) < [0.5, 0.3, 0.1, 0.05, 0.2]).astype(np.float64)


def test_exact_fold_sizes():
    fold = iterative_stratification(_labels(), [140, 30, 30], np.random.default_rng(1))
    assert np.bincount(fold).tolist() == [140, 30, 30]


def test_each_label_within_one_of_target():
    Y = _labels()
    sizes = [140, 30, 30]
    fold = iterative_stratification(Y, sizes, np.random.default_rng(1))
    for j in range(Y.shape[1]):
        positives = Y[:, j] == 1
        for m, size in enumerate(sizes):
            target = positives.sum() * size / len(Y)
            assert abs(int((positives & (fold == m)).sum()) - target) <= 1.0 + 1e-9


def test_hand_built_case():
    # label 1 (2 positives) is rarest and goes first: one record to each equal fold.
    Y = np.array([[1, 1], [1, 1], [1, 0], [1, 0], [0, 0], [0, 0]], dtype=float)
    fold = iterative_stratification(Y, [3, 3], np.random.default_rng(0))
    assert sorted(fold[[0, 1]].tolist()) == [0, 1]
    assert sorted(fold[[2, 3]].tolist()) == [0, 1]
    assert np.bincount(fold).tolist() == [3, 3]


def test_pairs_balance_co_occurrence_better():
    # labels 0 and 1 co-occur in 20 records and appear alone in 40 more each
    rows = [[1, 1]] * 20 + [[1, 0]] * 40 + [[0, 1]] * 40 + [[0, 0]] * 20
    Y = np.asarray(rows, dtype=float)
    both = (Y[:, 0] == 1) & (Y[:, 1] == 1)

    def pair_error(order):
        errors = []
        for seed in range(10):
            fold = iterative_stratification(Y, [60, 60], np.random.default_rng(seed), order=order)
            errors.append(abs(int((both & (fold == 0)).sum()) - 10))
        return sum(errors)

    assert pair_error(2) <= pair_error(1)
    assert pair_error(2) == 0


def test_nan_means_unlabelled():
    Y = _labels(60)
    Y[::7, 0] = np.nan
    fold = iterative_stratification(Y, [40, 20], np.random.default_rng(2))
    assert np.bincount(fold).tolist() == [40, 20]
    assert label_keys(np.array([[np.nan, 1.0]]), 1) == [((1,),)]


def test_seeded():
    Y = _labels()
    a = iterative_stratification(Y, [100, 100], np.random.default_rng(5))
    b = iterative_stratification(Y, [100, 100], np.random.default_rng(5))
    assert a.tolist() == b.tolist()


def test_single_label_records_use_the_label_in_pair_mode():
    assert label_keys(np.array([[0.0, 1.0, 0.0]]), 2) == [((1,),)]
    assert label_keys(np.array([[1.0, 1.0, 1.0]]), 2) == [((0, 1), (0, 2), (1, 2))]


@pytest.mark.parametrize(
    "bad", [np.array([[0.0, 2.0]]), np.array([0.0, 1.0]), np.array([["a", "b"]])]
)
def test_rejects_non_binary_or_non_matrix(bad):
    with pytest.raises(LabelError):
        iterative_stratification(
            bad, [len(bad)] if bad.ndim == 1 else [bad.shape[0]], np.random.default_rng(0)
        )


def test_sizes_must_cover_records():
    with pytest.raises(ValueError):
        iterative_stratification(_labels(10), [5, 4], np.random.default_rng(0))
