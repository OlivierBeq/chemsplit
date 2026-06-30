"""Tests for chemsplit._pair_assign.assign_pair_groups."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit._pair_assign import assign_pair_groups
from chemsplit.base import _ResolvedSizes


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_hand_constructed_both_novel():
    # 8 records, axis A groups: {0,1}->gA0, {2,3}->gA1, {4,5}->gA2, {6,7}->gA3 (4 groups of 2)
    labels_a = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    # axis B groups: {0,2,4,6}->gB0, {1,3,5,7}->gB1 (2 groups of 4)
    labels_b = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    sizes = _ResolvedSizes(n_train=6, n_valid=0, n_test=2)
    result = assign_pair_groups(labels_a, labels_b, sizes, _rng(0), mode="both_novel")
    train, test, valid, discard = (
        result["train"],
        result["test"],
        result["valid"],
        result["discard"],
    )
    assert len(valid) == 0
    # completeness and disjointness
    all_idx = np.sort(np.concatenate([train, test, valid, discard]))
    assert np.array_equal(all_idx, np.arange(8))
    assert len(set(train) & set(test)) == 0
    assert len(set(train) & set(discard)) == 0
    assert len(set(test) & set(discard)) == 0


def test_either_novel_test_is_superset_of_both_novel_test():
    rng_seed = 3
    labels_a = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    labels_b = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    sizes = _ResolvedSizes(n_train=7, n_valid=0, n_test=3)
    both = assign_pair_groups(labels_a, labels_b, sizes, _rng(rng_seed), mode="both_novel")
    either = assign_pair_groups(labels_a, labels_b, sizes, _rng(rng_seed), mode="either_novel")
    # either_novel's test set must be a superset of both_novel's test set (union vs intersection
    # of the same two axis-level test masks), since both calls use the same rng seed and hence the
    # same underlying axis assignments.
    assert set(both["test"]).issubset(set(either["test"]))
    # train is identical between the two modes by construction (train = axis_a_train ∩ axis_b_train
    # in both cases).
    assert np.array_equal(both["train"], either["train"])


def test_discard_only_nonempty_when_axes_disagree():
    # With identical axis labels (perfectly correlated groups), both axes agree on every record,
    # so both_novel and either_novel should produce empty (or near-empty) discard.
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
    sizes = _ResolvedSizes(n_train=8, n_valid=0, n_test=4)
    result = assign_pair_groups(labels, labels.copy(), sizes, _rng(1), mode="both_novel")
    # identical axes => axis_a_test == axis_b_test exactly => intersection == union => no disagreement
    assert len(result["discard"]) == 0


def test_determinism_same_seed_identical_output():
    rng_state = np.random.default_rng(42)
    labels_a = rng_state.integers(0, 5, size=30).astype(np.int64)
    labels_b = rng_state.integers(0, 6, size=30).astype(np.int64)
    sizes = _ResolvedSizes(n_train=20, n_valid=0, n_test=10)

    r1 = assign_pair_groups(labels_a, labels_b, sizes, np.random.default_rng(7), mode="both_novel")
    r2 = assign_pair_groups(labels_a, labels_b, sizes, np.random.default_rng(7), mode="both_novel")
    for key in ("train", "valid", "test", "discard"):
        assert np.array_equal(r1[key], r2[key])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_sqrt_sizing_monte_carlo_sanity(seed):
    """Intersecting two independent sqrt(f)-thinned axes should land the realized test fraction
    roughly near the target f_test. This is a combinatorial, group-based process on synthetic
    random data (not i.i.d. per-record coin flips), so we only check a loose 0.3x-3x band around
    the target rather than a tight tolerance -- the point is to catch a sizing-formula sign error
    (e.g. accidentally using f instead of sqrt(f)), not to validate exact statistics."""
    rng = np.random.default_rng(seed)
    n = 400
    # many small-ish groups on each axis so the balance-by-member-count assignment can actually
    # approximate a fractional cut.
    labels_a = rng.integers(0, 40, size=n).astype(np.int64)
    labels_b = rng.integers(0, 45, size=n).astype(np.int64)
    f_test_target = 0.09  # sqrt(0.09) = 0.3 per axis
    n_test = int(round(f_test_target * n))
    sizes = _ResolvedSizes(n_train=n - n_test, n_valid=0, n_test=n_test)

    result = assign_pair_groups(labels_a, labels_b, sizes, np.random.default_rng(seed + 1000), mode="both_novel")
    realized_frac = len(result["test"]) / n
    assert 0.3 * f_test_target <= realized_frac <= 3.0 * f_test_target, (
        f"realized test fraction {realized_frac} wildly off target {f_test_target} "
        "(sqrt-sizing formula likely wrong)"
    )


def test_three_way_completeness():
    rng = np.random.default_rng(11)
    n = 60
    labels_a = rng.integers(0, 15, size=n).astype(np.int64)
    labels_b = rng.integers(0, 18, size=n).astype(np.int64)
    sizes = _ResolvedSizes(n_train=36, n_valid=12, n_test=12)
    result = assign_pair_groups(labels_a, labels_b, sizes, np.random.default_rng(2), mode="both_novel")
    all_idx = np.sort(np.concatenate([result["train"], result["valid"], result["test"], result["discard"]]))
    assert np.array_equal(all_idx, np.arange(n))
    assert len(set(result["train"]) & set(result["valid"])) == 0
    assert len(set(result["valid"]) & set(result["test"])) == 0
