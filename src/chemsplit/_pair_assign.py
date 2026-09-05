"""Shared sqrt(f) two-axis independent group assignment."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from chemsplit.base import _ResolvedSizes, assign_groups
from chemsplit.types import IndexArray

__all__ = ["assign_pair_groups"]


def _axis_train_test(
    labels: IndexArray,
    n: int,
    f_target: float,
    mode_assignment: Literal["greedy_desc", "balanced", "random"],
    rng: np.random.Generator,
) -> tuple[set[int], set[int]]:
    """Run one axis's independent train/test split at fraction ``f_target``.

    This is a two-bucket call into :func:`chemsplit.base.assign_groups`, so there is no discard
    bucket at the axis level; records are only discarded later, where the two axes disagree.

    :param labels: dense group labels for this axis.
    :param n: the record count.
    :param f_target: the test fraction to aim for on this axis.
    :param mode_assignment: how groups are handed to buckets.
    :param rng: generator for ``mode_assignment="random"``.
    :return: the train and test index sets. Every labelled record lands in exactly one.
    """
    if n == 0:
        return set(), set()
    n_test = int(round(f_target * n))
    n_test = max(0, min(n, n_test))
    # assign_groups omits zero-capacity buckets, so clamp to keep both keys present
    n_test = max(1, min(n - 1, n_test)) if 0 < n_test < n else n_test
    sizes = _ResolvedSizes(n_train=n - n_test, n_valid=0, n_test=n_test)
    result = assign_groups(labels, sizes, mode_assignment, rng)
    train = set(int(i) for i in result.get("train", np.asarray([], dtype=np.int64)))
    test = set(int(i) for i in result.get("test", np.asarray([], dtype=np.int64)))
    return train, test


def assign_pair_groups(
    labels_a: IndexArray,
    labels_b: IndexArray,
    sizes: _ResolvedSizes,
    rng: np.random.Generator,
    *,
    mode: Literal["both_novel", "either_novel"] = "both_novel",
    group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
) -> dict[str, IndexArray]:
    """Assign records to train/valid/test by intersecting two independent group axes.

    ``labels_a`` and ``labels_b`` are dense ``0..g-1`` arrays of the same length, aligned by
    record. Each axis is thinned independently at ``sqrt(fraction)`` so that the intersection
    lands near the requested overall fraction, then the two axes' masks are combined:

    - ``"both_novel"`` intersects both the test masks and the train masks, and discards every
      record the two axes disagree on. This is the strict "novel on both axes" semantics.
    - ``"either_novel"`` unions the test masks, so novelty on one axis is enough, while train
      stays the conservative intersection. The rest is discarded.

    The three-way case extends the same idea, with each axis targeting ``sqrt(f_valid)`` and
    ``sqrt(f_test)`` for its own buckets. That extension follows the two-way logic but only the
    two-way path is covered by hand-checked tests.

    Nothing is rebalanced after intersecting, so the realised sizes are reported as they come
    out, which for a doubly constrained assignment will not match the request exactly.

    :param labels_a: dense group labels on the first axis.
    :param labels_b: dense group labels on the second axis.
    :param sizes: the overall train/valid/test targets.
    :param rng: generator for ``group_assignment="random"``.
    :param mode: require novelty on both axes, or on either one.
    :param group_assignment: how groups are handed to buckets within each axis; see
        :func:`chemsplit.base.assign_groups`.
    :return: a mapping from partition name to member indices, plus ``"discard"``.
    """
    n = len(labels_a)
    if len(labels_b) != n:
        raise ValueError("labels_a and labels_b must have the same length")

    f_test = sizes.n_test / n if n else 0.0
    f_valid = sizes.n_valid / n if n else 0.0
    sqrt_test = math.sqrt(f_test)
    sqrt_valid = math.sqrt(f_valid)

    if sizes.n_valid == 0:
        train_a, test_a = _axis_train_test(labels_a, n, sqrt_test, group_assignment, rng)
        train_b, test_b = _axis_train_test(labels_b, n, sqrt_test, group_assignment, rng)
        valid_a = valid_b = set()
    else:
        # per axis: valid takes sqrt(f_valid), test sqrt(f_test), the rest is train
        train_a, valid_a, test_a = _axis_three_way(
            labels_a, n, sqrt_valid, sqrt_test, group_assignment, rng
        )
        train_b, valid_b, test_b = _axis_three_way(
            labels_b, n, sqrt_valid, sqrt_test, group_assignment, rng
        )

    if mode == "both_novel":
        test = test_a & test_b
        valid = valid_a & valid_b
        train = train_a & train_b
    elif mode == "either_novel":
        test = test_a | test_b
        valid = (valid_a | valid_b) - test
        train = train_a & train_b
    else:
        raise ValueError(f"unknown mode {mode!r}")

    assigned = train | valid | test
    discard = set(range(n)) - assigned

    return {
        "train": np.sort(np.asarray(sorted(train), dtype=np.int64)),
        "valid": np.sort(np.asarray(sorted(valid), dtype=np.int64)),
        "test": np.sort(np.asarray(sorted(test), dtype=np.int64)),
        "discard": np.sort(np.asarray(sorted(discard), dtype=np.int64)),
    }


def _axis_three_way(
    labels: IndexArray,
    n: int,
    sqrt_valid: float,
    sqrt_test: float,
    mode_assignment: Literal["greedy_desc", "balanced", "random"],
    rng: np.random.Generator,
) -> tuple[set[int], set[int], set[int]]:
    """Single-axis three-bucket split at (sqrt_valid, sqrt_test) fractions; remainder is train."""
    if n == 0:
        return set(), set(), set()
    n_valid = max(0, min(n, int(round(sqrt_valid * n))))
    n_test = max(0, min(n - n_valid, int(round(sqrt_test * n))))
    sizes = _ResolvedSizes(n_train=n - n_valid - n_test, n_valid=n_valid, n_test=n_test)
    result = assign_groups(labels, sizes, mode_assignment, rng)
    train = set(int(i) for i in result.get("train", np.asarray([], dtype=np.int64)))
    valid = set(int(i) for i in result.get("valid", np.asarray([], dtype=np.int64)))
    test = set(int(i) for i in result.get("test", np.asarray([], dtype=np.int64)))
    return train, valid, test
