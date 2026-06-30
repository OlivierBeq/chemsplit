"""Shared √f two-axis independent group assignment.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from chemsplit.base import _ResolvedSizes, assign_groups
from chemsplit.types import IndexArray

if TYPE_CHECKING:
    pass

__all__ = ["assign_pair_groups"]


def _axis_train_test(
    labels: IndexArray,
    n: int,
    f_target: float,
    mode_assignment: Literal["greedy_desc", "balanced", "random"],
    rng: np.random.Generator,
) -> tuple[set[int], set[int]]:
    """Run a single axis's independent train/test split at fraction ``f_target``.

    Returns ``(train_indices, test_indices)`` as plain ``set[int]`` (every record with a defined
    label lands in exactly one of the two — this is a two-bucket-only call into
    :func:`chemsplit.base.assign_groups`, so there is no third/discard bucket at the axis level;
    discarding only happens later, from disagreement between the two axes).
    """
    if n == 0:
        return set(), set()
    n_test = int(round(f_target * n))
    n_test = max(0, min(n, n_test))
    # assign_groups drops any bucket whose capacity is 0 from its returned dict; guard both ends
    # so the caller always gets both keys back.
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
    """Assign records to train/valid/test by intersecting two independent, √f-sized group axes.

    ``labels_a``/``labels_b`` are dense ``0..g-1`` int64 group-label arrays, same length ``n``,
    aligned by record (record ``i`` belongs to group ``labels_a[i]`` on axis A and
    ``labels_b[i]`` on axis B). ``sizes`` gives the overall train/valid/test targets; each axis is
    independently thinned at ``sqrt(fraction)`` (see module docstring), then the two axes' masks
    are combined per ``mode``:

    - ``"both_novel"``: ``test = axis_a_test ∩ axis_b_test``, ``train = axis_a_train ∩
      axis_b_train``, everything else (the two axes disagree) → ``discard``. This is the strict
      "novel on both axes" semantics (cold-start pair / joint ligand+sequence splitting).
    - ``"either_novel"``: ``test = axis_a_test ∪ axis_b_test`` (novel on at least one axis counts
      as test), ``train = axis_a_train ∩ axis_b_train`` (train stays conservative — novel on
      neither axis), everything else → ``discard``.

    The three-way case (``sizes.n_valid > 0``) extends the same idea: each axis targets
    ``sqrt(f_valid)``/``sqrt(f_test)`` independently for its valid/test buckets (via two
    axis-level three-bucket ``assign_groups`` calls), and the three final buckets are formed by
    pairwise intersection/union of the matching bucket on each axis, in bucket order
    ``train, valid, test``. This extension follows the same logic as the train/test-only case but
    is not independently hand-verified — the train/test-only path is the one covered by
    hand-checked tests below.

    No rebalancing is attempted after intersection: realized sizes are reported as-is (typical,
    expected behavior for a doubly-constrained assignment — see the module docstring's note on why
    the intersection fraction is only approximately the target).
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
        # Three-bucket axis split: valid gets sqrt(f_valid), test gets sqrt(f_test), remainder
        # is train, per axis independently.
        train_a, valid_a, test_a = _axis_three_way(labels_a, n, sqrt_valid, sqrt_test, group_assignment, rng)
        train_b, valid_b, test_b = _axis_three_way(labels_b, n, sqrt_valid, sqrt_test, group_assignment, rng)

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
