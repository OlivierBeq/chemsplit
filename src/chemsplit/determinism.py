"""Deterministic seeding, tie-breaking and rounding primitives.

Every seeded draw elsewhere in the library goes through :func:`seed_for`, every argmax or
argmin through :func:`argmax_tiebreak` or :func:`argmin_tiebreak`, and every size-fraction
rounding through :func:`floor_round`. A lint test greps the source tree for bare
``np.argmax(``/``np.argmin(`` outside this file.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

__all__ = [
    "EPS",
    "SeedBundle",
    "seed_for",
    "make_seed_bundle",
    "seeded_python_random",
    "argmax_tiebreak",
    "argmin_tiebreak",
    "first_argmax_2d",
    "first_ge_2d",
    "masked_argmax",
    "row_argmin",
    "stable_sort",
    "floor_round",
]

#: Fixed float-comparison tolerance.
EPS = 1e-9

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """The root :class:`numpy.random.SeedSequence` a splitter derives every purpose-specific
    generator from.

    :param root: the seed sequence every derived generator spawns from.
    :param resolved_seed: the entropy actually used, so a run seeded from OS entropy can be
        replayed.
    """

    root: np.random.SeedSequence
    resolved_seed: int | None = None
    """When ``random_state=None`` was resolved from OS entropy, the drawn entropy is recorded here
    so the run can be replayed (surfaced in ``SplitResult.metadata["resolved_seed"]``)."""


def seed_for(bundle: SeedBundle, purpose: str, k: int = 0) -> np.random.Generator:
    """Derive a purpose- and fold-specific PCG64 generator.

    Each splitter's ``purpose`` strings are fixed and documented in its Notes section; adding,
    removing or renaming one is a breaking change.

    :param bundle: the seed bundle to derive from.
    :param purpose: names the stream, so two different draws in one splitter stay independent.
    :param k: fold or repeat index within that stream.
    :return: a fresh generator, identical for identical arguments.
    """
    entropy_tag = int.from_bytes(
        hashlib.blake2b(purpose.encode("utf-8"), digest_size=8).digest(), "big"
    )
    child = np.random.SeedSequence(
        entropy=bundle.root.entropy,
        spawn_key=bundle.root.spawn_key + (entropy_tag, k),
    )
    return np.random.Generator(np.random.PCG64(child))


def make_seed_bundle(random_state: int | np.random.Generator | None) -> SeedBundle:
    """Build a :class:`SeedBundle` from a splitter's ``random_state``.

    ``None`` draws OS entropy and records it, so the run can be replayed. An int is used
    directly. A generator's state is *not* consumed: the root is derived from its PCG64 state
    word, so results survive that generator being advanced elsewhere.

    :param random_state: an int, a :class:`numpy.random.Generator`, or ``None``.
    :raises TypeError: if ``random_state`` is any other type.
    :return: the bundle, with ``resolved_seed`` always populated.
    """
    if random_state is None:
        root = np.random.SeedSequence()
        return SeedBundle(root=root, resolved_seed=int(root.entropy))
    if isinstance(random_state, np.random.Generator):
        state = random_state.bit_generator.state["state"]["state"]
        resolved = int(state)
        root = np.random.SeedSequence(entropy=resolved)
        return SeedBundle(root=root, resolved_seed=resolved)
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise TypeError(
            "random_state must be an int, a numpy.random.Generator, or None; "
            f"got {type(random_state).__name__}"
        )
    resolved = int(random_state)
    root = np.random.SeedSequence(entropy=resolved)
    return SeedBundle(root=root, resolved_seed=resolved)


def seeded_python_random(bundle: SeedBundle, purpose: str, k: int = 0) -> random.Random:
    """A seeded, isolated :class:`random.Random` for libraries that want the stdlib API.

    ``deap`` is the case in point. Going through this function keeps the module-level
    ``random`` state untouched.

    :param bundle: the seed bundle to derive from.
    :param purpose: names the stream.
    :param k: fold or repeat index within that stream.
    :return: a freshly seeded generator.
    """
    seed_int = int(seed_for(bundle, purpose, k).integers(0, 2**31 - 1))
    return random.Random(seed_int)


def argmax_tiebreak(func: Callable[[_T], float], items: Iterable[_T]) -> _T:
    """Return the element maximising ``func``, breaking ties by first occurrence.

    Comparison is strict, so the first element reaching the best score wins.

    :param func: the score to maximise.
    :param items: candidates, in tie-breaking priority order, usually ascending index.
    :raises ValueError: if ``items`` is empty.
    :return: the winning element.
    """
    best_item: _T | None = None
    best_score: float | None = None
    for item in items:
        score = func(item)
        if best_score is None or score > best_score:
            best_score = score
            best_item = item
    if best_item is None:
        raise ValueError("argmax_tiebreak() called with an empty iterable")
    return best_item


def argmin_tiebreak(func: Callable[[_T], float], items: Iterable[_T]) -> _T:
    """Return the element minimising ``func``, breaking ties by first occurrence.

    :param func: the score to minimise.
    :param items: candidates, in tie-breaking priority order.
    :raises ValueError: if ``items`` is empty.
    :return: the winning element.
    """
    best_item: _T | None = None
    best_score: float | None = None
    for item in items:
        score = func(item)
        if best_score is None or score < best_score:
            best_score = score
            best_item = item
    if best_item is None:
        raise ValueError("argmin_tiebreak() called with an empty iterable")
    return best_item


def masked_argmax(scores: np.ndarray, excluded: np.ndarray) -> int:
    """Vectorised :func:`argmax_tiebreak` over a score vector with some entries excluded.

    The greedy pick loops rebuilt an O(n) Python candidate list per pick; this is the same
    reduction in one numpy pass. Equivalent because :func:`numpy.argmax` returns the first
    occurrence of the maximum -- the ties-to-smallest-index rule -- and excluded entries are
    pushed to ``-inf``.

    :param scores: the scores to maximise, shape ``(n,)``. Must contain no NaN.
    :param excluded: boolean mask of entries that may not be selected, shape ``(n,)``.
    :raises ValueError: if the shapes disagree, or every entry is excluded.
    :return: the index of the maximum among the non-excluded entries, ties to the smallest index.
    """
    scores = np.asarray(scores)
    excluded = np.asarray(excluded, dtype=bool)
    if scores.shape != excluded.shape or scores.ndim != 1:
        raise ValueError(
            f"masked_argmax() needs two 1-D arrays of equal shape, got {scores.shape} "
            f"and {excluded.shape}"
        )
    if bool(excluded.all()):
        raise ValueError("masked_argmax() called with every entry excluded")
    # numpy's argmax returns the first occurrence of the maximum
    return int(np.argmax(np.where(excluded, -np.inf, scores)))


def first_ge_2d(M: np.ndarray, value: float) -> tuple[int, int]:
    """Row-major-first ``(i, j)`` where ``M[i, j] >= value``.

    The tie-break the greedy seed-pair searches use: take the *lexicographically smallest* pair
    within a tolerance of the maximum, which is not the same as the first pair attaining the
    maximum exactly -- a slightly smaller but earlier pair wins. Entries outside the region of
    interest must already be masked to a value below ``value``.

    :param M: a 2-D array with no NaN.
    :param value: the inclusive lower bound.
    :raises ValueError: if ``M`` is not 2-D, or no entry reaches ``value``.
    :return: the row and column of the first qualifying entry, scanning row-major.
    """
    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError(f"first_ge_2d() needs a 2-D array, got shape {M.shape}")
    hits = np.flatnonzero(M.ravel() >= value)  # row-major order == lexicographic (i, j)
    if hits.size == 0:
        raise ValueError("first_ge_2d(): no entry reaches the given value")
    return int(hits[0]) // M.shape[1], int(hits[0]) % M.shape[1]


def first_argmax_2d(M: np.ndarray) -> tuple[int, int]:
    """Row-major-first ``(i, j)`` of the maximum of a 2-D array.

    Replaces a Python scan over all n(n-1)/2 pairs that also materialised ``np.triu_indices``
    (8n^2 bytes).

    :param M: a 2-D array with at least one element, and no NaN.
    :raises ValueError: if ``M`` is not 2-D or is empty.
    :return: the row and column of the maximum, scanning in row-major order.
    """
    M = np.asarray(M)
    if M.ndim != 2 or M.size == 0:
        raise ValueError(f"first_argmax_2d() needs a non-empty 2-D array, got shape {M.shape}")
    # argmax on the flattened view returns the first occurrence, i.e. row-major-first
    flat = int(np.argmax(M))
    return flat // M.shape[1], flat % M.shape[1]


def row_argmin(M: np.ndarray) -> np.ndarray:
    """Vectorised row-wise :func:`argmin_tiebreak`, for nearest-neighbour lookups.

    A Python-level call per row is too slow on large matrices.

    :param M: a 2-D array with at least one column, and no NaN.
    :raises ValueError: if ``M`` is not 2-D, or has no columns.
    :return: per row, the column index of the minimum, ties to the smallest index.
    """
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[1] == 0:
        raise ValueError(
            f"row_argmin() needs a 2-D array with at least one column, got shape {M.shape}"
        )
    # numpy's argmin returns the first occurrence of the minimum
    return np.argmin(M, axis=1).astype(np.int64)


def stable_sort(
    seq: Sequence[_T], key: Callable[[_T], object], desc: bool = False
) -> list[_T]:
    """Sort so that equal keys keep their original relative order, including when descending.

    ``sorted(..., reverse=True)`` is documented to be stable in this sense, so this is a thin
    wrapper, used in place of ad hoc ``sorted`` calls.

    :param seq: the sequence to sort.
    :param key: the sort key.
    :param desc: sort descending.
    :return: a new sorted list.
    """
    return sorted(seq, key=key, reverse=desc)


def floor_round(x: float) -> int:
    """Round half up on non-negative reals.

    Not Python's ``round()``, which rounds half to even, and not ``numpy.round``. Neither is
    used for size computation anywhere in the library.

    :param x: a non-negative real.
    :return: ``math.floor(x + 0.5)``.
    """
    return math.floor(x + 0.5)
