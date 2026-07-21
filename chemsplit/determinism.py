"""Deterministic seeding, tie-breaking, and rounding primitives.

Every other module in chemsplit MUST route every seeded random draw through :func:`seed_for`,
every argmax/argmin through :func:`argmax_tiebreak`/:func:`argmin_tiebreak`, and every
size-fraction rounding through :func:`floor_round`.

A lint test greps the whole ``chemsplit/`` source tree for bare ``np.argmax(``/``np.argmin(``
outside this file and fails on any hit.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
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
    "stable_sort",
    "floor_round",
]

#: fixed float-comparison tolerance.
EPS = 1e-9

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """Wraps the root :class:`numpy.random.SeedSequence` a splitter's ``_run`` derives every
    purpose-specific generator from."""

    root: np.random.SeedSequence
    resolved_seed: int | None = None
    """When ``random_state=None`` was resolved from OS entropy, the drawn entropy is recorded here
    so the run can be replayed (surfaced in ``SplitResult.metadata["resolved_seed"]``)."""


def seed_for(bundle: SeedBundle, purpose: str, k: int = 0) -> np.random.Generator:
    """Derive a purpose- and fold-specific PCG64 generator from ``bundle``.

    ``purpose`` strings are fixed per splitter and documented in each splitter's Determinism
    block; adding, removing, or renaming a ``purpose`` is a breaking (MAJOR) change.
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
    """Build a :class:`SeedBundle` from a splitter's ``random_state`` constructor argument.

    - ``None`` — a fresh :class:`numpy.random.SeedSequence` drawing OS entropy; the drawn entropy
      is recorded on the bundle (``resolved_seed``) so the run can be replayed.
    - ``int`` — ``SeedSequence(entropy=random_state)``.
    - ``Generator`` — the generator's current state is **not** consumed. Instead the bundle's root
      is derived from the generator's internal PCG64 state word, so results stay reproducible even
      when a shared generator has been advanced elsewhere by other code.
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
    """A seeded, isolated :class:`random.Random` instance for third-party libraries (``deap``)
    that need Python's stdlib ``random`` API rather than a numpy Generator ("seeded via
    ``random.Random(int(...))``; the module-level ``random`` MUST NOT be touched").

    Never call ``random.seed(...)`` at module scope anywhere in chemsplit — always go through this
    function so global state is never touched.
    """
    seed_int = int(seed_for(bundle, purpose, k).integers(0, 2**31 - 1))
    return random.Random(seed_int)


def argmax_tiebreak(func: Callable[[_T], float], items: Iterable[_T]) -> _T:
    """Return the element of ``items`` maximising ``func``, breaking ties by first occurrence.

    ``items`` MUST be supplied in the tie-breaking priority order (typically ascending record or
    group index — "the winner is the one with the smallest record index"). Comparison is
    strict (``>``), so the first-seen element achieving the best score wins any tie.
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
    """Mirror of :func:`argmax_tiebreak`: returns the element minimising ``func``, ties broken by
    first occurrence (strict ``<`` comparison)."""
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


def stable_sort(
    seq: Sequence[_T], key: Callable[[_T], object], desc: bool = False
) -> list[_T]:
    """A stable sort where equal keys retain ascending-index (original relative) order, even when
    ``desc=True``.

    Python's ``sorted(..., reverse=True)`` is documented to be stable in this sense (it doesn't
    reverse the relative order of equal elements), so this is a thin wrapper — used everywhere in
    chemsplit instead of ad hoc ``sorted(..., reverse=...)`` calls.
    """
    return sorted(seq, key=key, reverse=desc)


def floor_round(x: float) -> int:
    """Round-half-up on non-negative reals: ``math.floor(x + 0.5)``.

    This is deliberately NOT Python's built-in ``round()`` (banker's rounding) nor
    ``numpy.round``; neither may be used anywhere in chemsplit for size computation.
    """
    return math.floor(x + 0.5)
