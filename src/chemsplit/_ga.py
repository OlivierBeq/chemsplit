"""Shared GA primitives for SIMPDSplitter and AVESplitter.

Only `mutate_bits` is shared (identical RNG-consumption order in both). Population/crossover/
repair/selection differ genuinely and stay separate.
"""

from __future__ import annotations

import random

import numpy as np

__all__ = ["mutate_bits"]


def mutate_bits(mask: np.ndarray, indpb: float, rng: random.Random) -> np.ndarray:
    """Flip each position of a boolean mask independently with probability `indpb`.

    Consumes exactly `len(mask)` calls to `rng.random()`, in order.
    """
    flip = np.array([rng.random() < indpb for _ in range(len(mask))])
    out = mask.copy()
    out[flip] = ~out[flip]
    return out
