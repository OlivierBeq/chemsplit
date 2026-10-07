"""Compiled binary-fingerprint similarity kernel.

A Tanimoto numerator is an intersection popcount and its denominator a union count, so the only
rounding happens in one final float64 division. Each ``(i, j)`` is an independent integer
accumulation with no cross-thread reduction, making the result independent of the thread count.

The numba kernels live in :mod:`chemsplit._kernels_numba`, imported lazily: ``import chemsplit``
must stay cheap and rdkit/numba-free. Blocks are dispatched to a serial or a threaded kernel by
cell count; the two are bit-identical, so this trades only speed. Narrow fetches lose more to
thread start-up than they gain, and the serial kernel also loads from the numba cache ~0.3 s
faster, which dominates a small split.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["tanimoto_block"]

_CACHE: dict[str, Any] = {}


def _compiled() -> Any:
    """Return the compiled kernel, importing numba on first use.

    :raises MissingDependencyError: if numba is unavailable.
    :return: the compiled kernel.
    """
    if "fn" in _CACHE:
        return _CACHE["fn"]
    try:
        from chemsplit._kernels_numba import tanimoto_block as fn
    except ImportError as exc:  # pragma: no cover - numba is a hard dependency
        from chemsplit.exceptions import MissingDependencyError

        raise MissingDependencyError(
            "numba is required for fingerprint similarity; reinstall chemsplit"
        ) from exc
    _CACHE["fn"] = fn
    return fn


def tanimoto_block(
    words_a: np.ndarray,
    pop_a: np.ndarray,
    words_b: np.ndarray,
    pop_b: np.ndarray,
) -> np.ndarray:
    """Tanimoto **distance** block, shape ``(len(words_a), len(words_b))``, as float64.

    :param words_a: left-hand packed fingerprints.
    :param pop_a: left-hand popcounts.
    :param words_b: right-hand packed fingerprints.
    :param pop_b: right-hand popcounts.
    :return: the distance block.
    """
    out = np.empty((words_a.shape[0], words_b.shape[0]), dtype=np.float64)
    return _compiled()(words_a, pop_a.astype(np.int64), words_b, pop_b.astype(np.int64), out)
