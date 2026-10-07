"""Compiled binary-fingerprint similarity kernel.

A Tanimoto numerator is an intersection popcount and its denominator a union count, so the only
rounding happens in one final float64 division. Each ``(i, j)`` is an independent integer
accumulation with no cross-thread reduction, making the result independent of the thread count.

``numba`` is imported lazily: ``import chemsplit`` must stay cheap and rdkit/sklearn-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["tanimoto_block"]

_CACHE: dict[str, Any] = {}


def _compiled() -> Any:
    """Return the compiled kernel, building it on first use.

    ``cache=True`` so the JIT cost is paid once per installation, not once per process.

    :raises MissingDependencyError: if numba is unavailable.
    :return: the compiled kernel.
    """
    if "fn" in _CACHE:
        return _CACHE["fn"]
    try:
        from numba import njit, prange
    except ImportError as exc:  # pragma: no cover - numba is a hard dependency
        from chemsplit.exceptions import MissingDependencyError

        raise MissingDependencyError(
            "numba is required for fingerprint similarity; reinstall chemsplit"
        ) from exc

    @njit(inline="always")
    def _popcount64(x: Any) -> Any:
        # SWAR popcount: numba has no intrinsic, and np.bitwise_count needs numpy>=2.0
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + (
            (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
        )
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

    @njit(parallel=True, cache=True)
    def _block(wa: Any, pa: Any, wb: Any, pb: Any, out: Any) -> Any:
        na, n_words = wa.shape
        nb = wb.shape[0]
        for i in prange(na):
            for j in range(nb):
                inter = np.uint64(0)
                for w in range(n_words):
                    inter += _popcount64(wa[i, w] & wb[j, w])
                inter_i = np.int64(inter)
                union = pa[i] + pb[j] - inter_i
                # two all-zero fingerprints count as similarity 1.0, i.e. distance 0.0
                out[i, j] = 1.0 - (inter_i / union) if union > 0 else 0.0
        return out

    _CACHE["fn"] = _block
    return _block


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
    return _compiled()(
        words_a, pop_a.astype(np.int64), words_b, pop_b.astype(np.int64), out
    )
