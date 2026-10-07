"""numba implementation of the Tanimoto block kernel.

Importing this module imports numba, so it is imported lazily by :mod:`chemsplit._kernels`;
``import chemsplit`` must stay cheap and rdkit/numba-free.

The kernels are module-level on purpose: ``cache=True`` cannot key a function defined in a
local scope, so a closure would silently recompile (~1.5 s) in every process.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange

__all__ = ["tanimoto_block"]


@njit(inline="always", cache=True)
def _popcount64(x: Any) -> Any:
    # SWAR popcount: numba has no intrinsic, and np.bitwise_count needs numpy>=2.0
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(parallel=True, cache=True)
def tanimoto_block(wa: Any, pa: Any, wb: Any, pb: Any, out: Any) -> Any:
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
