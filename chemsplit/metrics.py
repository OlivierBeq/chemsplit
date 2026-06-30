"""Distance/similarity kernels and pairwise-distance machinery.

- Blocked computation MUST produce the same values as unblocked — no metric may use a running
  mean or any other order-dependent reduction.
- ``n_jobs`` MUST NOT change any returned value.
- Symmetry is enforced by computing the upper triangle and mirroring, never by averaging.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp

MetricName = Literal["tanimoto", "dice", "cosine", "euclidean", "manhattan", "tanimoto_count"]

_BOUNDED_METRICS = frozenset({"tanimoto", "dice", "cosine", "tanimoto_count"})

__all__ = [
    "condensed_distances",
    "is_bounded_metric",
    "nn_distance",
    "pairwise_distances",
    "tanimoto_similarity_matrix",
]


def is_bounded_metric(metric: str) -> bool:
    """Return True if ``metric`` is guaranteed to lie in ``[0, 1]``."""
    return metric in _BOUNDED_METRICS


# --------------------------------------------------------------------------------------
# Popcount machinery for packed-bit Tanimoto/Dice on binary fingerprints.
# --------------------------------------------------------------------------------------

_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")

if not _HAS_BITWISE_COUNT:
    _POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_u64(arr: np.ndarray) -> np.ndarray:
    """Elementwise popcount of a uint64 array, returned as int64 (safe for summation)."""
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(arr).astype(np.int64)
    # Fallback: view as uint8 bytes, look up per-byte popcount, sum across the 8 bytes.
    as_bytes = arr.view(np.uint8).reshape(*arr.shape, 8)
    return _POPCOUNT_TABLE[as_bytes].sum(axis=-1, dtype=np.int64)


def _pack_rows_to_u64(X: np.ndarray) -> np.ndarray:
    """Pack a dense (n, n_bits) 0/1 uint8 matrix into (n, ceil(n_bits/64)) uint64 words."""
    n, n_bits = X.shape
    n_words = -(-n_bits // 64)
    pad = n_words * 64 - n_bits
    if pad:
        X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    bits = X.astype(np.uint8).reshape(n, n_words, 64)
    # Little-endian bit order within each 64-bit word; order is internal and consistent, so it
    # doesn't matter for popcount-based similarity as long as it's applied identically everywhere.
    weights = (np.uint64(1) << np.arange(64, dtype=np.uint64))
    words = (bits.astype(np.uint64) * weights).sum(axis=-1, dtype=np.uint64)
    return words


def _to_packed_words(Xf) -> tuple[np.ndarray, np.ndarray]:
    """Return (packed_u64_words (n, n_words), popcounts (n,)) for a binary feature matrix."""
    if sp.issparse(Xf):
        Xf = Xf.toarray()
    Xf = np.asarray(Xf)
    if Xf.dtype != np.uint8:
        Xf = (Xf != 0).astype(np.uint8)
    words = _pack_rows_to_u64(Xf)
    popcounts = _popcount_u64(words).sum(axis=1)
    return words, popcounts


def _is_binary_like(Xf) -> bool:
    if sp.issparse(Xf):
        data = Xf.data
    else:
        data = np.asarray(Xf)
    if data.size == 0:
        return True
    sample = data if data.size <= 4096 else data.reshape(-1)[:4096]
    return bool(np.all((sample == 0) | (sample == 1)))


# --------------------------------------------------------------------------------------
# Block-wise pairwise distance computation.
# --------------------------------------------------------------------------------------


def _block_ranges(n: int, block_size: int):
    for start in range(0, n, block_size):
        yield start, min(start + block_size, n)


def _tanimoto_block(words_a, pop_a, words_b, pop_b) -> np.ndarray:
    """float64 Tanimoto distance block, shape (len(a), len(b))."""
    na, nb = words_a.shape[0], words_b.shape[0]
    inter = np.zeros((na, nb), dtype=np.int64)
    for w in range(words_a.shape[1]):
        anded = np.bitwise_and(words_a[:, w][:, None], words_b[:, w][None,:])
        inter += _popcount_u64(anded)
    union = pop_a[:, None].astype(np.int64) + pop_b[None,:].astype(np.int64) - inter
    sim = np.ones((na, nb), dtype=np.float64)  # both-zero convention: similarity 1.0
    nonzero = union > 0
    sim[nonzero] = inter[nonzero] / union[nonzero]
    return 1.0 - sim


def _dice_block(words_a, pop_a, words_b, pop_b) -> np.ndarray:
    na, nb = words_a.shape[0], words_b.shape[0]
    inter = np.zeros((na, nb), dtype=np.int64)
    for w in range(words_a.shape[1]):
        anded = np.bitwise_and(words_a[:, w][:, None], words_b[:, w][None,:])
        inter += _popcount_u64(anded)
    denom = pop_a[:, None].astype(np.int64) + pop_b[None,:].astype(np.int64)
    sim = np.ones((na, nb), dtype=np.float64)
    nonzero = denom > 0
    sim[nonzero] = (2.0 * inter[nonzero]) / denom[nonzero]
    return 1.0 - sim


def _tanimoto_count_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mins = np.minimum(a[:, None,:], b[None,:,:]).sum(axis=-1)
    maxs = np.maximum(a[:, None,:], b[None,:,:]).sum(axis=-1)
    sim = np.ones(mins.shape, dtype=np.float64)
    nonzero = maxs > 0
    sim[nonzero] = mins[nonzero] / maxs[nonzero]
    return 1.0 - sim


def _cosine_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    dot = a @ b.T
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na[:, None] * nb[None,:]
    sim = np.zeros(denom.shape, dtype=np.float64)
    both_zero = (na[:, None] == 0.0) & (nb[None,:] == 0.0)
    sim[both_zero] = 1.0
    nonzero = denom > 0
    sim[nonzero] = dot[nonzero] / denom[nonzero]
    return 1.0 - sim


def _minkowski_block(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = np.abs(a[:, None,:] - b[None,:,:])
    if p == 1:
        return diff.sum(axis=-1)
    return np.sqrt((diff**2).sum(axis=-1))


def pairwise_distances(
    Xf,
    Yf=None,
    metric: MetricName = "tanimoto",
    n_jobs: int = 1,
    block_size: int = 2048,
) -> np.ndarray:
    """Full pairwise distance matrix, shape ``(n, m)``, dtype float32.

    ``n_jobs`` is accepted for API compatibility but the current implementation is single-process;
    accepting it now (without silently changing results) keeps the signature stable for a future
    parallel backend, per the determinism requirement that ``n_jobs`` must never change output.
    """
    symmetric = Yf is None
    Yf = Xf if symmetric else Yf
    n = Xf.shape[0]
    m = Yf.shape[0]
    out = np.empty((n, m), dtype=np.float64)

    if metric in ("tanimoto", "dice"):
        wa, pa = _to_packed_words(Xf)
        wb, pb = (wa, pa) if symmetric else _to_packed_words(Yf)
        block_fn = _tanimoto_block if metric == "tanimoto" else _dice_block
        if symmetric:
            for i0, i1 in _block_ranges(n, block_size):
                for j0, j1 in _block_ranges(m, block_size):
                    if j0 < i0:
                        continue
                    block = block_fn(wa[i0:i1], pa[i0:i1], wb[j0:j1], pb[j0:j1])
                    out[i0:i1, j0:j1] = block
                    if j0 != i0:
                        out[j0:j1, i0:i1] = block.T
        else:
            for i0, i1 in _block_ranges(n, block_size):
                for j0, j1 in _block_ranges(m, block_size):
                    out[i0:i1, j0:j1] = block_fn(wa[i0:i1], pa[i0:i1], wb[j0:j1], pb[j0:j1])
    elif metric == "tanimoto_count":
        Xd = Xf.toarray() if sp.issparse(Xf) else np.asarray(Xf)
        Yd = Xd if symmetric else (Yf.toarray() if sp.issparse(Yf) else np.asarray(Yf))
        for i0, i1 in _block_ranges(n, block_size):
            for j0, j1 in _block_ranges(m, block_size):
                if symmetric and j0 < i0:
                    continue
                block = _tanimoto_count_block(Xd[i0:i1], Yd[j0:j1])
                out[i0:i1, j0:j1] = block
                if symmetric and j0 != i0:
                    out[j0:j1, i0:i1] = block.T
    elif metric in ("cosine", "euclidean", "manhattan"):
        Xd = Xf.toarray() if sp.issparse(Xf) else np.asarray(Xf)
        Yd = Xd if symmetric else (Yf.toarray() if sp.issparse(Yf) else np.asarray(Yf))
        for i0, i1 in _block_ranges(n, block_size):
            for j0, j1 in _block_ranges(m, block_size):
                if symmetric and j0 < i0:
                    continue
                if metric == "cosine":
                    block = _cosine_block(Xd[i0:i1], Yd[j0:j1])
                else:
                    block = _minkowski_block(Xd[i0:i1], Yd[j0:j1], p=1 if metric == "manhattan" else 2)
                out[i0:i1, j0:j1] = block
                if symmetric and j0 != i0:
                    out[j0:j1, i0:i1] = block.T
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    if symmetric:
        np.fill_diagonal(out, 0.0)

    return out.astype(np.float32)


def condensed_distances(Xf, metric: MetricName = "tanimoto", n_jobs: int = 1) -> np.ndarray:
    """Condensed (upper-triangle, i<j, i ascending then j ascending) float32 distance vector.

    Matches ``scipy.spatial.distance.pdist``'s ordering convention.
    """
    D = pairwise_distances(Xf, metric=metric, n_jobs=n_jobs)
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return D[iu].astype(np.float32)


def nn_distance(
    Q,
    R,
    metric: MetricName = "tanimoto",
    n_jobs: int = 1,
    return_index: bool = False,
):
    """Nearest-neighbour distance (and optionally index) from each row of Q to R.

    Blocked over R to avoid materialising a full dense Q x R matrix at once when R is large.
    """
    n_q = Q.shape[0]
    best_dist = np.full(n_q, np.inf, dtype=np.float64)
    best_idx = np.full(n_q, -1, dtype=np.int64)
    block = 4096
    n_r = R.shape[0]
    for j0, j1 in _block_ranges(n_r, block):
        Rb = R[j0:j1]
        Dblock = pairwise_distances(Q, Rb, metric=metric, n_jobs=n_jobs).astype(np.float64)
        local_best = Dblock.argmin(axis=1)
        local_best_val = Dblock[np.arange(n_q), local_best]
        improve = local_best_val < best_dist
        best_dist[improve] = local_best_val[improve]
        best_idx[improve] = local_best[improve] + j0
    if return_index:
        return best_dist.astype(np.float32), best_idx
    return best_dist.astype(np.float32)


def tanimoto_similarity_matrix(Xf) -> np.ndarray:
    """Convenience wrapper: ``1 - pairwise_distances(Xf, metric="tanimoto")``."""
    return 1.0 - pairwise_distances(Xf, metric="tanimoto")
