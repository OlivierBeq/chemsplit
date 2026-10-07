"""Distance/similarity kernels and pairwise-distance machinery.

- Blocked computation MUST produce the same values as unblocked -- no metric may use a running
  mean or any other order-dependent reduction.
- ``n_jobs`` MUST NOT change any returned value.
- Symmetry is enforced by computing the upper triangle and mirroring, never by averaging.

Tanimoto and Dice on binary fingerprints run on packed bits, through ``numpy.bitwise_count``
where it exists and a per-byte lookup table otherwise.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from chemsplit.types import FeatureMatrix

MetricName = Literal[
    "tanimoto", "dice", "cosine", "euclidean", "manhattan", "tanimoto_count", "mahalanobis"
]

_BOUNDED_METRICS = frozenset({"tanimoto", "dice", "cosine", "tanimoto_count"})

__all__ = [
    "condensed_distances",
    "is_bounded_metric",
    "nn_distance",
    "pairwise_distances",
    "tanimoto_similarity_matrix",
]


def is_bounded_metric(metric: str) -> bool:
    """Report whether a metric's values are guaranteed to lie in ``[0, 1]``.

    :param metric: a metric name.
    :return: ``True`` for a bounded metric."""
    return metric in _BOUNDED_METRICS


_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")

if not _HAS_BITWISE_COUNT:
    _POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_u64(arr: np.ndarray) -> np.ndarray:
    """Elementwise popcount of a uint64 array, returned as int64 (safe for summation)."""
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(arr).astype(np.int64)
    # fallback: view as uint8, look up per-byte popcount, sum across the 8 bytes
    as_bytes = arr.view(np.uint8).reshape(*arr.shape, 8)
    return _POPCOUNT_TABLE[as_bytes].sum(axis=-1, dtype=np.int64)


def _pack_rows_to_u64(X: np.ndarray) -> np.ndarray:
    """Pack a dense (n, n_bits) 0/1 uint8 matrix into (n, ceil(n_bits/64)) uint64 words."""
    n, n_bits = X.shape
    n_words = -(-n_bits // 64)
    pad = n_words * 64 - n_bits
    if pad:
        X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    # bit order is internal; only popcount and AND are used downstream
    packed_bytes = np.packbits(X, axis=-1, bitorder="little")
    words = packed_bytes.reshape(n, n_words * 8).view(np.uint64)
    return words


def _to_packed_words(Xf: FeatureMatrix) -> tuple[np.ndarray, np.ndarray]:
    """Return (packed_u64_words (n, n_words), popcounts (n,)) for a binary feature matrix."""
    if sp.issparse(Xf):
        Xf = Xf.toarray()
    Xf = np.asarray(Xf)
    if Xf.dtype != np.uint8:
        Xf = (Xf != 0).astype(np.uint8)
    words = _pack_rows_to_u64(Xf)
    popcounts = _popcount_u64(words).sum(axis=1)
    return words, popcounts


def _is_binary_like(Xf: FeatureMatrix) -> bool:
    if sp.issparse(Xf):
        data = Xf.data
    else:
        data = np.asarray(Xf)
    if data.size == 0:
        return True
    sample = data if data.size <= 4096 else data.flat[:4096]
    return bool(np.all((sample == 0) | (sample == 1)))


def _block_ranges(n: int, block_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, n, block_size):
        yield start, min(start + block_size, n)


def _tanimoto_block(
    words_a: np.ndarray, pop_a: np.ndarray, words_b: np.ndarray, pop_b: np.ndarray
) -> np.ndarray:
    """float64 Tanimoto distance block, shape (len(a), len(b)).

    Delegates to the compiled kernel in :mod:`chemsplit._kernels`.
    """
    from chemsplit import _kernels

    return _kernels.tanimoto_block(words_a, pop_a, words_b, pop_b)


def _dice_block(
    words_a: np.ndarray, pop_a: np.ndarray, words_b: np.ndarray, pop_b: np.ndarray
) -> np.ndarray:
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
    # min(x,y) = (x+y-|x-y|)/2, so an L1 block gives mins and maxs without an (na,nb,d) array
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    l1 = cdist(a, b, metric="cityblock")
    sum_a = a.sum(axis=-1)
    sum_b = b.sum(axis=-1)
    total = sum_a[:, None] + sum_b[None,:]
    mins = 0.5 * (total - l1)
    maxs = total - mins
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


def _to_dense_f64(F: FeatureMatrix) -> np.ndarray:
    return np.asarray(F.toarray() if sp.issparse(F) else F, dtype=np.float64)


def _mahalanobis_whiten(*mats: FeatureMatrix) -> list[np.ndarray]:
    """Map every matrix into a space where Euclidean distance equals Mahalanobis distance.

    The inverse covariance ``VI`` is the pseudo-inverse of the covariance of *all* rows passed
    (so ``d(a, b)`` does not depend on which argument a row came from) and is factored once as
    ``VI = L @ L.T``; returning ``F @ L`` for each input keeps blocked computation identical to
    unblocked. The pseudo-inverse makes a singular covariance (``n < d``, constant columns,
    binary fingerprints) well-defined: directions of zero variance contribute zero distance.
    """
    dense = [_to_dense_f64(F) for F in mats]
    stack = np.vstack(dense)
    d = stack.shape[1]
    if stack.shape[0] < 2 or d == 0:
        return [np.zeros((F.shape[0], 0), dtype=np.float64) for F in dense]
    C = np.atleast_2d(np.cov(stack, rowvar=False))
    VI = np.linalg.pinv(C, hermitian=True)
    w, U = np.linalg.eigh(VI)
    L = U * np.sqrt(np.clip(w, 0.0, None))[None, :]
    return [F @ L for F in dense]


def _minkowski_block(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return cdist(a, b, metric="cityblock" if p == 1 else "euclidean")


def pairwise_distances(
    Xf: FeatureMatrix,
    Yf: FeatureMatrix | None = None,
    metric: MetricName = "tanimoto",
    n_jobs: int = 1,
    block_size: int = 2048,
) -> np.ndarray:
    """Compute the full pairwise distance matrix.

    ``metric="mahalanobis"`` estimates its covariance from the rows passed in this call, so the
    same pair of records can come out at different distances in calls over different record
    sets.

    :param Xf: the left-hand feature matrix.
    :param Yf: the right-hand feature matrix, or ``None`` to use ``Xf``.
    :param metric: the metric name.
    :param n_jobs: worker count. The binary-metric kernels are already multi-threaded
        internally (BLAS, or numba's ``prange``), so this is advisory; results never depend
        on it.
    :param block_size: rows per block.
    :raises UnknownMetricError: if ``metric`` is not recognised.
    :return: an ``(n, m)`` float32 matrix.
    """
    if metric == "mahalanobis":
        if Yf is None:
            (Xf,) = _mahalanobis_whiten(Xf)
        else:
            Xf, Yf = _mahalanobis_whiten(Xf, Yf)
        metric = "euclidean"
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
                    p = 1 if metric == "manhattan" else 2
                    block = _minkowski_block(Xd[i0:i1], Yd[j0:j1], p=p)
                out[i0:i1, j0:j1] = block
                if symmetric and j0 != i0:
                    out[j0:j1, i0:i1] = block.T
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    if symmetric:
        np.fill_diagonal(out, 0.0)

    return out.astype(np.float32)


def condensed_distances(
    Xf: FeatureMatrix, metric: MetricName = "tanimoto", n_jobs: int = 1
) -> np.ndarray:
    """Compute the condensed upper-triangle distance vector.

    Ordering matches ``scipy.spatial.distance.pdist``: ``i < j``, ``i`` ascending then ``j``.

    :param Xf: the feature matrix.
    :param metric: the metric name.
    :param n_jobs: worker count. Results never depend on it.
    :raises UnknownMetricError: if ``metric`` is not recognised.
    :return: a float32 vector of length ``n * (n - 1) / 2``.
    """
    D = pairwise_distances(Xf, metric=metric, n_jobs=n_jobs)
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return D[iu].astype(np.float32)


def nn_distance(
    Q: FeatureMatrix,
    R: FeatureMatrix,
    metric: MetricName = "tanimoto",
    n_jobs: int = 1,
    return_index: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Find each row of ``Q``'s nearest neighbour in ``R``.

    Blocked over ``R``, so no full dense ``Q x R`` matrix is materialised.

    :param Q: the query feature matrix.
    :param R: the reference feature matrix.
    :param metric: the metric name.
    :param n_jobs: worker count. Results never depend on it.
    :param return_index: also return the index of each nearest neighbour.
    :raises UnknownMetricError: if ``metric`` is not recognised.
    :return: the distances, or ``(distances, indices)`` when ``return_index`` is set.
    """
    if metric == "mahalanobis":
        # whiten once over Q and all of R, so every block shares one covariance
        Q, R = _mahalanobis_whiten(Q, R)
        metric = "euclidean"
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


def tanimoto_similarity_matrix(Xf: FeatureMatrix) -> np.ndarray:
    """Compute the pairwise Tanimoto similarity matrix.

    :param Xf: the feature matrix.
    :return: ``1 - pairwise_distances(Xf, metric="tanimoto")``."""
    return 1.0 - pairwise_distances(Xf, metric="tanimoto")
