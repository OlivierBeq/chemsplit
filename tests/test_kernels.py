"""The compiled similarity kernel must match a plain-numpy reference exactly.

The oracle lives here, not in the package: an AND-popcount loop written for clarity, defining the
expected semantics independently of the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit import _kernels
from chemsplit.metrics import _popcount_u64, _to_packed_words


def _reference(
    wa: np.ndarray, pa: np.ndarray, wb: np.ndarray, pb: np.ndarray
) -> np.ndarray:
    """Tanimoto distance block via a plain numpy AND-popcount loop over the 64-bit words."""
    inter = np.zeros((wa.shape[0], wb.shape[0]), dtype=np.int64)
    for w in range(wa.shape[1]):
        inter += _popcount_u64(np.bitwise_and(wa[:, w][:, None], wb[:, w][None,:]))
    union = pa[:, None].astype(np.int64) + pb[None,:].astype(np.int64) - inter
    sim = np.ones(inter.shape, dtype=np.float64)  # both-zero convention: similarity 1.0
    nonzero = union > 0
    sim[nonzero] = inter[nonzero] / union[nonzero]
    return 1.0 - sim


def _fingerprints(n: int, n_bits: int, density: float, seed: int = 0) -> np.ndarray:
    """Binary fingerprints, including the awkward cases deliberately."""
    rng = np.random.default_rng(seed)
    X = (rng.random((n, n_bits)) < density).astype(np.uint8)
    if n >= 5:
        X[0] = 0  # all-zero: exercises the both-zero convention
        X[1] = 0  # a second all-zero, so a zero/zero pair exists
        X[2] = X[3]  # duplicate: distance 0.0
        X[4] = 1  # all-ones
    return X


@pytest.mark.parametrize(
    ("n", "n_bits", "density"),
    [
        (64, 2048, 0.03),  # ECFP4-like sparsity
        (64, 167, 0.25),  # MACCS width, not a multiple of 64
        (40, 2048, 0.60),  # dense
        (33, 65, 0.50),  # just over one word, so padding is exercised
        (16, 64, 0.50),  # exactly one word
    ],
)
def test_kernel_matches_reference(n: int, n_bits: int, density: float) -> None:
    X = _fingerprints(n, n_bits, density)
    words, pops = _to_packed_words(X)
    got = _kernels.tanimoto_block(words, pops, words, pops)
    assert np.array_equal(got, _reference(words, pops, words, pops))
    assert got.dtype == np.float64


def test_kernel_matches_reference_on_rectangular_blocks() -> None:
    """pairwise_distances calls the kernel on off-diagonal blocks, which are not square."""
    X = _fingerprints(50, 2048, 0.05, seed=1)
    Y = _fingerprints(23, 2048, 0.05, seed=2)
    wx, px = _to_packed_words(X)
    wy, py = _to_packed_words(Y)
    assert np.array_equal(
        _kernels.tanimoto_block(wx, px, wy, py), _reference(wx, px, wy, py)
    )


@pytest.mark.parametrize("n_threads", [1, 2, 8])
def test_kernel_is_thread_count_invariant(n_threads: int) -> None:
    """Each (i, j) accumulates independently, so the thread count cannot change a value."""
    import numba

    X = _fingerprints(256, 2048, 0.05, seed=4)
    words, pops = _to_packed_words(X)
    expected = _reference(words, pops, words, pops)
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(min(n_threads, numba.config.NUMBA_NUM_THREADS))
        got = _kernels.tanimoto_block(words, pops, words, pops)
    finally:
        numba.set_num_threads(previous)
    assert np.array_equal(got, expected)


def test_blocked_neighbour_lists_match_the_dense_path() -> None:
    """The matrix-free threshold-graph route must agree with slicing the dense matrix."""
    from chemsplit import clustering as _clustering
    from chemsplit.clustering import _neighbor_lists, butina, butina_from_neighbors
    from chemsplit.metrics import pairwise_distances

    X = _fingerprints(300, 2048, 0.08, seed=7)
    cutoff = 0.65
    dense = pairwise_distances(X, metric="tanimoto")
    expected = _neighbor_lists(dense, cutoff)

    n = X.shape[0]
    got: list[np.ndarray] = []
    for start in range(0, n, 64):  # a block size that does not divide n evenly
        stop = min(start + 64, n)
        block = pairwise_distances(X[start:stop], X, metric="tanimoto")
        for row_offset in range(stop - start):
            i = start + row_offset
            idx = np.nonzero(block[row_offset] <= cutoff + _clustering.EPS)[0]
            got.append(np.sort(idx[idx != i]))

    for a, b in zip(expected, got, strict=True):
        assert np.array_equal(a, b)
    for reorder in (False, True):
        assert butina_from_neighbors(got, n, reorder=reorder) == butina(
            dense, cutoff, reorder=reorder
        )

