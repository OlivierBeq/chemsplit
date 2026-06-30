import numpy as np
import pytest
import scipy.sparse as sp

from chemsplit import metrics


def _random_binary(n, d, density=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, d)) < density).astype(np.uint8)
    return X


class TestZeroVectorConventions:
    def test_tanimoto_both_zero_is_distance_zero(self):
        X = np.zeros((2, 16), dtype=np.uint8)
        D = metrics.pairwise_distances(X, metric="tanimoto")
        assert D[0, 1] == pytest.approx(0.0)

    def test_dice_both_zero_is_distance_zero(self):
        X = np.zeros((2, 16), dtype=np.uint8)
        D = metrics.pairwise_distances(X, metric="dice")
        assert D[0, 1] == pytest.approx(0.0)

    def test_cosine_both_zero_is_distance_zero(self):
        X = np.zeros((2, 16), dtype=np.float64)
        D = metrics.pairwise_distances(X, metric="cosine")
        assert D[0, 1] == pytest.approx(0.0)

    def test_cosine_one_zero_is_distance_one(self):
        X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        D = metrics.pairwise_distances(X, metric="cosine")
        assert D[0, 1] == pytest.approx(1.0)


class TestDiagonalAndSymmetry:
    @pytest.mark.parametrize("metric", ["tanimoto", "dice", "cosine", "euclidean", "manhattan"])
    def test_diagonal_is_exactly_zero(self, metric):
        X = _random_binary(20, 32, seed=1) if metric in ("tanimoto", "dice") else np.random.default_rng(2).random((20, 8))
        D = metrics.pairwise_distances(X, metric=metric)
        assert np.all(np.diagonal(D) == 0.0)

    @pytest.mark.parametrize("metric", ["tanimoto", "dice", "cosine", "euclidean", "manhattan"])
    def test_symmetric(self, metric):
        X = _random_binary(15, 32, seed=3) if metric in ("tanimoto", "dice") else np.random.default_rng(4).random((15, 6))
        D = metrics.pairwise_distances(X, metric=metric)
        assert np.array_equal(D, D.T)


class TestBlockedVsUnblocked:
    @pytest.mark.parametrize("metric", ["tanimoto", "dice"])
    @pytest.mark.parametrize("block_size", [4, 7, 2048])
    def test_block_size_invariant(self, metric, block_size):
        X = _random_binary(37, 64, seed=5)
        D_ref = metrics.pairwise_distances(X, metric=metric, block_size=2048)
        D_blk = metrics.pairwise_distances(X, metric=metric, block_size=block_size)
        assert np.array_equal(D_ref, D_blk)

    def test_asymmetric_matches_symmetric_slice(self):
        X = _random_binary(10, 32, seed=6)
        D_full = metrics.pairwise_distances(X, metric="tanimoto")
        D_cross = metrics.pairwise_distances(X[:4], X, metric="tanimoto")
        assert np.array_equal(D_full[:4], D_cross)


class TestSparseVsDenseAgree:
    def test_sparse_and_dense_agree(self):
        X = _random_binary(12, 40, seed=7)
        Xs = sp.csr_matrix(X)
        D_dense = metrics.pairwise_distances(X, metric="tanimoto")
        D_sparse = metrics.pairwise_distances(Xs, metric="tanimoto")
        assert np.allclose(D_dense, D_sparse, atol=1e-6)


class TestCondensedAndNN:
    def test_condensed_matches_pdist_ordering(self):
        X = _random_binary(6, 16, seed=8)
        D = metrics.pairwise_distances(X, metric="tanimoto")
        cond = metrics.condensed_distances(X, metric="tanimoto")
        expected = []
        n = X.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                expected.append(D[i, j])
        assert np.allclose(cond, np.array(expected, dtype=np.float32))

    def test_nn_distance_matches_brute_force(self):
        Q = _random_binary(5, 20, seed=9)
        R = _random_binary(30, 20, seed=10)
        d, idx = metrics.nn_distance(Q, R, metric="tanimoto", return_index=True)
        D = metrics.pairwise_distances(Q, R, metric="tanimoto")
        assert np.allclose(d, D.min(axis=1), atol=1e-6)
        assert np.array_equal(idx, D.argmin(axis=1))


class TestKnownValues:
    def test_tanimoto_known_value(self):
        a = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        b = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        X = np.vstack([a, b])
        D = metrics.pairwise_distances(X, metric="tanimoto")
        # intersection=1, union=3 -> similarity 1/3 -> distance 2/3
        assert D[0, 1] == pytest.approx(2.0 / 3.0, abs=1e-6)

    def test_manhattan_known_value(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = metrics.pairwise_distances(X, metric="manhattan")
        assert D[0, 1] == pytest.approx(7.0)

    def test_euclidean_known_value(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = metrics.pairwise_distances(X, metric="euclidean")
        assert D[0, 1] == pytest.approx(5.0)


@pytest.mark.slow
def test_throughput_sanity():
    X = _random_binary(1500, 2048, density=0.02, seed=11)
    import time

    t0 = time.perf_counter()
    metrics.pairwise_distances(X, metric="tanimoto")
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0
