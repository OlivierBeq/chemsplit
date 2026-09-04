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

    def test_nn_distance_without_return_index(self):
        Q = _random_binary(5, 20, seed=9)
        R = _random_binary(30, 20, seed=10)
        d = metrics.nn_distance(Q, R, metric="tanimoto")
        D = metrics.pairwise_distances(Q, R, metric="tanimoto")
        assert np.allclose(d, D.min(axis=1), atol=1e-6)


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


class TestTanimotoCount:
    def test_known_value(self):
        a = np.array([[2.0, 0.0, 4.0]])
        b = np.array([[1.0, 0.0, 3.0]])
        X = np.vstack([a, b])
        D = metrics.pairwise_distances(X, metric="tanimoto_count")
        # mins sum = 1+0+3 = 4, maxs sum = 2+0+4 = 6 -> similarity 4/6 -> distance 1/3
        assert D[0, 1] == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_diagonal_and_symmetry(self):
        rng = np.random.default_rng(12)
        X = rng.integers(0, 5, size=(10, 8)).astype(np.float64)
        D = metrics.pairwise_distances(X, metric="tanimoto_count")
        assert np.all(np.diagonal(D) == 0.0)
        assert np.array_equal(D, D.T)

    def test_both_zero_rows_distance_zero(self):
        X = np.zeros((2, 5), dtype=np.float64)
        D = metrics.pairwise_distances(X, metric="tanimoto_count")
        assert D[0, 1] == pytest.approx(0.0)

    def test_asymmetric_query_reference(self):
        Q = np.array([[1.0, 2.0, 0.0]])
        R = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
        D = metrics.pairwise_distances(Q, R, metric="tanimoto_count")
        assert D.shape == (1, 2)


class TestUnknownMetric:
    def test_raises_value_error(self):
        X = np.zeros((2, 4))
        with pytest.raises(ValueError, match="unknown metric"):
            metrics.pairwise_distances(X, metric="not_a_real_metric")


class TestIsBoundedMetric:
    @pytest.mark.parametrize("metric", ["tanimoto", "dice", "cosine", "tanimoto_count"])
    def test_bounded_metrics(self, metric):
        assert metrics.is_bounded_metric(metric) is True

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
    def test_unbounded_metrics(self, metric):
        assert metrics.is_bounded_metric(metric) is False


class TestTanimotoSimilarityMatrix:
    def test_is_one_minus_distance(self):
        X = _random_binary(8, 16, seed=13)
        S = metrics.tanimoto_similarity_matrix(X)
        D = metrics.pairwise_distances(X, metric="tanimoto")
        assert np.allclose(S, 1.0 - D)
        assert np.all(np.diagonal(S) == 1.0)


class TestNonUint8Input:
    def test_boolean_array_coerced_to_binary(self):
        X = np.array([[True, False, True], [False, False, True]])
        D = metrics.pairwise_distances(X, metric="tanimoto")
        assert D.shape == (2, 2)
        assert D[0, 0] == 0.0


class TestAsymmetricNonPackedMetrics:
    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan"])
    def test_query_reference_shape(self, metric):
        Q = np.random.default_rng(14).random((4, 5))
        R = np.random.default_rng(15).random((7, 5))
        D = metrics.pairwise_distances(Q, R, metric=metric)
        assert D.shape == (4, 7)

    def test_asymmetric_tanimoto_count_matches_symmetric_slice(self):
        X = np.random.default_rng(16).integers(0, 4, size=(9, 6)).astype(np.float64)
        D_full = metrics.pairwise_distances(X, metric="tanimoto_count")
        D_cross = metrics.pairwise_distances(X[:3], X, metric="tanimoto_count")
        assert np.array_equal(D_full[:3], D_cross)

    def test_tanimoto_count_symmetric_multiblock_matches_single_block(self):
        X = np.random.default_rng(17).integers(0, 4, size=(9, 6)).astype(np.float64)
        D_ref = metrics.pairwise_distances(X, metric="tanimoto_count", block_size=2048)
        D_blk = metrics.pairwise_distances(X, metric="tanimoto_count", block_size=3)
        assert np.array_equal(D_ref, D_blk)

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan"])
    def test_symmetric_multiblock_matches_single_block(self, metric):
        X = np.random.default_rng(18).random((9, 5))
        D_ref = metrics.pairwise_distances(X, metric=metric, block_size=2048)
        D_blk = metrics.pairwise_distances(X, metric=metric, block_size=3)
        assert np.allclose(D_ref, D_blk)


class TestPopcountFallback:
    """The lookup-table popcount fallback (for numpy versions lacking ``np.bitwise_count``) is
    only reachable on such an older numpy; forced here via monkeypatch since this environment's
    numpy has it, to keep the fallback path itself under test."""

    def test_fallback_matches_bitwise_count(self, monkeypatch):
        table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        monkeypatch.setattr(metrics, "_HAS_BITWISE_COUNT", False)
        monkeypatch.setattr(metrics, "_POPCOUNT_TABLE", table, raising=False)
        arr = np.array([0x1, 0xFF, 0xFFFFFFFFFFFFFFFF, 0], dtype=np.uint64)
        got = metrics._popcount_u64(arr)
        want = np.array([bin(int(v)).count("1") for v in arr], dtype=np.int64)
        assert np.array_equal(got, want)


class TestIsBinaryLike:
    """``_is_binary_like`` has no call sites elsewhere in chemsplit (verified: only its own
    definition matches a repo-wide grep) -- it appears to be unused/dead code, kept here perhaps
    for a future caller. Tested directly for correctness rather than left completely unverified."""

    def test_all_zero_one_is_binary(self):
        assert metrics._is_binary_like(np.array([0, 1, 1, 0], dtype=np.uint8)) is True

    def test_non_binary_values_are_not_binary(self):
        assert metrics._is_binary_like(np.array([0, 1, 2], dtype=np.uint8)) is False

    def test_empty_is_binary(self):
        assert metrics._is_binary_like(np.array([], dtype=np.uint8)) is True

    def test_sparse_input(self):
        assert metrics._is_binary_like(sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.uint8))) is True


@pytest.mark.slow
def test_throughput_sanity():
    X = _random_binary(1500, 2048, density=0.02, seed=11)
    import time

    t0 = time.perf_counter()
    metrics.pairwise_distances(X, metric="tanimoto")
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0
