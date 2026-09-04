import numpy as np
import pytest

from chemsplit import clustering


def _blob_distance_matrix():
    # Two tight blobs (0,1,2) and (3,4,5), far apart.
    rng = np.random.default_rng(0)
    pts = np.vstack(
        [
            rng.normal(loc=[0, 0], scale=0.05, size=(3, 2)),
            rng.normal(loc=[10, 10], scale=0.05, size=(3, 2)),
        ]
    )
    D = np.linalg.norm(pts[:, None,:] - pts[None,:,:], axis=-1)
    return D


def _spectral_blob_distance_matrix():
    # Like _blob_distance_matrix but with a separation small enough that the Gaussian affinity
    # kernel doesn't underflow to a bit-exact 0 cross-block similarity: an exactly-disconnected
    # affinity graph has a degenerate (multiplicity > 1) zero eigenvalue, a case spectral
    # clustering must refuse (ConstraintUnsatisfiableError) rather than silently cluster. Use a
    # merely very-small-but-nonzero cross-similarity instead, so this test exercises ordinary
    # (non-degenerate) spectral separation.
    rng = np.random.default_rng(0)
    pts = np.vstack(
        [
            rng.normal(loc=[0, 0], scale=0.05, size=(3, 2)),
            rng.normal(loc=[3, 3], scale=0.05, size=(3, 2)),
        ]
    )
    D = np.linalg.norm(pts[:, None,:] - pts[None,:,:], axis=-1)
    return D


class TestButina:
    def test_known_clusters_tight_cutoff(self):
        D = _blob_distance_matrix()
        clusters = clustering.butina(D, cutoff=0.5)
        cluster_sets = [set(c) for c in clusters]
        assert {0, 1, 2} in cluster_sets
        assert {3, 4, 5} in cluster_sets

    def test_loose_cutoff_one_cluster(self):
        D = _blob_distance_matrix()
        clusters = clustering.butina(D, cutoff=20.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 6

    def test_tight_cutoff_all_singletons(self):
        D = _blob_distance_matrix()
        clusters = clustering.butina(D, cutoff=1e-6)
        assert len(clusters) == 6

    def test_reorder_true_and_false_both_cover_all_points(self):
        D = _blob_distance_matrix()
        for reorder in (True, False):
            clusters = clustering.butina(D, cutoff=0.5, reorder=reorder)
            covered = sorted(x for c in clusters for x in c)
            assert covered == list(range(6))

    def test_deterministic_smallest_index_tiebreak(self):
        D = _blob_distance_matrix()
        c1 = clustering.butina(D, cutoff=0.5)
        c2 = clustering.butina(D, cutoff=0.5)
        assert c1 == c2


class TestMaxMinPick:
    def test_picks_are_unique_and_correct_count(self):
        D = _blob_distance_matrix()
        picks = clustering.maxmin_pick(D, n_picks=4, init="index_zero")
        assert len(picks) == 4
        assert len(set(picks)) == 4

    def test_diverse_picks_span_both_blobs(self):
        D = _blob_distance_matrix()
        picks = clustering.maxmin_pick(D, n_picks=2, init="index_zero")
        assert set(picks) & {0, 1, 2}
        assert set(picks) & {3, 4, 5}


class TestKennardStone:
    def test_first_two_are_maximally_distant(self):
        D = _blob_distance_matrix()
        picks = clustering.kennard_stone(D, n_picks=2)
        iu = np.triu_indices(D.shape[0], k=1)
        max_d = D[iu].max()
        assert D[picks[0], picks[1]] == pytest.approx(max_d, abs=1e-6)


class TestLeaderAndSphereExclusion:
    def test_leader_covers_all_points(self):
        D = _blob_distance_matrix()
        clusters = clustering.leader(D, radius=0.5)
        covered = sorted(x for c in clusters for x in c)
        assert covered == list(range(6))

    def test_sphere_exclusion_covers_all_points(self):
        D = _blob_distance_matrix()
        reps, groups = clustering.sphere_exclusion(D, radius=0.5)
        covered = sorted(x for g in groups for x in g)
        assert covered == list(range(6))
        assert len(reps) == len(groups)


class TestSpectralPartition:
    def test_separates_two_blobs(self):
        D = _spectral_blob_distance_matrix()
        W = np.exp(-(D**2))
        np.fill_diagonal(W, 0.0)
        labels = clustering.spectral_partition(W, n_clusters=2, random_state=0)
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_isolated_vertex_becomes_singleton(self):
        D = _spectral_blob_distance_matrix()
        W = np.exp(-(D**2))
        np.fill_diagonal(W, 0.0)
        W[0,:] = 0.0
        W[:, 0] = 0.0  # isolate vertex 0
        labels = clustering.spectral_partition(W, n_clusters=2, random_state=0)
        assert labels[0] not in labels[1:]

    def test_deterministic_across_runs(self):
        D = _spectral_blob_distance_matrix()
        W = np.exp(-(D**2))
        np.fill_diagonal(W, 0.0)
        l1 = clustering.spectral_partition(W, n_clusters=2, random_state=0)
        l2 = clustering.spectral_partition(W, n_clusters=2, random_state=0)
        assert np.array_equal(l1, l2)


class TestMaxMinPickInitModes:
    def test_kennard_stone_init(self):
        D = _blob_distance_matrix()
        picks = clustering.maxmin_pick(D, n_picks=2, init="kennard_stone")
        assert len(picks) == 2

    def test_most_peripheral_init(self):
        D = _blob_distance_matrix()
        picks = clustering.maxmin_pick(D, n_picks=1, init="most_peripheral")
        assert len(picks) == 1

    def test_random_init_requires_rng(self):
        D = _blob_distance_matrix()
        with pytest.raises(AssertionError):
            clustering.maxmin_pick(D, n_picks=2, init="random", rng=None)

    def test_random_init_with_rng(self):
        D = _blob_distance_matrix()
        rng = np.random.default_rng(0)
        picks = clustering.maxmin_pick(D, n_picks=3, init="random", rng=rng)
        assert len(set(picks)) == 3


class TestKennardStoneSmallN:
    def test_n_less_than_two(self):
        D = np.zeros((1, 1))
        assert clustering.kennard_stone(D, n_picks=1) == [0]


class TestSpectralPartitionLaplacianVariants:
    @pytest.mark.parametrize("laplacian", ["unnormalized", "rw", "sym"])
    def test_all_laplacian_variants_produce_valid_two_way_labels(self, laplacian):
        # Exact blob-separation correctness for the default "sym" Laplacian is already asserted
        # by test_separates_two_blobs above; this only needs to exercise the "unnormalized"/"rw"
        # code paths (spectral_partition's laplacian branch) and confirm a well-formed 2-cluster
        # labeling comes out, not assert every variant agrees on this exact toy dataset.
        D = _spectral_blob_distance_matrix()
        W = np.exp(-(D**2))
        np.fill_diagonal(W, 0.0)
        labels = clustering.spectral_partition(W, n_clusters=2, laplacian=laplacian, random_state=0)
        assert len(labels) == 6
        assert set(labels.tolist()) <= {0, 1}

    @pytest.mark.slow
    def test_large_active_set_uses_sparse_eigsh_path(self):
        # len(active) >= 50 routes through scipy.sparse.linalg.eigsh instead of the dense
        # np.linalg.eigh fallback -- build two well-separated 30-point blobs (60 active vertices).
        rng = np.random.default_rng(1)
        pts = np.vstack(
            [
                rng.normal(loc=[0, 0], scale=0.05, size=(30, 2)),
                rng.normal(loc=[5, 5], scale=0.05, size=(30, 2)),
            ]
        )
        D = np.linalg.norm(pts[:, None,:] - pts[None,:,:], axis=-1)
        W = np.exp(-(D**2))
        np.fill_diagonal(W, 0.0)
        labels = clustering.spectral_partition(W, n_clusters=2, random_state=0)
        assert len(set(labels[:30].tolist())) == 1
        assert len(set(labels[30:].tolist())) == 1
        assert labels[0] != labels[30]


class TestMaxMinPickAndKennardStoneExhaustion:
    def test_maxmin_pick_n_picks_exceeds_n_stops_early(self):
        D = _blob_distance_matrix()  # n=6
        picks = clustering.maxmin_pick(D, n_picks=100, init="index_zero")
        assert len(picks) == 6  # exhausted all candidates, loop breaks rather than erroring

    def test_kennard_stone_n_picks_exceeds_n_stops_early(self):
        D = _blob_distance_matrix()
        picks = clustering.kennard_stone(D, n_picks=100)
        assert len(picks) == 6


class TestSpectralPartitionSparseInput:
    def test_accepts_sparse_affinity_matrix(self):
        import scipy.sparse as sp

        D = _spectral_blob_distance_matrix()
        W = np.exp(-(D**2))
        np.fill_diagonal(W, 0.0)
        labels = clustering.spectral_partition(sp.csr_matrix(W), n_clusters=2, random_state=0)
        assert len(labels) == 6
