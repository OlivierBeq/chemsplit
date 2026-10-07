import random

import numpy as np
import pytest

from chemsplit import determinism as det


def test_seed_for_reproducible():
    bundle = det.make_seed_bundle(42)
    g1 = det.seed_for(bundle, "foo.purpose", 0)
    g2 = det.seed_for(bundle, "foo.purpose", 0)
    assert np.array_equal(g1.integers(0, 1000, size=10), g2.integers(0, 1000, size=10))


def test_seed_for_different_purpose_or_k_differ():
    bundle = det.make_seed_bundle(42)
    a = det.seed_for(bundle, "foo", 0).integers(0, 10**9, size=20)
    b = det.seed_for(bundle, "bar", 0).integers(0, 10**9, size=20)
    c = det.seed_for(bundle, "foo", 1).integers(0, 10**9, size=20)
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_make_seed_bundle_int_vs_none_vs_generator():
    b_int = det.make_seed_bundle(7)
    assert (
        b_int.resolved_seed == 7
    )  # resolved_seed is always populated, not just for random_state=None
    b_none = det.make_seed_bundle(None)
    assert isinstance(b_none.resolved_seed, int)

    gen = np.random.default_rng(123)
    gen.integers(0, 10**6, size=3)  # advance the generator
    b_gen = det.make_seed_bundle(gen)
    # a fresh generator on the same seed must give the same root, proving the bundle derives
    # from the generator's current state rather than consuming draws from it
    b_gen_again = det.make_seed_bundle(gen)
    g1 = det.seed_for(b_gen, "x", 0).integers(0, 100, size=5)
    g2 = det.seed_for(b_gen_again, "x", 0).integers(0, 100, size=5)
    # gen's state did not advance between the two make_seed_bundle calls
    assert np.array_equal(g1, g2)


def test_make_seed_bundle_rejects_bool():
    with pytest.raises(TypeError):
        det.make_seed_bundle(True)


def test_seeded_python_random_isolated_from_global():
    bundle = det.make_seed_bundle(0)
    state_before = random.getstate()
    r1 = det.seeded_python_random(bundle, "ga.purpose", 0)
    r2 = det.seeded_python_random(bundle, "ga.purpose", 0)
    assert r1.random() == r2.random()
    assert random.getstate() == state_before


def test_argmax_tiebreak_smallest_index_wins():
    items = [0, 1, 2, 3]
    scores = {0: 5.0, 1: 5.0, 2: 3.0, 3: 5.0}
    assert det.argmax_tiebreak(lambda i: scores[i], items) == 0


def test_argmin_tiebreak_smallest_index_wins():
    items = [0, 1, 2, 3]
    scores = {0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0}
    assert det.argmin_tiebreak(lambda i: scores[i], items) == 0


def test_row_argmin_smallest_column_wins_ties():
    M = np.array([[3.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 4.0, 0.1]])
    out = det.row_argmin(M)
    assert out.tolist() == [1, 0, 2]
    assert out.dtype == np.int64


def test_row_argmin_matches_argmin_tiebreak():
    rng = np.random.default_rng(0)
    M = rng.integers(0, 3, size=(20, 6)).astype(float)  # many ties
    expected = [
        det.argmin_tiebreak(lambda j, r=r: M[r, j], range(M.shape[1])) for r in range(M.shape[0])
    ]
    assert det.row_argmin(M).tolist() == expected


@pytest.mark.parametrize("bad", [np.zeros(3), np.zeros((2, 0))])
def test_row_argmin_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        det.row_argmin(bad)


def test_stable_sort_ascending_ties_preserved():
    seq = ["b0", "a1", "b2", "a3"]  # keys: b,a,b,a
    out = det.stable_sort(seq, key=lambda s: s[0])
    assert out == ["a1", "a3", "b0", "b2"]


def test_stable_sort_descending_ties_still_ascending_index():
    seq = [(0, "x"), (1, "x"), (2, "y"), (3, "x")]
    out = det.stable_sort(seq, key=lambda t: t[1], desc=True)
    # "y" group first (only elem index 2), then "x" group in original relative order 0,1,3
    assert out == [(2, "y"), (0, "x"), (1, "x"), (3, "x")]


def test_floor_round():
    assert det.floor_round(2.5) == 3
    assert det.floor_round(2.4999999) == 2
    assert det.floor_round(0.0) == 0
    assert det.floor_round(-0.4) == 0


def test_masked_argmax_matches_argmax_tiebreak_on_tie_heavy_scores():
    """The vectorised scan must reproduce ties-to-smallest-index exactly.

    Tanimoto matrices are almost entirely tied, so this is where a wrong rule would bite.
    """
    rng = np.random.default_rng(0)
    for trial in range(50):
        n = int(rng.integers(2, 40))
        # coarse quantisation => many exact ties
        scores = np.round(rng.random(n), 1)
        excluded = rng.random(n) < 0.3
        excluded[int(rng.integers(0, n))] = False  # keep at least one candidate
        candidates = [i for i in range(n) if not excluded[i]]
        expected = det.argmax_tiebreak(lambda i: scores[i], candidates)
        assert det.masked_argmax(scores, excluded) == expected, trial


def test_masked_argmax_rejects_degenerate_input():
    with pytest.raises(ValueError, match="every entry excluded"):
        det.masked_argmax(np.zeros(3), np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="equal shape"):
        det.masked_argmax(np.zeros(3), np.zeros(4, dtype=bool))


def test_first_argmax_2d_picks_the_lexicographically_smallest_maximal_pair():
    rng = np.random.default_rng(1)
    for trial in range(50):
        n = int(rng.integers(2, 25))
        M = np.round(rng.random((n, n)), 1)
        upper = np.triu(M, k=1)
        if upper.size == 0 or not np.any(upper > 0):
            continue
        # the construct this replaced: scan all pairs, keep the smallest maximal (i, j)
        iu = np.triu_indices(n, k=1)
        dvals = upper[iu]
        max_d = dvals.max()
        expected = min(
            (int(iu[0][k]), int(iu[1][k])) for k in range(len(dvals)) if dvals[k] >= max_d
        )
        assert det.first_argmax_2d(upper) == expected, trial


def test_first_ge_2d_prefers_an_earlier_near_miss_over_the_exact_maximum():
    """The seed-pair rule takes the smallest pair *within tolerance* of the maximum.

    An earlier pair sitting just below the maximum must win over a later pair attaining it
    exactly. Using a plain argmax here silently picks the wrong seed, which changes the whole
    selection that follows.
    """
    M = np.full((4, 4), -np.inf)
    M[0, 3] = 0.9999999   # earlier, marginally below the maximum
    M[1, 2] = 1.0         # the exact maximum
    assert det.first_ge_2d(M, float(M.max()) - 1e-6) == (0, 3)
    assert det.first_argmax_2d(M) == (1, 2)


def test_first_ge_2d_matches_a_python_scan():
    rng = np.random.default_rng(3)
    for trial in range(40):
        n = int(rng.integers(2, 20))
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        M = np.where(upper, np.round(rng.random((n, n)), 2), -np.inf)
        if not np.isfinite(M).any():
            continue
        value = float(M.max()) - 1e-6
        iu = np.triu_indices(n, k=1)
        expected = min(
            (int(iu[0][k]), int(iu[1][k]))
            for k in range(len(iu[0]))
            if M[iu[0][k], iu[1][k]] >= value
        )
        assert det.first_ge_2d(M, value) == expected, trial


def test_first_ge_2d_rejects_an_unreachable_value():
    with pytest.raises(ValueError, match="no entry reaches"):
        det.first_ge_2d(np.zeros((3, 3)), 1.0)
