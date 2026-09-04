"""Property-based tests using ``hypothesis``.

Each test documents which invariant/property it checks and why, rather than pinning specific
example values the way the rest of the suite does — these are meant to catch violations across
the *space* of valid inputs, not just the handful of examples other tests happen to construct.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from chemsplit._optimize import BalanceProblem, _objective
from chemsplit._unionfind import UnionFind, dense_label_encode
from chemsplit.base import (
    SplitResult,
    _ResolvedSizes,
    _decode_index_array,
    _encode_index_array,
    assign_groups,
    resolve_sizes,
)
from chemsplit.determinism import argmax_tiebreak, argmin_tiebreak, floor_round, stable_sort
from chemsplit.exceptions import ParameterError
from chemsplit.metrics import pairwise_distances

# ---------------------------------------------------------------------------
# 1. resolve_sizes -- arithmetic invariants and pure-function determinism.
# ---------------------------------------------------------------------------


@st.composite
def _size_specs(draw):
    n = draw(st.integers(min_value=2, max_value=300))

    def one():
        choice = draw(st.integers(min_value=0, max_value=2))
        if choice == 0:
            return None
        if choice == 1:
            return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        return draw(st.integers(min_value=0, max_value=n))

    return n, one(), one(), one()


@pytest.mark.core
@given(_size_specs())
@settings(deadline=None, max_examples=200)
def test_resolve_sizes_arithmetic_invariants_and_determinism(design):
    """whenever resolve_sizes succeeds, n_train+n_valid+n_test <= n, n_train>=1,
    n_test>=1, and -- since it's meant to be a pure function -- calling it again with identical
    arguments gives an identical result."""
    n, train_size, valid_size, test_size = design
    try:
        sizes = resolve_sizes(n, train_size, valid_size, test_size)
    except ParameterError:
        return  # not every generated combination is a valid size spec; that's expected
    assert sizes.n_train + sizes.n_valid + sizes.n_test <= n
    assert sizes.n_train >= 1
    assert sizes.n_test >= 1
    assert sizes.n_valid >= 0
    again = resolve_sizes(n, train_size, valid_size, test_size)
    assert again == sizes


# ---------------------------------------------------------------------------
# 2. floor_round -- round-half-away-from-zero, never banker's rounding.
# ---------------------------------------------------------------------------


@pytest.mark.core
@given(st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_floor_round_matches_its_own_definition(x):
    assert floor_round(x) == math.floor(x + 0.5)


@pytest.mark.core
@given(st.integers(min_value=0, max_value=100_000))
def test_floor_round_half_always_rounds_up_never_to_even(k):
    """The property that actually distinguishes floor_round from Python's builtin round(): a
    value ending in exactly.5 always rounds UP, regardless of whether the integer part is even
    or odd -- builtin round() would round 2.5 -> 2 (down, to even) but floor_round(2.5) -> 3."""
    assert floor_round(k + 0.5) == k + 1


# ---------------------------------------------------------------------------
# 3. argmax_tiebreak / argmin_tiebreak -- ties resolve to first occurrence.
# ---------------------------------------------------------------------------


@pytest.mark.core
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=1, max_size=30))
def test_argmax_tiebreak_returns_first_occurrence_of_the_max(scores):
    items = list(range(len(scores)))  # ascending index = tie-break priority order
    result = argmax_tiebreak(lambda i: scores[i], items)
    best = max(scores)
    expected = next(i for i in items if scores[i] == best)
    assert result == expected


@pytest.mark.core
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=1, max_size=30))
def test_argmin_tiebreak_returns_first_occurrence_of_the_min(scores):
    items = list(range(len(scores)))
    result = argmin_tiebreak(lambda i: scores[i], items)
    best = min(scores)
    expected = next(i for i in items if scores[i] == best)
    assert result == expected


# ---------------------------------------------------------------------------
# 4. stable_sort -- equal keys retain ascending original-index order, for both
# desc=True and desc=False (this is exactly the property naive sorted(reverse=True) violates
# for tied keys, which is why stable_sort exists as its own function).
# ---------------------------------------------------------------------------


@pytest.mark.core
@given(st.lists(st.integers(min_value=0, max_value=5), min_size=0, max_size=40))
def test_stable_sort_ties_preserve_ascending_original_order(keys):
    items = list(enumerate(keys))  # (original_index, key)
    for desc in (False, True):
        out = stable_sort(items, key=lambda t: t[1], desc=desc)
        for _key, group in itertools.groupby(out, key=lambda t: t[1]):
            idxs = [orig_idx for orig_idx, _ in group]
            assert idxs == sorted(idxs), f"desc={desc}: tie group {idxs} not ascending"


# ---------------------------------------------------------------------------
# 5. dense_label_encode -- dense ids in first-appearance order.
# ---------------------------------------------------------------------------


@pytest.mark.core
@given(st.lists(st.integers(min_value=0, max_value=5), min_size=0, max_size=40))
def test_dense_label_encode_first_appearance_order(keys):
    encoded = dense_label_encode(keys)
    assert len(encoded) == len(keys)
    distinct_in_order = list(dict.fromkeys(keys))  # first-appearance order, de-duplicated
    key_to_id = {k: i for i, k in enumerate(distinct_in_order)}
    expected = [key_to_id[k] for k in keys]
    assert list(int(v) for v in encoded) == expected
    assert len(set(int(v) for v in encoded)) == len(distinct_in_order)


# ---------------------------------------------------------------------------
# 6. UnionFind -- smallest-index representative, independent of the order
# union() calls are applied in.
# ---------------------------------------------------------------------------


@pytest.mark.core
@given(st.integers(min_value=1, max_value=15), st.data())
@settings(deadline=None, max_examples=100)
def test_unionfind_order_independent_and_smallest_index_representative(n, data):
    unions = data.draw(
        st.lists(
            st.tuples(st.integers(min_value=0, max_value=n - 1), st.integers(min_value=0, max_value=n - 1)),
            max_size=25,
        )
    )
    permuted = data.draw(st.permutations(unions))

    uf1 = UnionFind(n)
    for i, j in unions:
        uf1.union(i, j)
    uf2 = UnionFind(n)
    for i, j in permuted:
        uf2.union(i, j)

    for i in range(n):
        assert uf1.find(i) == uf2.find(i), "component membership must not depend on union order"

    for rep, members in uf1.components().items():
        assert rep == min(members)


# ---------------------------------------------------------------------------
# 7. Run-length index encoding -- exact round-trip, and the >=3-run boundary.
# ---------------------------------------------------------------------------


@pytest.mark.core
@given(
    st.lists(st.integers(min_value=0, max_value=10_000), unique=True, min_size=0, max_size=150).map(sorted)
)
def test_index_array_encode_decode_roundtrip(values):
    arr = np.asarray(values, dtype=np.int64)
    decoded = _decode_index_array(_encode_index_array(arr))
    assert np.array_equal(decoded, arr)


def test_index_array_run_of_two_is_not_collapsed_but_three_is():
    """the corresponding rule: runs of >= 3 consecutive integers collapse to a [start, end] pair; a run of
    exactly 2 must stay as two literal ints (this exact boundary is unlikely to be hit reliably by
    the property test above, so it's pinned as an explicit example)."""
    assert _encode_index_array(np.array([5, 6], dtype=np.int64)) == [5, 6]
    assert _encode_index_array(np.array([5, 6, 7], dtype=np.int64)) == [[5, 7]]


# ---------------------------------------------------------------------------
# 8. SplitResult -- any valid partition of range(n) constructs without raising, and
# assign_groups never splits a group across buckets.
# ---------------------------------------------------------------------------


@pytest.mark.core
@given(st.integers(min_value=0, max_value=60), st.data())
@settings(deadline=None, max_examples=75)
def test_splitresult_accepts_any_valid_partition(n, data):
    perm = list(data.draw(st.permutations(range(n))))
    cuts = sorted(data.draw(st.lists(st.integers(min_value=0, max_value=n), min_size=2, max_size=2)))
    a, b = cuts
    train, valid, test, discard = (
        sorted(perm[:a]),
        sorted(perm[a:b]),
        sorted(perm[b:]),
        [],
    )
    result = SplitResult(
        train=np.asarray(train, dtype=np.int64),
        valid=np.asarray(valid, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
        discard=np.asarray(discard, dtype=np.int64),
        groups=None,
        splitter_id="random",
        params={},
        n_records=n,
        metadata={},
    )
    assert result.n_records == n
    assert len(result.train) + len(result.valid) + len(result.test) + len(result.discard) == n


@pytest.mark.core
@given(st.integers(min_value=1, max_value=50), st.data())
@settings(deadline=None, max_examples=75)
def test_assign_groups_never_splits_a_group(n, data):
    labels_raw = data.draw(st.lists(st.integers(min_value=0, max_value=9), min_size=n, max_size=n))
    labels = np.asarray(dense_label_encode(labels_raw), dtype=np.int64)
    n_groups = int(labels.max()) + 1 if n else 0

    cuts = sorted(data.draw(st.lists(st.integers(min_value=0, max_value=n), min_size=2, max_size=2)))
    sizes = _ResolvedSizes(n_train=cuts[0], n_valid=cuts[1] - cuts[0], n_test=n - cuts[1])
    mode = data.draw(st.sampled_from(["greedy_desc", "balanced", "random"]))
    rng = np.random.default_rng(0)

    result = assign_groups(labels, sizes, mode, rng)
    # buckets with a zero-capacity target are omitted from the result dict entirely (assign_groups
    # never allocates to a bucket with no room) -- default to an empty array for those.
    empty = np.asarray([], dtype=np.int64)
    bucket_arrays = {name: result.get(name, empty) for name in ("train", "valid", "test")}
    all_idx = np.concatenate(list(bucket_arrays.values()))
    assert len(all_idx) == n
    assert len(set(all_idx.tolist())) == n  # disjoint and complete -- assign_groups never discards

    for g in range(n_groups):
        members = set(np.nonzero(labels == g)[0].tolist())
        containing = [name for name, arr in bucket_arrays.items() if members & set(arr.tolist())]
        assert len(containing) == 1, f"group {g} split across buckets: {containing}"
        assert members <= set(bucket_arrays[containing[0]].tolist())


# ---------------------------------------------------------------------------
# 9. pairwise_distances -- bounded-metric range, exact-zero diagonal, exact symmetry.
# ---------------------------------------------------------------------------


@pytest.mark.core
@given(
    arrays(dtype=np.uint8, shape=st.tuples(st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=16)), elements=st.integers(min_value=0, max_value=1)),
    st.sampled_from(["tanimoto", "dice", "cosine"]),
)
@settings(deadline=None, max_examples=60)
def test_pairwise_distances_bounded_metric_properties(X, metric):
    D = pairwise_distances(X, metric=metric)
    assert np.all(D >= -1e-6)
    assert np.all(D <= 1 + 1e-6)
    assert np.all(np.diag(D) == 0.0), "D[i][i] must be forced to exactly 0.0"
    assert np.array_equal(D, D.T), "symmetry must be exact (mirrored, never averaged)"


# ---------------------------------------------------------------------------
# 10. Order invariance for order_invariant=True splitters: permuting the
# input records permutes the output partition identically -- a record's fate depends only on
# its own content, never on its position in X.
# ---------------------------------------------------------------------------


def _fate_map(result, n):
    """{original_index: "train"|"valid"|"test"|"discard"} for every record 0.n-1."""
    fate = {}
    for name in ("train", "valid", "test", "discard"):
        for i in getattr(result, name).tolist():
            fate[i] = name
    assert len(fate) == n
    return fate


@given(st.permutations(range(12)))
@settings(deadline=None, max_examples=15)
def test_temporal_splitter_is_order_invariant(perm):
    from chemsplit.splitters.lineage import TemporalSplitter

    rng = np.random.default_rng(0)
    base_dates = np.array(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01",
         "2020-07-01", "2020-08-01", "2020-09-01", "2020-10-01", "2020-11-01", "2020-12-01"],
        dtype="datetime64[D]",
    )
    smiles = ["C" * (i + 1) for i in range(12)]

    sp1 = TemporalSplitter(random_state=0)
    r1 = sp1.split_result(smiles, dates=base_dates)[0]
    fate1 = _fate_map(r1, 12)

    perm = list(perm)
    smiles2 = [smiles[i] for i in perm]
    dates2 = base_dates[perm]
    sp2 = TemporalSplitter(random_state=0)
    r2 = sp2.split_result(smiles2, dates=dates2)[0]
    fate2_permuted = _fate_map(r2, 12)
    # map fate2's local (post-permutation) indices back to original record identity
    fate2 = {perm[local_i]: fate for local_i, fate in fate2_permuted.items()}

    assert fate1 == fate2


@given(st.permutations(range(12)))
@settings(deadline=None, max_examples=15)
def test_source_splitter_is_order_invariant(perm):
    """SourceSplitter groups purely by supplied source *value*, not position -- so as long as no
    two sources tie exactly in size (this fixture is built so they don't: sizes 6 and 6 would tie;
    use 8/4 instead), the split must not depend on where in X a record appears."""
    from chemsplit.splitters.lineage import SourceSplitter

    smiles = ["C" * (i + 1) for i in range(12)]
    source = ["A"] * 8 + ["B"] * 4

    sp1 = SourceSplitter(source=source, random_state=0)
    r1 = sp1.split_result(smiles)[0]
    fate1 = _fate_map(r1, 12)

    perm = list(perm)
    smiles2 = [smiles[i] for i in perm]
    source2 = [source[i] for i in perm]
    sp2 = SourceSplitter(source=source2, random_state=0)
    r2 = sp2.split_result(smiles2)[0]
    fate2_permuted = _fate_map(r2, 12)
    fate2 = {perm[local_i]: fate for local_i, fate in fate2_permuted.items()}

    assert fate1 == fate2


# ---------------------------------------------------------------------------
# 11. _optimize.py's shared _objective -- permutation invariance (every backend must call this one
# function so their results are comparable; that only holds if the objective itself doesn't
# care about item order, only about the (item, assignment) pairing).
# ---------------------------------------------------------------------------


@given(st.data())
@settings(deadline=None, max_examples=40)
def test_balance_objective_is_permutation_invariant(data):
    n_items = data.draw(st.integers(min_value=2, max_value=10))
    n_buckets = data.draw(st.integers(min_value=2, max_value=4))
    item_size = np.asarray(
        data.draw(st.lists(st.integers(min_value=1, max_value=20), min_size=n_items, max_size=n_items)),
        dtype=np.int64,
    )
    bucket_target_size = np.asarray(
        data.draw(
            st.lists(
                st.floats(min_value=1, max_value=50, allow_nan=False, allow_infinity=False),
                min_size=n_buckets,
                max_size=n_buckets,
            )
        ),
        dtype=np.float64,
    )
    assignment = np.asarray(
        data.draw(st.lists(st.integers(min_value=0, max_value=n_buckets - 1), min_size=n_items, max_size=n_items)),
        dtype=np.int64,
    )
    problem = BalanceProblem(n_items=n_items, n_buckets=n_buckets, item_size=item_size, bucket_target_size=bucket_target_size)
    obj1 = _objective(problem, assignment)

    perm = np.asarray(data.draw(st.permutations(range(n_items))))
    problem2 = BalanceProblem(
        n_items=n_items, n_buckets=n_buckets, item_size=item_size[perm], bucket_target_size=bucket_target_size
    )
    obj2 = _objective(problem2, assignment[perm])

    assert math.isclose(obj1, obj2, rel_tol=1e-9, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 12. SplitResult.to_json() / from_json() round-trip for arbitrary valid partitions.
# ---------------------------------------------------------------------------


@pytest.mark.core
@given(st.integers(min_value=0, max_value=40), st.data())
@settings(deadline=None, max_examples=50)
def test_splitresult_json_roundtrip(n, data):
    import json

    perm = list(data.draw(st.permutations(range(n))))
    cuts = sorted(data.draw(st.lists(st.integers(min_value=0, max_value=n), min_size=2, max_size=2)))
    a, b = cuts
    train, valid, test = sorted(perm[:a]), sorted(perm[a:b]), sorted(perm[b:])

    original = SplitResult(
        train=np.asarray(train, dtype=np.int64),
        valid=np.asarray(valid, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
        discard=np.asarray([], dtype=np.int64),
        groups=None,
        splitter_id="random",
        params={"a": 1, "b": [1, 2, 3]},
        n_records=n,
        metadata={"x": 2.5},
    )
    restored = SplitResult.from_json(original.to_json())

    assert np.array_equal(restored.train, original.train)
    assert np.array_equal(restored.valid, original.valid)
    assert np.array_equal(restored.test, original.test)
    assert np.array_equal(restored.discard, original.discard)
    assert restored.splitter_id == original.splitter_id
    assert restored.n_records == original.n_records
    assert json.loads(json.dumps(restored.params)) == json.loads(json.dumps(original.params))
    assert json.loads(json.dumps(restored.metadata)) == json.loads(json.dumps(original.metadata))
