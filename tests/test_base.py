import json
from typing import ClassVar

import numpy as np
import pytest

from chemsplit.base import BaseSplitter, GroupSplitter, SplitResult, Strictness, resolve_sizes
from chemsplit.determinism import seed_for
from chemsplit.exceptions import ConfigurationError, EmptyInputError, InputKindError, ParameterError


class ToyRandomSplitter(BaseSplitter):
    splitter_id = "random"
    family = "baseline"
    strictness = Strictness.OPTIMISTIC
    accepts: ClassVar[tuple[str,...]] = ("features",)
    deterministic_without_seed: ClassVar[bool] = False
    order_invariant: ClassVar[bool] = True

    def _partition(self, ctx):
        rng = seed_for(ctx.rng_seeds, "random.permutation", 0)
        perm = rng.permutation(ctx.n)
        a = ctx.sizes.n_train
        b = a + ctx.sizes.n_valid
        c = b + ctx.sizes.n_test
        train = np.sort(perm[:a])
        valid = np.sort(perm[a:b])
        test = np.sort(perm[b:c])
        discard = np.sort(perm[c:])
        return [
            SplitResult(
                train=train.astype(np.int64),
                valid=valid.astype(np.int64),
                test=test.astype(np.int64),
                discard=discard.astype(np.int64),
                groups=None,
                splitter_id=self.splitter_id,
                params=self.get_params(),
                n_records=ctx.n,
                metadata={"resolved_seed": ctx.extra.get("resolved_seed")},
            )
        ]


class ToyGroupSplitter(GroupSplitter):
    splitter_id = "murcko_scaffold"
    family = "scaffold"
    strictness = Strictness.MODERATE
    accepts: ClassVar[tuple[str,...]] = ("features",)

    def _group_labels(self, ctx):
        # group by the integer value of column 0 -- a synthetic, deterministic "scaffold key"
        from chemsplit._unionfind import dense_label_encode

        keys = [int(row[0]) for row in ctx.get_features()]
        return dense_label_encode(keys)


def make_X(n: int) -> np.ndarray:
    return np.arange(n, dtype=float).reshape(-1, 1)


def test_random_splitter_shapes_and_determinism():
    X = make_X(20)
    s1 = ToyRandomSplitter(random_state=0)
    s2 = ToyRandomSplitter(random_state=0)
    (train1, test1), = list(s1.split(X))
    (train2, test2), = list(s2.split(X))
    assert np.array_equal(train1, train2)
    assert np.array_equal(test1, test2)
    assert len(train1) == 16 and len(test1) == 4  # default 0.8/0.2


def test_different_seed_changes_output():
    X = make_X(50)
    (t1, _), = list(ToyRandomSplitter(random_state=0).split(X))
    (t2, _), = list(ToyRandomSplitter(random_state=1).split(X))
    assert not np.array_equal(t1, t2)


def test_split_rejects_nonzero_valid_size():
    X = make_X(20)
    splitter = ToyRandomSplitter(train_size=0.6, valid_size=0.2, test_size=0.2)
    with pytest.raises(ConfigurationError):
        list(splitter.split(X))
    # but split_with_validation works
    (tr, va, te), = list(splitter.split_with_validation(X))
    assert len(tr) + len(va) + len(te) == 20


def test_input_kind_rejected():
    class SmilesOnly(ToyRandomSplitter):
        accepts: ClassVar[tuple[str,...]] = ("smiles",)

    X = make_X(10)
    with pytest.raises(InputKindError):
        list(SmilesOnly().split(X))


def test_empty_input_error():
    X = make_X(1)
    with pytest.raises(EmptyInputError):
        list(ToyRandomSplitter().split(X))


def test_group_splitter_assign_groups_end_to_end():
    # 12 records, keys 0,0,0,1,1,2,2,2,2,3,3,3 -> groups of sizes 3,2,4,3
    keys = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    X = np.array(keys, dtype=float).reshape(-1, 1)
    splitter = ToyGroupSplitter(train_size=0.7, test_size=0.3, random_state=0)
    (train, test), = list(splitter.split(X))
    assert set(train.tolist()) | set(test.tolist()) == set(range(12))
    assert set(train.tolist()) & set(test.tolist()) == set()
    # no group split across the boundary: check group 2 (indices 5,6,7,8) all on one side
    group2 = {5, 6, 7, 8}
    assert group2 <= set(train.tolist()) or group2 <= set(test.tolist())


def test_group_splitter_deterministic():
    keys = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7]
    X = np.array(keys, dtype=float).reshape(-1, 1)
    s1 = ToyGroupSplitter(random_state=0, group_assignment="random")
    s2 = ToyGroupSplitter(random_state=0, group_assignment="random")
    (t1, _), = list(s1.split(X))
    (t2, _), = list(s2.split(X))
    assert np.array_equal(t1, t2)


# ---------------------------------------------------------------------------
# resolve_sizes
# ---------------------------------------------------------------------------


def test_resolve_sizes_default():
    r = resolve_sizes(100, None, None, None)
    assert (r.n_train, r.n_valid, r.n_test) == (80, 0, 20)


def test_resolve_sizes_bool_rejected():
    with pytest.raises(ParameterError):
        resolve_sizes(10, True, None, None)


def test_resolve_sizes_two_unset_ambiguous():
    with pytest.raises(ParameterError):
        resolve_sizes(10, 0.5, None, None)


def test_resolve_sizes_one_unset_absorbs_remainder():
    r = resolve_sizes(10, 0.5, None, 0.2)
    assert r.n_train == 5
    assert r.n_test == 2
    assert r.n_valid == 3


def test_resolve_sizes_leftover_goes_to_train():
    r = resolve_sizes(10, 0.5, 0.2, 0.2)
    assert r.n_train + r.n_valid + r.n_test == 10
    assert r.n_train == 6  # 5 + leftover(1)


def test_resolve_sizes_int_zero_rejected_per_pseudocode():
    with pytest.raises(ParameterError):
        resolve_sizes(10, 0, None, 5)


def test_resolve_sizes_exceeds_n():
    with pytest.raises(ParameterError):
        resolve_sizes(10, 8, 5, None)


# ---------------------------------------------------------------------------
# SplitResult invariants
# ---------------------------------------------------------------------------


def _arr(*vals):
    return np.asarray(vals, dtype=np.int64)


def test_split_result_i1_not_ascending():
    from chemsplit.exceptions import InvariantError

    with pytest.raises(InvariantError):
        SplitResult(
            train=_arr(2, 1),
            valid=_arr(),
            test=_arr(3),
            discard=_arr(),
            groups=None,
            splitter_id="random",
            params={},
            n_records=4,
            metadata={},
        )


def test_split_result_i3_groups_not_first_appearance():
    from chemsplit.exceptions import InvariantError

    with pytest.raises(InvariantError):
        SplitResult(
            train=_arr(0, 1),
            valid=_arr(),
            test=_arr(),
            discard=_arr(),
            groups=np.array([1, 0], dtype=np.int64),  # first appearance is label 1, not 0
            splitter_id="random",
            params={},
            n_records=2,
            metadata={},
        )


def test_split_result_i4_non_json_native_params_are_stringified():
    """``params`` is "JSON-serialisable, fully resolved" -- an audit record, not a
    reconstruction mechanism. A constructor argument that isn't natively JSON-representable (a
    live object, a callable, a numpy array/tuple/scalar) is stringified/canonicalized rather than
    making I4 impossible to satisfy for any splitter with a non-trivial parameter type (e.g. the
    embedding family's ``LatentSpaceSplitter(embedding=<callable>)``)."""
    sentinel = object()
    result = SplitResult(
        train=_arr(0),
        valid=_arr(),
        test=_arr(1),
        discard=_arr(),
        groups=None,
        splitter_id="random",
        params={
            "bad": sentinel,
            "a_tuple": (1, 2),
            "a_numpy_int": np.int64(3),
            "a_numpy_array": np.array([1.0, 2.0]),
        },
        n_records=2,
        metadata={},
    )
    assert result.params["bad"] == str(sentinel)
    assert result.params["a_tuple"] == [1, 2]
    assert result.params["a_numpy_int"] == 3 and isinstance(result.params["a_numpy_int"], int)
    assert result.params["a_numpy_array"].startswith("<ndarray")
    # Now genuinely JSON-round-trippable, per I4.
    json.loads(json.dumps(result.params))
