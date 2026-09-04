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


def _valid_result(**overrides):
    base = dict(
        train=_arr(0, 1),
        valid=_arr(),
        test=_arr(2, 3),
        discard=_arr(),
        groups=None,
        splitter_id="random",
        params={},
        n_records=4,
        metadata={},
    )
    base.update(overrides)
    return SplitResult(**base)


class TestSplitResultInvariants:
    def test_i1_wrong_dtype(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I1"):
            _valid_result(train=np.array([0, 1], dtype=np.int32), test=_arr(2, 3))

    def test_i1_not_1d(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I1"):
            _valid_result(train=np.array([[0, 1]], dtype=np.int64), test=_arr(2, 3))

    def test_i2_not_complete(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I2"):
            _valid_result(n_records=5)  # only 4 records covered

    def test_i2_overlap(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I2"):
            _valid_result(train=_arr(0, 1, 2), test=_arr(2, 3))

    def test_i2_out_of_range_index(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I2"):
            _valid_result(train=_arr(0, 1), test=_arr(2, 99), n_records=4)

    def test_i3_wrong_shape(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I3"):
            _valid_result(groups=np.array([0, 0, 1], dtype=np.int64))  # 3 != n_records=4

    def test_i3_wrong_dtype(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I3"):
            _valid_result(groups=np.array([0, 0, 1, 1], dtype=np.int32))

    def test_i3_gap_in_group_ids(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I3"):
            _valid_result(groups=np.array([0, 0, 2, 2], dtype=np.int64))  # skips 1

    def test_i3_negative_group_id(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I3"):
            _valid_result(groups=np.array([-1, 0, 1, 1], dtype=np.int64))

    def test_i3_valid_groups_pass(self):
        result = _valid_result(groups=np.array([0, 0, 1, 1], dtype=np.int64))
        assert result.groups.tolist() == [0, 0, 1, 1]

    def test_i4_message_names_the_field(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I4"):
            # NaN survives _canonicalize_params's json.dumps *attempt* check differently than a
            # genuinely non-serialisable object -- use a value that canonicalize doesn't rescue.
            _valid_result(params={"x": float("nan")})  # json.dumps(nan) succeeds but round-trip
            # produces NaN != NaN, which is exactly I4's "does not round-trip" case.

    def test_i5_bad_splitter_id(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I5"):
            _valid_result(splitter_id="Butina")  # uppercase, not valid snake_case

    def test_i5_old_dotted_format_rejected(self):
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I5"):
            _valid_result(splitter_id="baseline.1")


class TestSplitResultAccessors:
    def test_as_tuple(self):
        result = _valid_result()
        assert result.as_tuple() == (result.train, result.test) or (
            np.array_equal(result.as_tuple()[0], result.train)
            and np.array_equal(result.as_tuple()[1], result.test)
        )

    def test_as_triple(self):
        result = _valid_result(valid=_arr(1), train=_arr(0), test=_arr(2, 3))
        tr, va, te = result.as_triple()
        assert np.array_equal(tr, result.train)
        assert np.array_equal(va, result.valid)
        assert np.array_equal(te, result.test)

    def test_to_frame_shape_and_partitions(self):
        result = _valid_result(groups=np.array([0, 0, 1, 1], dtype=np.int64))
        frame = result.to_frame()
        assert list(frame.columns) == ["index", "partition", "group"]
        assert len(frame) == 4
        assert set(frame["partition"]) == {"train", "test"}
        assert list(frame.sort_values("index")["index"]) == [0, 1, 2, 3]

    def test_to_frame_none_groups(self):
        result = _valid_result()
        frame = result.to_frame()
        assert frame["group"].isna().all()

    def test_to_json_from_json_round_trip(self):
        result = _valid_result(groups=np.array([0, 0, 1, 1], dtype=np.int64), metadata={"k": 1})
        text = result.to_json()
        payload = json.loads(text)
        assert payload["schema"] == "chemsplit/split/1"
        restored = SplitResult.from_json(text)
        assert np.array_equal(restored.train, result.train)
        assert np.array_equal(restored.test, result.test)
        assert np.array_equal(restored.groups, result.groups)
        assert restored.splitter_id == result.splitter_id
        assert restored.n_records == result.n_records
        assert restored.metadata == result.metadata

    def test_to_json_from_json_round_trip_no_groups(self):
        result = _valid_result()
        restored = SplitResult.from_json(result.to_json())
        assert restored.groups is None

    def test_run_length_encoding_collapses_long_runs(self):
        from chemsplit.base import _encode_index_array, _decode_index_array

        arr = np.array([0, 1, 2, 3, 4, 10, 20, 21, 22], dtype=np.int64)
        encoded = _encode_index_array(arr)
        # run of 5 (0.4) and run of 3 (20.22) collapse; the isolated 10 and the length-2 run at
        # the boundary do not.
        assert [0, 4] in encoded
        assert [20, 22] in encoded
        assert 10 in encoded
        decoded = _decode_index_array(encoded)
        assert np.array_equal(decoded, arr)

    def test_run_length_encoding_short_runs_stay_literal(self):
        from chemsplit.base import _encode_index_array

        arr = np.array([0, 1, 5, 6], dtype=np.int64)  # two runs of length 2 each
        encoded = _encode_index_array(arr)
        assert encoded == [0, 1, 5, 6]

    def test_run_length_round_trip_empty(self):
        from chemsplit.base import _encode_index_array, _decode_index_array

        arr = np.array([], dtype=np.int64)
        assert _encode_index_array(arr) == []
        assert _decode_index_array([]).tolist() == []


class TestResolveSizesMore:
    def test_negative_size_rejected(self):
        with pytest.raises(ParameterError):
            resolve_sizes(10, -1, None, 0.5)

    def test_float_out_of_range_rejected(self):
        with pytest.raises(ParameterError):
            resolve_sizes(10, 1.5, None, 0.2)

    def test_train_size_zero_valid(self):
        # train_size explicitly 0.0 is legal per SizeSpec semantics (that partition is empty) --
        # though resolve_sizes's own final "count_tr >= 1" check will then reject it; assert that
        # specific downstream rejection rather than an earlier one.
        with pytest.raises(ParameterError):
            resolve_sizes(10, 0.0, None, 0.5)


class TestCanonicalizeParams:
    def test_list_and_dict_values_recurse(self):
        from chemsplit.base import _canonicalize_params

        # np.float32 (unlike np.float64, which IS a Python `float` subclass and so is already
        # caught by the earlier isinstance(value, float) branch) is what actually exercises the
        # dedicated np.floating branch.
        out = _canonicalize_params({"a": [1, np.int64(2), (3, 4)], "b": {"c": np.float32(1.5)}})
        assert out == {"a": [1, 2, [3, 4]], "b": {"c": pytest.approx(1.5)}}
        assert isinstance(out["b"]["c"], float) and not isinstance(out["b"]["c"], np.floating)

    def test_already_serialisable_scalar_passthrough(self):
        from chemsplit.base import _canonicalize_params

        assert _canonicalize_params("plain string") == "plain string"
        assert _canonicalize_params(42) == 42
        assert _canonicalize_params(None) is None

    def test_i4_genuinely_unserialisable_after_canonicalization(self):
        """A dict with a non-string-or-numeric KEY (e.g. a tuple) survives
        ``_canonicalize_params`` unchanged (it only recurses into values), so it still fails
        ``json.dumps`` inside ``_check_i4`` for real -- exercising I4's actual except branch,
        distinct from the "round-trips to a different value" (NaN) case tested elsewhere."""
        from chemsplit.exceptions import InvariantError

        with pytest.raises(InvariantError, match="I4"):
            _valid_result(params={("tuple", "key"): "value"})


class TestResolveSizesEdgeCases:
    def test_invalid_type_size_spec(self):
        with pytest.raises(ParameterError):
            resolve_sizes(10, "half", None, 0.5)

    def test_one_unset_remainder_negative_raises(self):
        with pytest.raises(ParameterError):
            resolve_sizes(10, 8, None, 5)  # train=8 + test=5 > n=10 before valid absorbs

    def test_resolved_test_size_below_one_raises(self):
        with pytest.raises(ParameterError):
            resolve_sizes(10, 9, 1, None)  # test absorbs 10-9-1=0

    def test_all_three_explicit_sum_exceeds_n_raises(self):
        with pytest.raises(ParameterError, match="exceeds n"):
            resolve_sizes(10, 6, 3, 3)  # n_unset == 0, sum == 12 > n == 10


class TestContextGetFeatures:
    def test_raises_without_featurizer_when_no_raw_features(self):
        from chemsplit.base import _Context, _ResolvedSizes
        from chemsplit.determinism import make_seed_bundle

        ctx = _Context(
            n=2, mols=["a", "b"], smiles=None, y=None, groups_in=None, dates=None, targets=None,
            sequences=None, sizes=_ResolvedSizes(n_train=1, n_valid=0, n_test=1),
            rng_seeds=make_seed_bundle(0),
        )
        with pytest.raises(ValueError, match="requires a featurizer"):
            ctx.get_features()

    def test_caches_transform_result(self):
        from chemsplit.base import _Context, _ResolvedSizes
        from chemsplit.determinism import make_seed_bundle

        calls = []

        class FakeFeaturizer:
            name = "fake"

            def get_params(self):
                return {}

            def transform(self, mols):
                calls.append(1)
                return np.zeros((len(mols), 2))

        ctx = _Context(
            n=3, mols=["a", "b", "c"], smiles=None, y=None, groups_in=None, dates=None,
            targets=None, sequences=None, sizes=_ResolvedSizes(n_train=2, n_valid=0, n_test=1),
            rng_seeds=make_seed_bundle(0),
        )
        feat = FakeFeaturizer()
        out1 = ctx.get_features(feat)
        out2 = ctx.get_features(feat)
        assert out1 is out2
        assert len(calls) == 1

    def test_raw_features_returned_directly_ignoring_featurizer(self):
        from chemsplit.base import _Context, _ResolvedSizes
        from chemsplit.determinism import make_seed_bundle

        raw = np.ones((3, 2))
        ctx = _Context(
            n=3, mols=None, smiles=None, y=None, groups_in=None, dates=None, targets=None,
            sequences=None, sizes=_ResolvedSizes(n_train=2, n_valid=0, n_test=1),
            rng_seeds=make_seed_bundle(0), raw_features=raw,
        )
        assert ctx.get_features() is raw


class TestGetParamNames:
    def test_mro_walk_includes_mixin_params(self):
        class Mixin:
            def __init__(self, *, mixin_param=1, **kw):
                pass

        class Combined(Mixin, ToyRandomSplitter):
            def __init__(self, *, own_param=2, **kw):
                super().__init__(**kw)
                self.own_param = own_param

        names = Combined._get_param_names()
        assert "own_param" in names
        assert "mixin_param" in names
        assert "train_size" in names  # inherited from BaseSplitter.__init__


class TestSanitizeParams:
    def test_direct_call(self):
        out = ToyRandomSplitter._sanitize_params({"a": np.int64(3), "b": (1, 2)})
        assert out == {"a": 3, "b": [1, 2]}


class TestValidateBaseParams:
    def test_n_splits_invalid_type_string(self):
        with pytest.raises(ParameterError, match="n_splits"):
            ToyRandomSplitter(n_splits="bogus")

    def test_n_splits_auto_allowed(self):
        # "auto"/"loo" are subclass-specific sentinels BaseSplitter itself tolerates without
        # raising -- construction succeeds even though this base class does nothing special with
        # them (a concrete splitter like KFoldSplitter interprets "loo" itself).
        s = ToyRandomSplitter(n_splits="auto")
        assert s.n_splits == "auto"

    def test_n_splits_below_one(self):
        with pytest.raises(ParameterError, match="n_splits"):
            ToyRandomSplitter(n_splits=0)

    def test_train_size_bool_rejected(self):
        with pytest.raises(ParameterError, match="bool"):
            ToyRandomSplitter(train_size=True)

    def test_train_size_float_out_of_range(self):
        with pytest.raises(ParameterError):
            ToyRandomSplitter(train_size=1.5)

    def test_train_size_negative_int(self):
        with pytest.raises(ParameterError):
            ToyRandomSplitter(train_size=-3)

    def test_n_jobs_bool_rejected(self):
        with pytest.raises(ParameterError, match="n_jobs"):
            ToyRandomSplitter(n_jobs=True)

    def test_n_jobs_non_int_rejected(self):
        with pytest.raises(ParameterError, match="n_jobs"):
            ToyRandomSplitter(n_jobs="1")


class TestRunAndSplitResultVariants:
    def test_split_result_direct_call(self):
        X = make_X(20)
        results = ToyRandomSplitter(random_state=0).split_result(X)
        assert len(results) == 1
        assert isinstance(results[0], SplitResult)

    def test_compute_groups_not_group_forming_raises(self):
        with pytest.raises(NotImplementedError):
            ToyRandomSplitter().compute_groups(make_X(10))

    def test_requires_labels_missing_y_raises_label_error(self):
        from chemsplit.exceptions import LabelError

        class RequiresLabels(ToyRandomSplitter):
            requires_labels: ClassVar[bool] = True

        with pytest.raises(LabelError):
            list(RequiresLabels().split(make_X(10)))

    def test_y_length_mismatch_raises_label_error(self):
        from chemsplit.exceptions import LabelError

        with pytest.raises(LabelError):
            list(ToyRandomSplitter().split(make_X(10), y=np.zeros(3)))

    def test_resolved_valid_size_bool_is_false(self):
        # bool is technically truthy/falsy but is explicitly excluded from "non-empty valid_size"
        # semantics. __init__ itself already rejects a bool valid_size (_validate_base_params),
        # so reach _resolved_valid_size_nonzero's own bool guard by mutating post-construction.
        s = ToyRandomSplitter()
        s.valid_size = True
        assert s._resolved_valid_size_nonzero() is False

    def test_splitter_id_mismatch_raises_invariant_error(self):
        from chemsplit.exceptions import InvariantError

        class WrongIdSplitter(ToyRandomSplitter):
            def _partition(self, ctx):
                results = super()._partition(ctx)
                # Sneak a mismatched splitter_id past the subclass's own construction to trigger
                # _run's own post-_partition consistency check.
                bad = results[0]
                object.__setattr__(bad, "splitter_id", "stratified_random")
                return [bad]

        with pytest.raises(InvariantError):
            list(WrongIdSplitter().split(make_X(20)))

    def test_get_n_splits_non_int_n_splits_defaults_to_one(self):
        s = ToyRandomSplitter(n_splits="auto")
        assert s.get_n_splits() == 1


class TestAssignGroupsBalancedAndKfold:
    def test_assign_groups_balanced_mode(self):
        from chemsplit.base import _ResolvedSizes, assign_groups

        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        sizes = _ResolvedSizes(n_train=5, n_valid=0, n_test=4)
        rng = np.random.default_rng(0)
        out = assign_groups(labels, sizes, "balanced", rng)
        all_idx = sorted(out["train"].tolist() + out["test"].tolist())
        assert all_idx == list(range(9))

    def test_assign_groups_kfold_direct(self):
        from chemsplit.base import assign_groups_kfold

        labels = np.array([0, 0, 1, 1, 2, 3, 4, 5], dtype=np.int64)
        rng = np.random.default_rng(0)
        folds = assign_groups_kfold(labels, n_splits=3, rng=rng)
        assert len(folds) == 3
        all_idx = sorted(np.concatenate(folds).tolist())
        assert all_idx == list(range(8))
        # groups stay atomic within a fold: group 0 (indices 0,1) never split across folds
        for fold in folds:
            fold_set = set(fold.tolist())
            assert (0 in fold_set) == (1 in fold_set)


class TestGroupSplitterConstruction:
    def test_negative_size_tolerance_rejected(self):
        with pytest.raises(ParameterError, match="size_tolerance"):
            ToyGroupSplitter(size_tolerance=-0.1)

    def test_invalid_group_assignment_rejected(self):
        with pytest.raises(ParameterError, match="group_assignment"):
            ToyGroupSplitter(group_assignment="not_a_real_mode")


class TestGroupSplitterComputeGroupsAndKFold:
    def test_compute_groups_returns_labels(self):
        keys = [0, 0, 1, 1, 2, 2]
        X = np.array(keys, dtype=float).reshape(-1, 1)
        labels = ToyGroupSplitter().compute_groups(X)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[4] == labels[5]

    def test_n_splits_greater_than_one_yields_k_folds(self):
        keys = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        X = np.array(keys, dtype=float).reshape(-1, 1)
        splitter = ToyGroupSplitter(n_splits=3, random_state=0)
        results = splitter.split_result(X)
        assert len(results) == 3
        # every record appears in exactly one fold's test set across the 3 folds
        all_test = sorted(sum((r.test.tolist() for r in results), []))
        assert all_test == list(range(12))

    def test_n_splits_greater_than_one_with_valid_size_raises(self):
        keys = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        X = np.array(keys, dtype=float).reshape(-1, 1)
        splitter = ToyGroupSplitter(
            n_splits=2, valid_size=0.2, train_size=0.6, test_size=0.2, random_state=0
        )
        with pytest.raises(ConfigurationError, match="nested-CV"):
            splitter.split_result(X)
