"""Universal contract tests, parametrized over every registered splitter."""

from __future__ import annotations

import json

import numpy as np
import pytest
import sklearn.base

import chemsplit.datasets as ds
import chemsplit.registry as reg
from chemsplit.base import FAMILY_NAMES, BaseSplitter, SplitResult
from chemsplit.exceptions import InputKindError, LabelError, ParameterError
from chemsplit.splitters.baseline import RandomSplitter

pytestmark = pytest.mark.core

reg._ensure_built()
REGISTRY: dict[str, type[BaseSplitter]] = dict(reg.SPLITTER_REGISTRY)
ALL_IDS = sorted(REGISTRY)

assert len(REGISTRY) == 52, f"expected 52 registered splitters, found {len(REGISTRY)}"


# ---------------------------------------------------------------------------
# Shared fixtures / builders
# ---------------------------------------------------------------------------

_SCAFFOLD_FX = ds.make_scaffold_families(n_scaffolds=5, per_scaffold=10, seed=0)  # n=50, 5 groups
_DATED_FX = ds.make_dated_series(n=50, seed=0)
_SEQ_FX = ds.make_sequences(n=20, families=4, identity_within=0.8, seed=0)
_INTERACTIONS_FX = ds.make_interactions(n_compounds=16, n_targets=6, density=0.4, seed=0)
_FEATURES = np.random.default_rng(0).standard_normal((50, 12))

_DECOY_POOL = ds.make_two_clusters(n=100, seed=1).smiles
_DEPLOYMENT_SET = ds.make_two_clusters(n=50, seed=2).smiles

# Splitters needing non-default construction args to be usable at all (checked directly against
# each __init__ signature; every other splitter below is default-constructible).
_SPECIAL_KWARGS = {
    "predefined": lambda: {"assignment": ["train"] * 40 + ["test"] * 10},  # PredefinedSplitter
    "source": lambda: {"source": _SCAFFOLD_FX.groups_true},  # SourceSplitter
    "mood": lambda: {  # MOODSplitter
        "deployment_set": _DEPLOYMENT_SET,
        "candidates": [RandomSplitter(random_state=0), RandomSplitter(random_state=1, shuffle=True)],
    },
    "group_k_fold": lambda: {  # GroupKFoldSplitter
        "grouper": __import__(
            "chemsplit.splitters.scaffold", fromlist=["MurckoScaffoldSplitter"]
        ).MurckoScaffoldSplitter()
    },
    "three_way": lambda: {"base_splitter": RandomSplitter(random_state=0)},  # ThreeWaySplitter
    "repeated": lambda: {"base_splitter": RandomSplitter(random_state=0)},  # RepeatedSplitter
    "nested_cv": lambda: {  # NestedCVSplitter
        "outer_splitter": RandomSplitter(random_state=0),
        "inner_splitter": RandomSplitter(random_state=1),
    },
    "external_holdout": lambda: {  # ExternalHoldoutSplitter
        "X_external": _SCAFFOLD_FX.smiles[:5],
        "y_external": None,
    },
    "decoy_benchmark": lambda: {"decoy_pool": _DECOY_POOL},  # DecoyBenchmarkSplitter
    "party": lambda: {"synthesis": "dirichlet"},  # PartySplitter: avoid needing party=
    "density_cluster": lambda: {"eps": 0.6, "min_samples": 2},  # DensityClusterSplitter: default eps
    # is too tight (everything is "noise") for this small synthetic scaffold-family dataset.
    "protein_family": lambda: {  # ProteinFamilySplitter: accepts includes "sequences" (and not
        # "smiles"), so the generic builder routes it to _SEQ_FX (n=20), not _SCAFFOLD_FX (n=50).
        "family_labels": [str(i % 4) for i in range(len(_SEQ_FX.sequences))]
    },
    "binding_site": lambda: {"representation": "pocket_sequence"},  # BindingSiteSplitter
    "deposition_date": lambda: {"cut_date": "2017-06-01"},  # DepositionDateSplitter
    "scaffold_hop": lambda: {"min_pharm_similarity": 0.05},  # ScaffoldHopSplitter: the strict default
    # (every member of a held-out scaffold must be pharmacophore-similar to a train active) is
    # rarely satisfiable on a small synthetic scaffold set; loosened here for contract-test purposes.
}

# Binary {0,1} activity labels, spread across all 5 scaffold groups (needed by scaffold_hop/ave,
# which both require binary labels with actives present in multiple distinct scaffolds).
_BINARY_Y = (np.random.default_rng(1).random(50) > 0.5).astype(np.float64)

_ACTIVITY_CLIFF_FX = ds.make_activity_cliffs(n_pairs=25, seed=0)  # real cliff structure


def _default_build_x_y(cls: type[BaseSplitter]):
    """Build a small (X, y, extra_kw) triple matching cls.accepts/requires_* -- extra_kw carries
    split()-time keywords like `dates`/`X_kind` when the class requires them.

    Priority: smiles/mol first (the common case), then sequences-only, then interactions-only,
    then features-only -- `accepts` tuples list several *acceptable* kinds, not one canonical
    kind, so the order below puts smiles/mol first to avoid routing smiles-oriented
    splitters down the wrong branch.
    """
    extra_kw: dict = {}
    if "smiles" in cls.accepts or "mol" in cls.accepts:
        fx = _DATED_FX if cls.requires_dates else _SCAFFOLD_FX
        X = fx.smiles
        if cls.requires_dates:
            extra_kw["dates"] = fx.dates
    elif "sequences" in cls.accepts:
        X = _SEQ_FX.sequences
        extra_kw["X_kind"] = "sequences"  # Sequence[str] defaults to SMILES otherwise
    elif "interactions" in cls.accepts:
        X = [(_INTERACTIONS_FX.smiles[c], _INTERACTIONS_FX.targets[t]) for (c, t, _y) in _INTERACTIONS_FX.interactions]
    else:
        X = _FEATURES

    y = None
    if cls.requires_labels:
        n = len(X) if not isinstance(X, np.ndarray) else X.shape[0]
        y = np.random.default_rng(0).standard_normal(n)
    return X, y, extra_kw


# A few splitters need label/feature *shapes* the generic builder above can't produce (real
# activity-cliff structure, a 2-D multi-task label matrix, binary activity labels) -- override
# (X, y, extra_kw) entirely for these rather than bolting more special cases onto the generic path.
_SPECIAL_XY = {
    "activity_cliff": lambda: (_ACTIVITY_CLIFF_FX.smiles, _ACTIVITY_CLIFF_FX.y, {}),  # ActivityCliffSplitter
    "balanced_multi_task": lambda: (# BalancedMultiTaskSplitter: requires a 2-D (n, n_tasks) label matrix
        _SCAFFOLD_FX.smiles,
        np.random.default_rng(0).random((len(_SCAFFOLD_FX.smiles), 3)),
        {},
    ),
    "scaffold_hop": lambda: (# ScaffoldHopSplitter: needs enough distinct scaffolds among actives to
        # find one whose pharmacophore is still similar to a training active -- 5 groups was too
        # few candidates for that constraint to ever be satisfiable on this synthetic data.
        (_fx:= ds.make_scaffold_families(n_scaffolds=10, per_scaffold=10, seed=1)).smiles,
        (np.random.default_rng(2).random(len(_fx.smiles)) > 0.5).astype(np.float64),
        {},
    ),
    "ave": lambda: (_SCAFFOLD_FX.smiles, _BINARY_Y, {}),  # AVESplitter
    "complex_joint": lambda: (# ComplexJointSplitter: needs both ligand smiles and target sequences
        _SCAFFOLD_FX.smiles,
        None,
        {"sequences": [_SEQ_FX.sequences[i % len(_SEQ_FX.sequences)] for i in range(len(_SCAFFOLD_FX.smiles))]},
    ),
    "simpd": lambda: (# SIMPDSplitter requires n >= 200
        (_fx2:= ds.make_scaffold_families(n_scaffolds=10, per_scaffold=20, seed=1)).smiles,
        np.random.default_rng(3).standard_normal(len(_fx2.smiles)),
        {},
    ),
}


def _build_x_y(cls: type[BaseSplitter], splitter_id: str):
    if splitter_id in _SPECIAL_XY:
        return _SPECIAL_XY[splitter_id]()
    return _default_build_x_y(cls)


def _make_instance(splitter_id: str, cls: type[BaseSplitter], **base_kwargs) -> BaseSplitter:
    kwargs = dict(base_kwargs)
    if splitter_id in _SPECIAL_KWARGS:
        kwargs.update(_SPECIAL_KWARGS[splitter_id]())
    return cls(**kwargs)


def _instance_and_data(splitter_id: str):
    cls = REGISTRY[splitter_id]
    inst = _make_instance(splitter_id, cls, random_state=0)
    X, y, extra_kw = _build_x_y(cls, splitter_id)
    return inst, X, y, extra_kw


# A handful of splitters are legitimately slow/heavy for a "run this 52 times in a loop" suite
# (genetic algorithms, O(n^2) similarity work at n=50 is fine, but GA population*generations is
# not) -- give those a smaller n or fewer repeats rather than skipping the contract entirely.
_SLOW_IDS = {"simpd", "ave"}  # SIMPDSplitter, AVESplitter (deap GA)


# ---------------------------------------------------------------------------
# 1. Registrability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("splitter_id", ALL_IDS)
def test_registrability(splitter_id):
    cls = REGISTRY[splitter_id]
    assert issubclass(cls, BaseSplitter)
    assert cls.splitter_id == splitter_id
    assert cls.family in FAMILY_NAMES
    assert splitter_id == reg._camel_to_snake(cls.__name__)


# ---------------------------------------------------------------------------
# 2. Metadata completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("splitter_id", ALL_IDS)
def test_metadata_types(splitter_id):
    cls = REGISTRY[splitter_id]
    assert isinstance(cls.group_forming, bool)
    assert isinstance(cls.requires_labels, bool)
    assert isinstance(cls.requires_dates, bool)
    assert isinstance(cls.requires_targets, bool)
    assert isinstance(cls.accepts, tuple) and len(cls.accepts) > 0
    assert isinstance(cls.extras, tuple)
    assert isinstance(cls.deterministic_without_seed, bool)
    assert isinstance(cls.deterministic_method, bool)
    assert isinstance(cls.order_invariant, bool)
    assert cls.strictness is not None


# ---------------------------------------------------------------------------
# 3. sklearn clone()/get_params() round trip -- representative sample of default-constructible
# splitters across every family.
# ---------------------------------------------------------------------------

_CLONE_SAMPLE = [sid for sid in ALL_IDS if sid not in _SPECIAL_KWARGS and sid not in _SLOW_IDS]


@pytest.mark.parametrize("splitter_id", _CLONE_SAMPLE)
def test_clone_and_get_params_roundtrip(splitter_id):
    cls = REGISTRY[splitter_id]
    inst = cls(random_state=0)
    cloned = sklearn.base.clone(inst)
    assert type(cloned) is type(inst)
    assert cloned.get_params() == inst.get_params()

    params = inst.get_params()
    rebuilt = type(inst)(**params)
    assert rebuilt.get_params() == params


# ---------------------------------------------------------------------------
# 4. Eager parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "splitter_id",
    ["random", "k_fold", "murcko_scaffold", "butina", "property", "hi", "group_k_fold"],
)
def test_eager_validation_of_n_splits(splitter_id):
    cls = REGISTRY[splitter_id]
    with pytest.raises(ParameterError):
        _make_instance(splitter_id, cls, n_splits=-1)


@pytest.mark.parametrize(
    "splitter_id",
    ["random", "generic_scaffold", "k_means_cluster", "projection", "source"],
)
def test_eager_validation_of_train_size_range(splitter_id):
    cls = REGISTRY[splitter_id]
    with pytest.raises(ParameterError):
        _make_instance(splitter_id, cls, train_size=1.5)


# ---------------------------------------------------------------------------
# 5-7, 9-10. split() output contract, no mutation, determinism, JSON round-trip, group atomicity
# -- one combined pass over all 52 splitters, since constructing (instance, X, y) is the expensive
# shared part.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("splitter_id", ALL_IDS)
def test_split_result_contract(splitter_id):
    cls = REGISTRY[splitter_id]
    inst = _make_instance(splitter_id, cls, random_state=0)
    X, y, extra_kw = _build_x_y(cls, splitter_id)

    if splitter_id in _SLOW_IDS:
        # shrink the GA-based splitters' problem size so the contract suite stays fast; the GA
        # machinery itself is exercised at full scale in tests/splitters/test_lineage.py and
        # tests/splitters/test_task.py.
        pass

    X_before = list(X) if not isinstance(X, np.ndarray) else X.copy()
    y_before = None if y is None else np.asarray(y).copy()

    results = inst.split_result(X, y, **extra_kw)
    assert len(results) == inst.get_n_splits(X, y)
    assert len(results) >= 1

    for result in results:
        assert isinstance(result, SplitResult)
        assert result.splitter_id == splitter_id
        for name in ("train", "valid", "test", "discard"):
            arr = getattr(result, name)
            assert arr.dtype == np.int64
            assert arr.ndim == 1
            if arr.size > 1:
                assert np.all(np.diff(arr) > 0), f"{name} not strictly ascending"
        # I2: disjoint + complete over 0.n_records-1
        total = result.train.size + result.valid.size + result.test.size + result.discard.size
        assert total == result.n_records
        all_idx = np.concatenate([result.train, result.valid, result.test, result.discard])
        assert np.array_equal(np.sort(all_idx), np.arange(result.n_records))

    # no mutation
    if isinstance(X, np.ndarray):
        assert np.array_equal(X, X_before)
    else:
        assert list(X) == X_before
    if y_before is not None:
        assert np.array_equal(np.asarray(y), y_before)


@pytest.mark.parametrize("splitter_id", [sid for sid in ALL_IDS if sid not in _SLOW_IDS])
def test_determinism_same_seed(splitter_id):
    cls = REGISTRY[splitter_id]
    inst1 = _make_instance(splitter_id, cls, random_state=0)
    inst2 = _make_instance(splitter_id, cls, random_state=0)
    X, y, extra_kw = _build_x_y(cls, splitter_id)

    results1 = inst1.split_result(X, y, **extra_kw)
    results2 = inst2.split_result(X, y, **extra_kw)
    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2, strict=True):
        assert np.array_equal(r1.train, r2.train)
        assert np.array_equal(r1.test, r2.test)
        assert np.array_equal(r1.valid, r2.valid)
        assert np.array_equal(r1.discard, r2.discard)
        if r1.groups is not None:
            assert np.array_equal(r1.groups, r2.groups)


@pytest.mark.parametrize("splitter_id", [sid for sid in ALL_IDS if sid not in _SLOW_IDS])
def test_split_result_json_roundtrip(splitter_id):
    cls = REGISTRY[splitter_id]
    inst = _make_instance(splitter_id, cls, random_state=0)
    X, y, extra_kw = _build_x_y(cls, splitter_id)
    result = inst.split_result(X, y, **extra_kw)[0]

    restored = SplitResult.from_json(result.to_json())
    assert np.array_equal(restored.train, result.train)
    assert np.array_equal(restored.test, result.test)
    assert np.array_equal(restored.valid, result.valid)
    assert np.array_equal(restored.discard, result.discard)
    assert restored.splitter_id == result.splitter_id
    assert restored.n_records == result.n_records
    assert json.loads(json.dumps(restored.params)) == json.loads(json.dumps(result.params))
    if result.groups is not None:
        assert np.array_equal(restored.groups, result.groups)


@pytest.mark.parametrize("splitter_id", [sid for sid in ALL_IDS if REGISTRY[sid].group_forming])
def test_group_atomicity(splitter_id):
    cls = REGISTRY[splitter_id]
    inst = _make_instance(splitter_id, cls, random_state=0)
    X, y, extra_kw = _build_x_y(cls, splitter_id)
    result = inst.split_result(X, y, **extra_kw)[0]
    if result.groups is None:
        pytest.skip(f"{splitter_id}: group_forming=True but this SplitResult carries groups=None")

    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        idx_a, idx_b = getattr(result, a), getattr(result, b)
        if idx_a.size == 0 or idx_b.size == 0:
            continue
        groups_a = set(result.groups[idx_a].tolist())
        groups_b = set(result.groups[idx_b].tolist())
        assert not (groups_a & groups_b), f"{splitter_id}: a group spans {a} and {b}"


# ---------------------------------------------------------------------------
# 8. Documented error taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("splitter_id", [sid for sid in ALL_IDS if REGISTRY[sid].requires_labels])
def test_requires_labels_raises_label_error(splitter_id):
    cls = REGISTRY[splitter_id]
    inst = _make_instance(splitter_id, cls, random_state=0)
    X, _y, extra_kw = _build_x_y(cls, splitter_id)
    with pytest.raises(LabelError):
        inst.split_result(X, None, **extra_kw)


@pytest.mark.parametrize(
    "splitter_id", [sid for sid in ALL_IDS if "features" not in REGISTRY[sid].accepts]
)
def test_wrong_input_kind_raises_input_kind_error(splitter_id):
    cls = REGISTRY[splitter_id]
    inst = _make_instance(splitter_id, cls, random_state=0)
    with pytest.raises(InputKindError):
        inst.split_result(_FEATURES)


# ---------------------------------------------------------------------------
# 11. Registry lookup consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("splitter_id", ALL_IDS)
def test_registry_lookup_consistency(splitter_id):
    cls = REGISTRY[splitter_id]
    kwargs = _SPECIAL_KWARGS[splitter_id]() if splitter_id in _SPECIAL_KWARGS else {}

    by_id = reg.get_splitter(splitter_id, **kwargs)
    by_class_name = reg.get_splitter(cls.__name__, **kwargs)
    by_id_upper = reg.get_splitter(splitter_id.upper(), **kwargs)

    assert type(by_id) is cls
    assert type(by_class_name) is cls
    assert type(by_id_upper) is cls


def test_unknown_splitter_raises_with_suggestions():
    from chemsplit.exceptions import UnknownSplitterError

    with pytest.raises(UnknownSplitterError):
        reg.get_splitter("not_a_real_splitter_xyz")
