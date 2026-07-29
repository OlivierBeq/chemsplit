"""Tests for chemsplit.splitters.lineage (the lineage family, renamed "lineage")."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit.datasets import make_dated_series, make_scaffold_families, make_two_clusters
from chemsplit.exceptions import (
    ConstraintUnsatisfiableError,
    InputError,
    LabelError,
    ParameterError,
)
from chemsplit.splitters.lineage import (
    PartySplitter,
    SIMPDSplitter,
    SourceSplitter,
    TemporalSplitter,
)

try:
    import deap  # noqa: F401

    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False


# ---------------------------------------------------------------------------
# TemporalSplitter
# ---------------------------------------------------------------------------


def test_temporal_requires_dates():
    fx = make_dated_series(n=100, seed=0)
    sp = TemporalSplitter(random_state=0)
    with pytest.raises(LabelError):
        sp.split_result(fx.smiles)


def test_temporal_single_cut_chronological_order():
    fx = make_dated_series(n=200, seed=0)
    sp = TemporalSplitter(random_state=0)
    result = sp.split_result(fx.smiles, dates=fx.dates)[0]
    assert len(result.train) + len(result.valid) + len(result.test) + len(result.discard) == 200
    if len(result.train) and len(result.test):
        # every test date is >= every train date (embargo=0, tie_policy default "train")
        assert fx.dates[result.train].max() <= fx.dates[result.test].min()


def test_temporal_embargo_discards_boundary_records():
    fx = make_dated_series(n=200, seed=0)
    sp_no_embargo = TemporalSplitter(random_state=0, embargo=0)
    sp_embargo = TemporalSplitter(random_state=0, embargo=60)
    r0 = sp_no_embargo.split_result(fx.smiles, dates=fx.dates)[0]
    r1 = sp_embargo.split_result(fx.smiles, dates=fx.dates)[0]
    assert len(r1.discard) >= len(r0.discard)
    assert r1.metadata["n_embargoed"] >= 0


def test_temporal_all_same_date_raises():
    fx = make_dated_series(n=50, seed=0)
    dates = np.full(50, fx.dates[0])
    sp = TemporalSplitter(random_state=0)
    with pytest.raises(ConstraintUnsatisfiableError):
        sp.split_result(fx.smiles, dates=dates)


def test_temporal_determinism():
    fx = make_dated_series(n=200, seed=0)
    sp = TemporalSplitter(random_state=0)
    r1 = sp.split_result(fx.smiles, dates=fx.dates)[0]
    r2 = sp.split_result(fx.smiles, dates=fx.dates)[0]
    assert np.array_equal(r1.train, r2.train)
    assert np.array_equal(r1.test, r2.test)


def test_temporal_rolling_mode_multiple_folds():
    fx = make_dated_series(n=300, seed=0)
    sp = TemporalSplitter(mode="rolling", n_windows=3, window="200D", random_state=0)
    assert sp.get_n_splits() == 3
    results = sp.split_result(fx.smiles, dates=fx.dates)
    assert len(results) == 3
    for i, r in enumerate(results):
        assert r.metadata["window_index"] == i
        assert len(r.train) + len(r.valid) + len(r.test) + len(r.discard) == 300
        if len(r.train) and len(r.test):
            assert fx.dates[r.train].max() <= fx.dates[r.test].min()


def test_temporal_expanding_train_grows():
    fx = make_dated_series(n=300, seed=0)
    sp = TemporalSplitter(mode="expanding", n_windows=3, window="200D", random_state=0)
    results = sp.split_result(fx.smiles, dates=fx.dates)
    sizes = [len(r.train) for r in results]
    assert sizes == sorted(sizes)  # non-decreasing train size across windows


def test_temporal_valid_cut_date():
    fx = make_dated_series(n=200, seed=0)
    sp = TemporalSplitter(random_state=0, valid_cut_date=str(fx.dates[100]))
    r = sp.split_result(fx.smiles, dates=fx.dates)[0]
    if len(r.valid):
        assert fx.dates[r.valid].min() >= np.datetime64(str(fx.dates[100]), "D")


# ---------------------------------------------------------------------------
# SIMPDSplitter
# ---------------------------------------------------------------------------


def test_simpd_requires_n_at_least_200():
    fx = make_two_clusters(n=100, seed=0)
    sp = SIMPDSplitter(random_state=0)
    with pytest.raises(ParameterError):
        sp.split_result(fx.smiles, y=np.zeros(100))


def test_simpd_unknown_target_key_rejected():
    with pytest.raises(ParameterError):
        SIMPDSplitter(targets={"not_a_real_target": 1.0})


@pytest.mark.skipif(not HAS_DEAP, reason="deap not installed")
def test_simpd_ga_runs_and_is_deterministic():
    fx = make_two_clusters(n=220, separation=0.9, seed=0)
    y = np.random.default_rng(0).normal(size=len(fx.smiles))
    kw = dict(population_size=24, n_generations=10, early_stop_patience=4, random_state=0)
    r1 = SIMPDSplitter(**kw).split_result(fx.smiles, y=y)[0]
    r2 = SIMPDSplitter(**kw).split_result(fx.smiles, y=y)[0]
    assert np.array_equal(r1.test, r2.test)
    assert len(r1.train) + len(r1.valid) + len(r1.test) == 220
    assert set(r1.metadata["achieved"]) == set(SIMPDSplitter._DEFAULT_TARGETS)


def test_simpd_missing_deap_raises_missing_dependency(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("deap"):
            raise ImportError("no deap")
        return real_import(name, *a, **kw)

    fx = make_two_clusters(n=220, seed=0)
    y = np.random.default_rng(0).normal(size=len(fx.smiles))
    monkeypatch.setattr(builtins, "__import__", fake_import)
    from chemsplit.exceptions import MissingDependencyError

    sp = SIMPDSplitter(population_size=10, n_generations=2, random_state=0)
    with pytest.raises(MissingDependencyError):
        sp.split_result(fx.smiles, y=y)


# ---------------------------------------------------------------------------
# SourceSplitter
# ---------------------------------------------------------------------------


def test_source_requires_source():
    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=10, seed=0)
    sp = SourceSplitter(random_state=0)
    with pytest.raises(InputError):
        sp.split_result(fx.smiles)


def test_source_groups_atomic_never_split():
    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=10, seed=0)
    source = [i % 12 for i in range(len(fx.smiles))]
    sp = SourceSplitter(source=source, random_state=0)
    r = sp.split_result(fx.smiles)[0]
    train_sources = {source[i] for i in r.train}
    test_sources = {source[i] for i in r.test}
    assert not (train_sources & test_sources)
    assert r.metadata["n_sources"] == 12


def test_source_small_source_policy_discard():
    n = 100
    # source 0 has 1 record (below min_source_size=2), everything else has >= 2
    source = [0] + [1 + (i % 20) for i in range(1, n)]
    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=10, seed=0)
    sp = SourceSplitter(source=source, min_source_size=2, small_source_policy="discard", random_state=0)
    r = sp.split_result(fx.smiles)[0]
    assert 0 not in r.train and 0 not in r.test
    assert 0 in r.discard


def test_source_missing_value_becomes_own_group():
    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=10, seed=0)
    source = [None if i == 0 else i % 10 for i in range(len(fx.smiles))]
    sp = SourceSplitter(source=source, random_state=0)
    r = sp.split_result(fx.smiles)[0]
    assert r.metadata["n_missing_source"] == 1


# ---------------------------------------------------------------------------
# PartySplitter
# ---------------------------------------------------------------------------


def test_party_given_leave_one_out():
    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=20, seed=0)
    party = [i % 4 for i in range(len(fx.smiles))]
    sp = PartySplitter(party=party, n_parties=4, held_out_party="each", random_state=0)
    assert sp.get_n_splits() == 4
    results = sp.split_result(fx.smiles)
    assert len(results) == 4
    for k, r in enumerate(results):
        assert r.metadata["held_out_party"] == k
        test_parties = {party[i] for i in r.test}
        train_parties = {party[i] for i in r.train}
        assert test_parties == {k}
        assert k not in train_parties


def test_party_n_parties_exceeds_n_raises():
    # n_parties > n can only be checked once X is known (at split time), not at construction.
    sp = PartySplitter(n_parties=1000, party=list(range(10)), random_state=0)
    with pytest.raises(ParameterError):
        sp.split_result(["C"] * 10)


def test_party_synthesis_modes_atomic_and_deterministic():
    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=20, seed=0)
    for synthesis in ("cluster", "dirichlet"):
        sp1 = PartySplitter(synthesis=synthesis, n_parties=4, held_out_party=0, random_state=0)
        sp2 = PartySplitter(synthesis=synthesis, n_parties=4, held_out_party=0, random_state=0)
        r1 = sp1.split_result(fx.smiles)[0]
        r2 = sp2.split_result(fx.smiles)[0]
        assert np.array_equal(r1.test, r2.test), synthesis
        assert r1.metadata["chemical_overlap_matrix"] is not None


def test_party_label_skew_requires_labels():
    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=20, seed=0)
    sp = PartySplitter(synthesis="label_skew", n_parties=4, held_out_party=0, random_state=0)
    with pytest.raises(InputError):
        sp.split_result(fx.smiles)
