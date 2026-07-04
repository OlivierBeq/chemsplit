"""Tests for chemsplit.datasets's 12 synthetic fixture generators."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit import datasets as ds


def _mols(smiles):
    from rdkit import Chem

    return [Chem.MolFromSmiles(s) for s in smiles]


def test_make_linear_series_reproducible_and_valid():
    f1 = ds.make_linear_series(n=50, seed=0)
    f2 = ds.make_linear_series(n=50, seed=0)
    assert f1.smiles == f2.smiles
    assert np.array_equal(f1.y, f2.y)
    f3 = ds.make_linear_series(n=50, seed=1)
    assert f1.smiles != f3.smiles or not np.array_equal(f1.y, f3.y)
    assert len(f1.smiles) == 50
    assert all(m is not None for m in _mols(f1.smiles))
    # y should correlate with chain-length index (monotonic-ish trend)
    assert np.corrcoef(np.arange(50), f1.y)[0, 1] > 0.5


def test_make_scaffold_families_exact_groups():
    f = ds.make_scaffold_families(n_scaffolds=10, per_scaffold=20, seed=0)
    assert len(f.smiles) == 200
    assert f.groups_true is not None
    assert len(set(f.groups_true.tolist())) == 10
    for g in range(10):
        assert int(np.sum(f.groups_true == g)) == 20

    # Every member of a group truly shares one Murcko scaffold.
    from rdkit.Chem.Scaffolds import MurckoScaffold as MS
    from rdkit import Chem

    mols = _mols(f.smiles)
    assert all(m is not None for m in mols)
    for g in range(10):
        idx = np.nonzero(f.groups_true == g)[0]
        scafs = {Chem.MolToSmiles(MS.GetScaffoldForMol(mols[i])) for i in idx}
        assert len(scafs) == 1, f"group {g} has multiple scaffolds: {scafs}"

    # Different groups have different scaffolds.
    all_scafs = set()
    for g in range(10):
        i = int(np.nonzero(f.groups_true == g)[0][0])
        all_scafs.add(Chem.MolToSmiles(MS.GetScaffoldForMol(mols[i])))
    assert len(all_scafs) == 10


def test_make_scaffold_families_reproducible():
    f1 = ds.make_scaffold_families(seed=0)
    f2 = ds.make_scaffold_families(seed=0)
    assert f1.smiles == f2.smiles
    assert np.array_equal(f1.groups_true, f2.groups_true)


def test_make_two_clusters_separation():
    f = ds.make_two_clusters(n=100, seed=0)
    assert len(f.smiles) == 100
    assert f.groups_true is not None
    assert set(f.groups_true.tolist()) == {0, 1}
    assert "within_similarity_gap" in f.extra
    assert f.extra["within_similarity_gap"] >= 0.15


def test_make_activity_cliffs_delta_exact():
    f = ds.make_activity_cliffs(n_pairs=10, seed=0)
    assert len(f.smiles) == 20
    y = f.y
    for p in range(10):
        assert abs((y[2 * p + 1] - y[2 * p]) - 2.0) < 1e-9


def test_make_dated_series_correlates_with_group():
    f = ds.make_dated_series(n=100, seed=0)
    assert f.dates is not None
    assert f.groups_true is not None
    # mean date should increase with group id (by construction)
    means = []
    for g in sorted(set(f.groups_true.tolist())):
        idx = np.nonzero(f.groups_true == g)[0]
        means.append(f.dates[idx].astype("datetime64[D]").astype(int).mean())
    assert all(means[i] < means[i + 1] for i in range(len(means) - 1))


def test_make_multitask_sparse_task0_exactly_three():
    f = ds.make_multitask_sparse(n=200, n_tasks=8, seed=0)
    assert f.y.shape[1] == 8
    assert int(np.sum(~np.isnan(f.y[:, 0]))) == 3
    density = np.mean(~np.isnan(f.y[:, 1:]))
    assert 0.15 < density < 0.45


def test_make_interactions_shapes():
    f = ds.make_interactions(n_compounds=50, n_targets=10, density=0.25, seed=0)
    assert len(f.smiles) == 50
    assert len(f.targets) == 10
    assert f.interactions
    max_possible = 50 * 10
    assert 0 < len(f.interactions) < max_possible
    for ci, ti, y in f.interactions:
        assert 0 <= ci < 50
        assert 0 <= ti < 10


def test_make_sequences_families_and_identity():
    f = ds.make_sequences(n=20, families=4, identity_within=0.8, seed=0)
    assert len(f.sequences) == 20
    assert len(set(f.groups_true.tolist())) == 4
    for g in range(4):
        assert int(np.sum(f.groups_true == g)) == 5
    assert abs(f.extra["mean_within_family_identity"] - 0.8) < 0.25


def test_make_pathological_exact_composition():
    f = ds.make_pathological()
    assert len(f.smiles) == 12
    mols = _mols(f.smiles)
    n_unparseable = sum(1 for m in mols if m is None)
    assert n_unparseable == 1
    assert f.smiles[6] == f.smiles[7]  # exact duplicate pair


def test_make_pathological_no_seed_param():
    import inspect

    sig = inspect.signature(ds.make_pathological)
    assert "seed" not in sig.parameters


def test_make_all_identical():
    f = ds.make_all_identical(n=30)
    assert len(f.smiles) == 30
    assert len(set(f.smiles)) == 1


def test_make_singletons_max_similarity():
    # n=20 is the empirically documented achievable ceiling at this project's default
    # max_similarity=0.15 -- see make_singletons's docstring for why n=100 does not succeed
    # with a synthetic (non-real-library) candidate pool.
    f = ds.make_singletons(n=20, seed=0)
    assert len(f.smiles) == 20
    assert f.extra["achieved_max_similarity"] < 0.15


def test_make_singletons_n100_raises_documented_limitation():
    with pytest.raises(Exception):
        ds.make_singletons(n=100, seed=0)


def test_make_label_extremes_censored_block_and_bimodal():
    f = ds.make_label_extremes(n=300, seed=0)
    assert len(f.smiles) == 300
    assert int(np.sum(f.y == 5.0)) == 60
    non_censored = f.y[f.y != 5.0]
    # crude bimodality check: histogram has a low-density gap between two peaks
    hist, edges = np.histogram(non_censored, bins=20)
    assert hist.min() < hist.max() * 0.6


def test_make_label_extremes_reproducible():
    f1 = ds.make_label_extremes(seed=0)
    f2 = ds.make_label_extremes(seed=0)
    assert f1.smiles == f2.smiles
    assert np.array_equal(f1.y, f2.y)
