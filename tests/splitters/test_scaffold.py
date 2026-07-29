"""Tests for the ``scaffold`` splitter family (chemsplit/splitters/scaffold.py)."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from chemsplit.exceptions import (
    ConstraintUnsatisfiableError,
    DegenerateGroupingError,
    ParameterError,
)
from chemsplit.splitters.scaffold import (
    ActivityCliffSplitter,
    GenericScaffoldSplitter,
    MatchedMolecularSeriesSplitter,
    MurckoScaffoldSplitter,
    RingSystemSplitter,
    ScaffoldTreeSplitter,
)

# A benzene-core family (5) + a pyridine-core family (5) + 2 acyclic molecules.
BENZENE_FAMILY = [
    "c1ccccc1C",
    "c1ccccc1CC",
    "c1ccccc1CCC",
    "c1ccccc1CCCC",
    "c1ccccc1CCCCC",
]
PYRIDINE_FAMILY = [
    "c1ccncc1C",
    "c1ccncc1CC",
    "c1ccncc1CCC",
    "c1ccncc1CCCC",
    "c1ccncc1CCCCC",
]
ACYCLIC = ["CCCCCC", "CCCCCCC"]

MIXED_SMILES = BENZENE_FAMILY + PYRIDINE_FAMILY + ACYCLIC  # 12 records


def test_murcko_groups_by_scaffold():
    sp = MurckoScaffoldSplitter(on_empty_scaffold="discard")
    groups = sp.compute_groups(MIXED_SMILES)
    # benzene family all share one group id, pyridine family another, distinct from benzene's.
    benzene_groups = set(groups[:5].tolist())
    pyridine_groups = set(groups[5:10].tolist())
    assert len(benzene_groups) == 1
    assert len(pyridine_groups) == 1
    assert benzene_groups != pyridine_groups


def test_murcko_on_empty_scaffold_own_group_vs_shared_group():
    # Mix in one ring-bearing molecule so "every acyclic record singleton" isn't itself the
    # degenerate (n_groups == n_records) edge case -- that scenario is covered separately by
    # test_murcko_all_acyclic_raises_degenerate.
    mols = BENZENE_FAMILY + ACYCLIC + ["CCC"]  # 5 same-scaffold + 3 distinct acyclic chains
    sp_own = MurckoScaffoldSplitter(on_empty_scaffold="own_group")
    g_own = sp_own.compute_groups(mols)
    assert len(set(g_own[5:].tolist())) == 3  # each acyclic molecule its own group

    sp_shared = MurckoScaffoldSplitter(on_empty_scaffold="shared_group")
    g_shared = sp_shared.compute_groups(mols)
    assert len(set(g_shared[5:].tolist())) == 1  # all acyclic molecules share one group


def test_murcko_all_acyclic_raises_degenerate():
    sp = MurckoScaffoldSplitter(on_empty_scaffold="own_group", train_size=0.7, test_size=0.3, random_state=0)
    with pytest.raises(DegenerateGroupingError):
        list(sp.split_result(ACYCLIC * 3))  # every record its own group -> degenerate


def test_murcko_split_result_end_to_end():
    sp = MurckoScaffoldSplitter(train_size=0.6, test_size=0.4, random_state=0, on_empty_scaffold="discard")
    [result] = sp.split_result(MIXED_SMILES)
    assert result.n_records == len(MIXED_SMILES)
    assert result.splitter_id == "murcko_scaffold"
    assert set(result.train.tolist()) | set(result.valid.tolist()) | set(
        result.test.tolist()
    ) | set(result.discard.tolist()) == set(range(len(MIXED_SMILES)))
    assert "n_groups" in result.metadata
    assert "scaffold_smiles" in result.metadata
    # params round-trips through JSON and includes base params.
    assert "n_splits" in result.params
    assert "train_size" in result.params
    assert result.to_json()  # I4 invariant already checked by SplitResult.__post_init__


def test_murcko_deterministic_same_seed():
    sp1 = MurckoScaffoldSplitter(train_size=0.6, test_size=0.4, random_state=42, on_empty_scaffold="discard")
    sp2 = MurckoScaffoldSplitter(train_size=0.6, test_size=0.4, random_state=42, on_empty_scaffold="discard")
    [r1] = sp1.split_result(MIXED_SMILES)
    [r2] = sp2.split_result(MIXED_SMILES)
    assert np.array_equal(r1.train, r2.train)
    assert np.array_equal(r1.test, r2.test)


def test_generic_scaffold_merges_heteroatom_variants():
    # Benzene vs pyridine differ only by one heteroatom -> same generic framework. (n=3, not 2,
    # so the default 0.8/0.2 size resolution doesn't round test_size down to 0 -- compute_groups
    # still calls resolve_sizes internally even though grouping itself ignores sizes.)
    sp = GenericScaffoldSplitter(on_empty_scaffold="discard")
    groups = sp.compute_groups(["c1ccccc1C", "c1ccncc1C", "c1ccc2ccccc2c1"])  # + naphthalenyl
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_generic_scaffold_rejects_include_chirality():
    with pytest.raises(ParameterError):
        GenericScaffoldSplitter(include_chirality=True)


def test_scaffold_tree_level_zero_matches_murcko():
    mols = BENZENE_FAMILY + PYRIDINE_FAMILY
    sp_tree = ScaffoldTreeSplitter(level=0, on_empty_scaffold="discard")
    sp_murcko = MurckoScaffoldSplitter(on_empty_scaffold="discard")
    g_tree = sp_tree.compute_groups(mols)
    g_murcko = sp_murcko.compute_groups(mols)
    assert np.array_equal(g_tree, g_murcko)


def test_scaffold_tree_level_param_validation():
    with pytest.raises(ParameterError):
        ScaffoldTreeSplitter(level=-1)
    with pytest.raises(ParameterError):
        ScaffoldTreeSplitter(level=5, max_rings=3)
    with pytest.raises(ParameterError):
        ScaffoldTreeSplitter(prune_rule="not_a_rule")


def test_ring_system_any_shared_transitive_union():
    # Two molecules sharing a benzene ring end up in the same component even if their overall
    # scaffolds differ (naphthalene-like fusion vs a lone benzene substituent). A 4th, ring-
    # disjoint molecule keeps the largest-component fraction below the 95% degenerate ceiling.
    sp = RingSystemSplitter(linkage="any_shared", on_empty_scaffold="discard")
    smiles = [
        "c1ccccc1CC1CCCCC1",
        "c1ccccc1C",
        "C1CCCCC1CC1CCCCC1",
        "c1ccc2ncccc2c1",  # quinoline: shares no ring system with the benzene/cyclohexane trio
    ]
    groups = sp.compute_groups(smiles)
    # record 0 (benzene+cyclohexane) shares benzene with record 1, and cyclohexane with record 2.
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_ring_system_all_shared_requires_identical_multiset():
    sp = RingSystemSplitter(linkage="all_shared", on_empty_scaffold="discard")
    smiles = ["c1ccccc1CC1CCCCC1", "c1ccccc1CC1CCCCC1", "c1ccccc1C"]
    groups = sp.compute_groups(smiles)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_ring_system_csk_key_mode_runs():
    sp = RingSystemSplitter(key="csk", on_empty_scaffold="discard")
    groups = sp.compute_groups(
        ["c1ccccc1C", "c1ccncc1C", "C1CCCCC1C", "c1ccc2ncccc2c1CCCCCCCC"]
    )
    assert groups.shape == (4,)


def test_matched_molecular_series_groups_homologous_series():
    # Each 5-molecule homologous alkyl-chain family collapses into its own single series via
    # shared constant contexts (verified: MMPA fragmentation groups the whole family into one
    # component) -- two ring-disjoint families keep the overall grouping non-degenerate (2 groups
    # over 10 records) while still proving the collapse-within-a-family behaviour.
    sp = MatchedMolecularSeriesSplitter(min_series_size=2, on_empty_scaffold="discard")
    groups = sp.compute_groups(BENZENE_FAMILY + PYRIDINE_FAMILY)
    assert len(set(groups[:5].tolist())) == 1
    assert len(set(groups[5:].tolist())) == 1
    assert groups[0] != groups[5]


def test_matched_molecular_series_discard_boundary_mode():
    sp = MatchedMolecularSeriesSplitter(
        enforce="discard_boundary",
        min_series_size=2,
        train_size=0.6,
        test_size=0.4,
        random_state=0,
        on_empty_scaffold="discard",
    )
    [result] = sp.split_result(BENZENE_FAMILY + PYRIDINE_FAMILY)
    total = result.train.size + result.valid.size + result.test.size + result.discard.size
    assert total == len(BENZENE_FAMILY) + len(PYRIDINE_FAMILY)
    assert result.metadata["boundary_discarded"] >= 0


def test_matched_molecular_series_max_cuts_validation():
    with pytest.raises(ParameterError):
        MatchedMolecularSeriesSplitter(max_cuts=4)
    with pytest.raises(ParameterError):
        MatchedMolecularSeriesSplitter(enforce="bogus")


def test_activity_cliff_finds_and_flags_cliff_pairs():
    # Two near-identical molecules (methyl vs ethyl homolog) with a large label gap -> a cliff.
    smiles = ["c1ccccc1C", "c1ccccc1CC", "CCCCCCCCCCCC", "CCCCCCCCCCCCC"]
    y = np.array([1.0, 5.0, 1.0, 1.05])  # log-scale: pair (0,1) has a 4.0 gap; log10(10)=1.0
    sp = ActivityCliffSplitter(
        similarity_threshold=0.3,
        fold_change_threshold=10.0,
        y_scale="log",
        train_size=0.5,
        test_size=0.5,
        random_state=0,
    )
    [result] = sp.split_result(smiles, y=y)
    assert result.metadata["n_cliff_pairs"] >= 1
    assert any(result.metadata["cliff_mask"])
    assert result.groups is None  # not group-forming


def test_activity_cliff_raises_when_no_cliffs_found():
    smiles = ["CCCC", "CCCCC", "CCCCCC"]
    y = np.array([1.0, 1.01, 1.02])  # no meaningful gap
    sp = ActivityCliffSplitter(similarity_threshold=0.99, fold_change_threshold=1000.0)
    with pytest.raises(ConstraintUnsatisfiableError):
        sp.split_result(smiles, y=y)


def test_activity_cliff_requires_labels():
    from chemsplit.exceptions import LabelError

    sp = ActivityCliffSplitter()
    with pytest.raises(LabelError):
        sp.split_result(["CCCC", "CCCCC"])


def test_activity_cliff_param_validation():
    with pytest.raises(ParameterError):
        ActivityCliffSplitter(similarity_threshold=1.5)
    with pytest.raises(ParameterError):
        ActivityCliffSplitter(fold_change_threshold=1.0)


def test_all_classes_accept_only_molecules():
    from chemsplit.exceptions import InputKindError

    X = np.zeros((5, 4))
    for cls in (MurckoScaffoldSplitter, GenericScaffoldSplitter, ScaffoldTreeSplitter):
        with pytest.raises(InputKindError):
            list(cls().split(X))


def test_get_params_includes_base_and_shared_params():
    sp = MurckoScaffoldSplitter(include_chirality=True, train_size=0.7, test_size=0.3, n_jobs=2)
    params = sp.get_params()
    for key in ("include_chirality", "on_empty_scaffold", "size_tolerance", "group_assignment",
                "n_splits", "train_size", "valid_size", "test_size", "random_state", "n_jobs",
                "verbose"):
        assert key in params, f"{key} missing from get_params() -- sklearn compatibility broken"
    assert params["include_chirality"] is True
    assert params["n_jobs"] == 2


def test_sklearn_clone_compatible():
    from sklearn.base import clone

    sp = MurckoScaffoldSplitter(train_size=0.7, test_size=0.3, random_state=7)
    cloned = clone(sp)
    assert cloned.get_params() == sp.get_params()
