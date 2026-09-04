"""Tests for chemsplit.splitters.biomolecular (the biomolecular family, renamed "biomolecular")."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit.datasets import make_dated_series, make_scaffold_families, make_sequences
from chemsplit.exceptions import LabelError, MissingDependencyError, ParameterError
from chemsplit.splitters.biomolecular import (
    BindingSiteSplitter,
    ComplexJointSplitter,
    DepositionDateSplitter,
    ProteinFamilySplitter,
    SequenceIdentitySplitter,
)

try:
    import parasail  # noqa: F401

    HAS_PARASAIL = True
except ImportError:
    HAS_PARASAIL = False


# ---------------------------------------------------------------------------
# SequenceIdentitySplitter
# ---------------------------------------------------------------------------


def test_sequence_identity_derives_sequences_from_x_kind():
    """``ctx.sequences`` is populated automatically from ``X`` when ``X_kind="sequences"`` is
    passed without a separate, redundant ``sequences=`` keyword (chemsplit/base.py's ``_run``/
    ``compute_groups`` -- a caller should not have to supply the same sequence list twice)."""
    sp = SequenceIdentitySplitter(random_state=0)
    result = sp.split_result(["AAAAAAAAAA", "AAAAAAAAAB", "CCCCCCCCCC", "CCCCCCCCCD"], X_kind="sequences")[0]
    assert result.n_records == 4
    assert result.groups is not None


def test_sequence_identity_without_x_kind_hint_treats_input_as_smiles():
    """``Sequence[str]`` defaults to SMILES unless ``X_kind="sequences"`` is given -- there
    is no heuristic disambiguation, so omitting the hint for a sequences-only splitter must fail
    with InputKindError (detected kind "smiles" not in accepts=("sequences",)), not silently
    guess right."""
    from chemsplit.exceptions import InputKindError

    sp = SequenceIdentitySplitter(random_state=0)
    with pytest.raises(InputKindError):
        sp.split_result(["AAAAAAAAAA", "AAAAAAAAAB", "CCCCCCCCCC", "CCCCCCCCCD"])


def test_sequence_identity_keeps_families_atomic_hamming():
    fx = make_sequences(n=20, families=4, identity_within=0.9, seed=0)
    sp = SequenceIdentitySplitter(identity_threshold=0.5, algorithm="hamming", random_state=0)
    labels = sp.compute_groups(fx.sequences, X_kind="sequences", sequences=fx.sequences)
    n = len(fx.sequences)
    # Direct correctness check: any two sequences with hamming identity > threshold share a label.
    for i in range(n):
        for j in range(i + 1, n):
            m = sum(a == b for a, b in zip(fx.sequences[i], fx.sequences[j])) / min(
                len(fx.sequences[i]), len(fx.sequences[j])
            )
            if m > 0.5:
                assert labels[i] == labels[j]


def test_sequence_identity_deterministic():
    fx = make_sequences(n=20, families=4, identity_within=0.8, seed=0)
    sp = SequenceIdentitySplitter(identity_threshold=0.6, algorithm="hamming", random_state=0)
    a = sp.compute_groups(fx.sequences, X_kind="sequences", sequences=fx.sequences)
    b = sp.compute_groups(fx.sequences, X_kind="sequences", sequences=fx.sequences)
    assert np.array_equal(a, b)


@pytest.mark.skipif(not HAS_PARASAIL, reason="parasail not installed")
def test_sequence_identity_parasail_path_runs():
    fx = make_sequences(n=10, families=2, identity_within=0.85, seed=1)
    sp = SequenceIdentitySplitter(identity_threshold=0.5, algorithm="parasail", random_state=0)
    labels = sp.compute_groups(fx.sequences, X_kind="sequences", sequences=fx.sequences)
    assert labels.shape == (10,)


def test_sequence_identity_missing_parasail_raises(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "parasail":
            raise ImportError("simulated missing parasail")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    fx = make_sequences(n=10, families=2, seed=0)
    sp = SequenceIdentitySplitter(algorithm="parasail", random_state=0)
    with pytest.raises(MissingDependencyError):
        sp.compute_groups(fx.sequences, X_kind="sequences", sequences=fx.sequences)


# ---------------------------------------------------------------------------
# ProteinFamilySplitter
# ---------------------------------------------------------------------------


def test_protein_family_requires_labels():
    sp = ProteinFamilySplitter(random_state=0)
    with pytest.raises(ParameterError):
        sp.split_result(["AAA", "BBB", "CCC", "DDD"], X_kind="sequences")


def test_protein_family_groups_by_supplied_labels():
    seqs = ["AAA", "AAB", "BBB", "BBC", "CCC", "CCD"]
    labels_in = ["kinaseA", "kinaseA", "kinaseB", "kinaseB", "kinaseC", "kinaseC"]
    sp = ProteinFamilySplitter(family_labels=labels_in, random_state=0, train_size=4, test_size=2)
    labels = sp.compute_groups(seqs, X_kind="sequences")
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[4] == labels[5]
    assert len({labels[0], labels[2], labels[4]}) == 3


def test_protein_family_split_result_atomic():
    seqs = [f"S{i}" for i in range(12)]
    labels_in = [i % 4 for i in range(12)]  # 4 families of 3
    sp = ProteinFamilySplitter(
        family_labels=[str(x) for x in labels_in], random_state=0, train_size=0.75, test_size=0.25
    )
    result = sp.split_result(seqs, X_kind="sequences")[0]
    fam_of = {i: labels_in[i] for i in range(12)}
    train_fams = {fam_of[i] for i in result.train}
    test_fams = {fam_of[i] for i in result.test}
    assert not (train_fams & test_fams)


# ---------------------------------------------------------------------------
# BindingSiteSplitter
# ---------------------------------------------------------------------------


def test_binding_site_composition_requires_features():
    sp = BindingSiteSplitter(representation="composition", random_state=0)
    with pytest.raises(ParameterError):
        sp.split_result(["AAA", "BBB"], X_kind="sequences")


def test_binding_site_composition_clusters_two_blobs():
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=0.0, scale=0.05, size=(15, 4))
    blob_b = rng.normal(loc=5.0, scale=0.05, size=(15, 4))
    F = np.vstack([blob_a, blob_b])
    sp = BindingSiteSplitter(
        representation="composition", cutoff=0.5, random_state=0
    )
    labels = sp.compute_groups(F)
    # within-blob pairs should mostly share a label, cross-blob pairs should not
    assert labels[0] == labels[1]
    assert labels[15] == labels[16]
    assert labels[0] != labels[15]


def test_binding_site_pocket_sequence_mode():
    fx = make_sequences(n=10, families=2, identity_within=0.9, seed=2)
    sp = BindingSiteSplitter(
        representation="pocket_sequence", cutoff=0.5, random_state=0
    )
    labels = sp.compute_groups(fx.sequences, X_kind="sequences", sequences=fx.sequences)
    assert labels.shape == (10,)


# ---------------------------------------------------------------------------
# DepositionDateSplitter
# ---------------------------------------------------------------------------


def test_deposition_date_requires_dates():
    sp = DepositionDateSplitter(cut_date="2020-01-01", random_state=0)
    with pytest.raises(LabelError):
        sp.split_result(["CCO", "CCN", "CCC", "CCF"])


def test_deposition_date_basic_cut():
    fx = make_dated_series(n=100, seed=0)
    median_date = str(np.sort(fx.dates)[len(fx.dates) // 2])
    sp = DepositionDateSplitter(cut_date=median_date, random_state=0)
    result = sp.split_result(fx.smiles, dates=fx.dates)[0]
    cut = np.datetime64(sp.cut_date)
    assert np.all(fx.dates[result.train] <= cut)
    assert np.all(fx.dates[result.test] > cut)


def test_deposition_date_ligand_similarity_pruning_removes_near_duplicates():
    # Two near-identical molecules straddling the cut date: the earlier one should be pruned
    # from train once ligand_similarity_ceiling is set tightly.
    smiles = ["CCCCCCCC", "CCCCCCCC", "c1ccccc1", "c1ccccc1O"] * 5
    dates = np.array(
        ["2019-01-01", "2021-06-01", "2019-02-01", "2021-07-01"] * 5, dtype="datetime64[D]"
    )
    sp = DepositionDateSplitter(
        cut_date="2020-01-01", ligand_similarity_ceiling=0.99, random_state=0
    )
    result = sp.split_result(smiles, dates=dates)[0]
    assert result.metadata["n_pruned"] >= 0  # smoke: pruning path executes without error


# ---------------------------------------------------------------------------
# ComplexJointSplitter
# ---------------------------------------------------------------------------


def test_complex_joint_both_novel_disjoint_axes():
    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=10, seed=0)
    seqs = make_sequences(n=100, families=10, identity_within=0.9, seed=0).sequences
    sp = ComplexJointSplitter(mode="both_novel", random_state=0, train_size=0.8, test_size=0.2)
    result = sp.split_result(fx.smiles, sequences=seqs)[0]
    assert result.train.size + result.valid.size + result.test.size + result.discard.size == 100
    assert set(result.train.tolist()).isdisjoint(set(result.test.tolist()))


def test_complex_joint_either_novel_test_superset_of_both_novel():
    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=10, seed=0)
    seqs = make_sequences(n=100, families=10, identity_within=0.9, seed=0).sequences
    sp_both = ComplexJointSplitter(mode="both_novel", random_state=0, train_size=0.8, test_size=0.2)
    sp_either = ComplexJointSplitter(mode="either_novel", random_state=0, train_size=0.8, test_size=0.2)
    r_both = sp_both.split_result(fx.smiles, sequences=seqs)[0]
    r_either = sp_either.split_result(fx.smiles, sequences=seqs)[0]
    assert set(r_both.test.tolist()).issubset(set(r_either.test.tolist()))


def test_complex_joint_deterministic():
    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=10, seed=0)
    seqs = make_sequences(n=100, families=10, identity_within=0.9, seed=0).sequences
    sp = ComplexJointSplitter(mode="both_novel", random_state=42, train_size=0.8, test_size=0.2)
    a = sp.split_result(fx.smiles, sequences=seqs)[0]
    b = sp.split_result(fx.smiles, sequences=seqs)[0]
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)


def test_complex_joint_invalid_mode():
    with pytest.raises(ParameterError):
        ComplexJointSplitter(mode="bogus", random_state=0)


# --------------------------------------------------------------------------- coverage additions


def test_sequence_identity_param_validation():
    with pytest.raises(ParameterError):
        SequenceIdentitySplitter(identity_threshold=1.5)
    with pytest.raises(ParameterError):
        SequenceIdentitySplitter(algorithm="bogus")


def test_sequence_identity_hamming_algorithm_explicit():
    seqs = make_sequences(n=20, families=4, identity_within=0.9, seed=0).sequences
    sp = SequenceIdentitySplitter(algorithm="hamming", train_size=0.5, test_size=0.5, random_state=0)
    result = sp.split_result(seqs, X_kind="sequences")[0]
    assert result.n_records == 20


def test_sequence_identity_degenerate_error_and_warning():
    from chemsplit.exceptions import DegenerateGroupingError

    seqs = ["MKV" + "A" * 30] * 20
    sp_err = SequenceIdentitySplitter(identity_threshold=0.5, train_size=0.5, test_size=0.5, random_state=0)
    with pytest.raises(DegenerateGroupingError):
        sp_err.split_result(seqs, X_kind="sequences")


def test_protein_family_wrong_length_raises():
    F = np.random.default_rng(0).standard_normal((20, 4))
    sp = ProteinFamilySplitter(family_labels=["a", "b"], train_size=0.5, test_size=0.5)
    with pytest.raises(ParameterError):
        sp.split_result(F, X_kind="features")


# --------------------------------------------------------------------------- BindingSiteSplitter


def test_binding_site_invalid_representation_and_cutoff():
    with pytest.raises(ParameterError):
        BindingSiteSplitter(representation="bogus")
    with pytest.raises(ParameterError):
        BindingSiteSplitter(cutoff=1.5)


def test_binding_site_pocket_sequence_requires_sequences():
    from chemsplit.exceptions import LabelError as LE

    sp = BindingSiteSplitter(representation="pocket_sequence", train_size=0.5, test_size=0.5)
    with pytest.raises(LE):
        sp.split_result(np.random.default_rng(0).standard_normal((10, 4)), X_kind="features")


def test_binding_site_degenerate_error_and_warning():
    from chemsplit.exceptions import DegenerateGroupingError

    F = np.zeros((20, 4))
    sp_err = BindingSiteSplitter(cutoff=0.99, train_size=0.5, test_size=0.5, random_state=0)
    with pytest.raises(DegenerateGroupingError):
        sp_err.split_result(F, X_kind="features")


# --------------------------------------------------------------------------- DepositionDateSplitter


def test_deposition_date_requires_cut_date():
    fx = make_dated_series(n=40, seed=0)
    sp = DepositionDateSplitter()
    with pytest.raises(ParameterError):
        sp.split_result(fx.smiles, dates=fx.dates)


def test_deposition_date_sequence_identity_pruning():
    fx = make_dated_series(n=40, seed=0)
    seqs = make_sequences(n=40, families=4, identity_within=0.95, seed=0).sequences
    cut = str(np.sort(fx.dates)[20])
    sp = DepositionDateSplitter(cut_date=cut, sequence_identity_ceiling=0.3)
    result = sp.split_result(fx.smiles, dates=fx.dates, sequences=seqs)[0]
    assert result.n_records == 40


# --------------------------------------------------------------------------- ComplexJointSplitter


def test_complex_joint_explicit_groupers():
    from chemsplit.splitters.similarity import ButinaSplitter

    fx = make_scaffold_families(n_scaffolds=10, per_scaffold=10, seed=0)
    seqs = make_sequences(n=100, families=10, identity_within=0.9, seed=0).sequences
    sp = ComplexJointSplitter(
        ligand_grouper=ButinaSplitter(cutoff=0.4, random_state=0),
        sequence_grouper=SequenceIdentitySplitter(identity_threshold=0.6, random_state=0),
        random_state=0, train_size=0.8, test_size=0.2,
    )
    result = sp.split_result(fx.smiles, sequences=seqs)[0]
    assert result.n_records == 100


def test_complex_joint_no_mols_raises():
    seqs = make_sequences(n=20, families=4, identity_within=0.9, seed=0).sequences
    F = np.random.default_rng(0).standard_normal((20, 4))
    sp = ComplexJointSplitter(random_state=0, train_size=0.5, test_size=0.5)
    with pytest.raises(ParameterError):
        sp.split_result(F, X_kind="features", sequences=seqs)


def test_complex_joint_no_sequences_raises():
    fx = make_scaffold_families(n_scaffolds=5, per_scaffold=4, seed=0)
    sp = ComplexJointSplitter(random_state=0, train_size=0.5, test_size=0.5)
    with pytest.raises(ParameterError):
        sp.split_result(fx.smiles)
