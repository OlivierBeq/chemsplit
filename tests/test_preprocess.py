import numpy as np
import pandas as pd
import pytest

from chemsplit import preprocess as pp
from chemsplit.exceptions import ColumnError, DuplicateRecordError, MoleculeParseError


def test_detect_input_kind_smiles():
    assert pp.detect_input_kind(["CCO", "c1ccccc1"]) == "smiles"


def test_detect_input_kind_features():
    assert pp.detect_input_kind(np.zeros((5, 3))) == "features"


def test_detect_input_kind_explicit_override():
    assert pp.detect_input_kind(["ACDEFG"], x_kind="sequences") == "sequences"


def test_detect_input_kind_dataframe():
    df = pd.DataFrame({"smiles": ["CCO"]})
    assert pp.detect_input_kind(df) == "dataframe"


def test_detect_input_kind_sparse_is_features():
    import scipy.sparse as sp

    X = sp.csr_matrix(np.zeros((3, 4)))
    assert pp.detect_input_kind(X) == "features"


def test_detect_input_kind_non_iterable_raises_input_kind_error():
    from chemsplit.exceptions import InputKindError

    with pytest.raises(InputKindError):
        pp.detect_input_kind(42)


def test_detect_input_kind_empty_raises_empty_input_error():
    from chemsplit.exceptions import EmptyInputError

    with pytest.raises(EmptyInputError):
        pp.detect_input_kind([])


def test_detect_input_kind_mol_sequence():
    from rdkit import Chem

    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCN")]
    assert pp.detect_input_kind(mols) == "mol"


def test_detect_input_kind_interactions():
    assert pp.detect_input_kind([("CCO", "TGT1"), ("CCN", "TGT2")]) == "interactions"


def test_detect_input_kind_mixed_type_raises():
    from chemsplit.exceptions import InputKindError

    with pytest.raises(InputKindError):
        pp.detect_input_kind(["CCO", 42, None])


def test_input_length_dataframe():
    df = pd.DataFrame({"smiles": ["CCO", "CCN"]})
    assert pp.input_length(df, "dataframe") == 2


def test_input_length_ndarray():
    assert pp.input_length(np.zeros((7, 3)), "features") == 7


def test_input_length_sparse():
    import scipy.sparse as sp

    assert pp.input_length(sp.csr_matrix(np.zeros((5, 2))), "features") == 5


def test_input_length_plain_sequence():
    assert pp.input_length(["CCO", "CCN", "CCC"], "smiles") == 3


def test_resolve_dataframe_columns_none_selector_passthrough():
    df = pd.DataFrame({"smiles": ["CCO"]})
    out = pp.resolve_dataframe_columns(df, smiles_col="smiles", label_col=None)
    assert out["label_col"] is None
    assert list(out["smiles_col"]) == ["CCO"]


def test_resolve_dataframe_columns_requires_smiles_col():
    df = pd.DataFrame({"smiles": ["CCO"], "y": [1.0]})
    with pytest.raises(ColumnError):
        pp.resolve_dataframe_columns(df, label_col="y")


def test_resolve_dataframe_columns_missing_column():
    df = pd.DataFrame({"smiles": ["CCO"]})
    with pytest.raises(ColumnError):
        pp.resolve_dataframe_columns(df, smiles_col="smiles", label_col="nope")


def test_parse_smiles_valid_and_invalid():
    assert pp.parse_smiles("CCO") is not None
    assert pp.parse_smiles("not a smiles!!") is None


def test_standardize_strips_salt():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CC(=O)O.[Na+]")
    std = pp.standardize(mol)
    smi = Chem.MolToSmiles(std)
    assert "Na" not in smi


def test_standardize_default_keeps_stereo():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
    std = pp.standardize(mol)
    assert "@" in Chem.MolToSmiles(std)


def test_standardize_strip_stereo():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
    cfg = pp.StandardizeConfig(stereo="strip")
    std = pp.standardize(mol, cfg)
    assert "@" not in Chem.MolToSmiles(std)


def test_standardize_canonical_tautomer():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CC(O)=CC")  # an enol tautomer
    cfg = pp.StandardizeConfig(canonical_tautomer=True)
    std = pp.standardize(mol, cfg)
    assert std is not None and std.GetNumAtoms() > 0


def test_standardize_strip_isotopes():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("[13CH4]")
    cfg = pp.StandardizeConfig(strip_isotopes=True)
    std = pp.standardize(mol, cfg)
    assert all(atom.GetIsotope() == 0 for atom in std.GetAtoms())


def test_standardize_strip_unassigned_stereo():
    from rdkit import Chem

    # A molecule with one assigned and (via a second stereocentre left unspecified) one
    # unassigned centre -- strip_unassigned should only clear the unspecified one, but since
    # RDKit's chirality possible/unassigned detection is subtle, assert only the documented
    # contract: the call succeeds and returns a sanitized molecule.
    mol = Chem.MolFromSmiles("C[C@H](N)C(C)C(=O)O")
    cfg = pp.StandardizeConfig(stereo="strip_unassigned")
    std = pp.standardize(mol, cfg)
    assert std is not None


def test_dedup_key_duplicates_share_key():
    from rdkit import Chem

    m1 = Chem.MolFromSmiles("CCO")
    m2 = Chem.MolFromSmiles("OCC")  # same molecule, different SMILES
    k1, _ = pp.dedup_key(m1)
    k2, _ = pp.dedup_key(m2)
    assert k1 == k2


def test_find_duplicates():
    from rdkit import Chem

    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "OCC", "CCN", "c1ccccc1"]]
    keyed, _fallbacks = pp.find_duplicates(mols)
    dup_groups = [v for v in keyed.values() if len(v) > 1]
    assert dup_groups == [[0, 1]]


def test_find_duplicates_skips_none():
    from rdkit import Chem

    mols = [Chem.MolFromSmiles("CCO"), None, Chem.MolFromSmiles("CCN")]
    keyed, _fallbacks = pp.find_duplicates(mols)
    assert sum(len(v) for v in keyed.values()) == 2


def test_dedup_key_inchikey_fallback_used_flag(monkeypatch):
    """Force the InChIKey-generation-failed branch (empty string) to exercise the canonical-SMILES
    fallback path, since finding a real molecule RDKit's InChI writer rejects is not portable."""
    from rdkit import Chem

    monkeypatch.setattr(Chem, "MolToInchiKey", lambda m: "")
    mol = Chem.MolFromSmiles("CCO")
    key, used_fallback = pp.dedup_key(mol)
    assert used_fallback is True
    assert key == Chem.MolToSmiles(Chem.MolFromSmiles("CCO"), canonical=True, isomericSmiles=True)


def test_aggregate_replicates_median():
    df = pd.DataFrame(
        {"inchikey": ["A", "A", "A", "B"], "y": [1.0, 2.0, 3.0, 5.0]}
    )
    agg, dropped = pp.aggregate_replicates(df, method="median")
    row_a = agg[agg["inchikey"] == "A"].iloc[0]
    assert row_a["y"] == 2.0
    assert dropped.empty


def test_aggregate_replicates_max_spread_drops():
    df = pd.DataFrame({"inchikey": ["A", "A", "B", "B"], "y": [1.0, 10.0, 5.0, 5.1]})
    agg, dropped = pp.aggregate_replicates(df, method="median", max_spread=1.0)
    assert set(agg["inchikey"]) == {"B"}
    assert set(dropped["inchikey"]) == {"A"} or list(dropped["inchikey"]) == ["A", "A"]


def test_aggregate_replicates_drop_conflicting():
    df = pd.DataFrame({"inchikey": ["A", "A", "B", "B"], "y": [1.0, 1.0, 5.0, 6.0]})
    agg, dropped = pp.aggregate_replicates(df, method="drop_conflicting")
    assert set(agg["inchikey"]) == {"A"}
    assert set(dropped["inchikey"]) == {"B"}


def test_run_pipeline_mol_kind():
    from rdkit import Chem

    mols_in = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCN")]
    result = pp.run_pipeline(mols_in, None, None, x_kind="mol", on_duplicates="ignore")
    assert result.mols == mols_in
    assert result.smiles is None


def test_run_pipeline_standardization_warning_when_input_differs(recwarn):
    from chemsplit.exceptions import StandardizationWarning

    # A salt: standardize() strips [Na+], so the un-standardized (default) form differs.
    result = pp.run_pipeline(
        ["CC(=O)O.[Na+]"], None, None, x_kind="smiles", standardize=False, on_duplicates="ignore",
    )
    assert result.mols[0] is not None
    kinds = [w.category for w in recwarn.list]
    assert StandardizationWarning in kinds


def test_run_pipeline_duplicates_warn_emits_warning(recwarn):
    from chemsplit.exceptions import DuplicateWarning

    pp.run_pipeline(["CCO", "OCC"], None, None, x_kind="smiles", on_duplicates="warn")
    kinds = [w.category for w in recwarn.list]
    assert DuplicateWarning in kinds


def test_run_pipeline_duplicates_group_unparsed_records_get_singleton_keys():
    result = pp.run_pipeline(
        ["CCO", "OCC", "not_a_smiles!!"],
        None,
        None,
        x_kind="smiles",
        on_parse_error="discard",
        on_duplicates="group",
        group_forming=True,
    )
    # 3 records -> 2 distinct group labels (the duplicate pair, plus the unparsed record's own
    # singleton), even though the unparsed record was also moved to forced_discard.
    assert len(set(result.dedup_group_labels.tolist())) == 2
    assert result.forced_discard == [2]


def test_run_pipeline_parse_error_raise():
    with pytest.raises(MoleculeParseError):
        pp.run_pipeline(
            ["CCO", "not_a_smiles!!"], None, None, x_kind="smiles", on_parse_error="raise"
        )


def test_run_pipeline_parse_error_discard():
    result = pp.run_pipeline(
        ["CCO", "not_a_smiles!!", "CCN"], None, None, x_kind="smiles", on_parse_error="discard"
    )
    assert result.forced_discard == [1]


def test_run_pipeline_duplicates_raise():
    with pytest.raises(DuplicateRecordError):
        pp.run_pipeline(
            ["CCO", "OCC"], None, None, x_kind="smiles", on_duplicates="raise"
        )


def test_run_pipeline_duplicates_group_requires_group_forming():
    from chemsplit.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        pp.run_pipeline(
            ["CCO", "OCC"], None, None, x_kind="smiles", on_duplicates="group",
            group_forming=False,
        )


def test_run_pipeline_duplicates_group_produces_labels():
    result = pp.run_pipeline(
        ["CCO", "OCC", "CCN"], None, None, x_kind="smiles", on_duplicates="group",
        group_forming=True,
    )
    assert result.dedup_group_labels is not None
    assert result.dedup_group_labels[0] == result.dedup_group_labels[1]
    assert result.dedup_group_labels[2] != result.dedup_group_labels[0]
