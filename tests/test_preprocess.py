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
    assert set(dropped["inchikey"]) == {"A", "A"} or list(dropped["inchikey"]) == ["A", "A"]


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
