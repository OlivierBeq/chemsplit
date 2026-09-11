"""End-to-end smoke tests for `python -m chemsplit` (chemsplit/cli.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from chemsplit import cli
from chemsplit.datasets import make_linear_series


@pytest.fixture()
def mols_csv(tmp_path):
    fx = make_linear_series(n=60, seed=0)
    df = pd.DataFrame({"smiles": fx.smiles, "y": fx.y})
    path = tmp_path / "mols.csv"
    df.to_csv(path, index=False)
    return path


def test_cli_list_table(capsys):
    rc = cli.main(["list"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "random" in out


def test_cli_list_json(capsys):
    rc = cli.main(["list", "--json"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    rows = json.loads(out)
    assert len(rows) == 53
    ids = {r["id"] for r in rows}
    assert "applicability_domain" in ids
    assert "random" in ids


def test_cli_list_family_filter(capsys):
    rc = cli.main(["list", "--family", "scaffold", "--json"])
    assert rc == cli.EXIT_OK
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 6
    assert all(r["family"] == "scaffold" for r in rows)


def test_cli_split_and_audit(tmp_path, mols_csv, capsys):
    split_out = tmp_path / "split.json"
    rc = cli.main(
        [
            "split",
            "--splitter",
            "random",
            "--input",
            str(mols_csv),
            "--smiles-col",
            "smiles",
            "--label-col",
            "y",
            "--seed",
            "0",
            "--out",
            str(split_out),
        ]
    )
    assert rc == cli.EXIT_OK
    assert split_out.exists()
    payload = json.loads(split_out.read_text())
    assert payload["schema"] == "chemsplit/split/1"

    audit_out = tmp_path / "audit.json"
    rc = cli.main(
        [
            "audit",
            "--split",
            str(split_out),
            "--input",
            str(mols_csv),
            "--smiles-col",
            "smiles",
            "--out",
            str(audit_out),
        ]
    )
    assert rc == cli.EXIT_OK
    assert audit_out.exists()
    report = json.loads(audit_out.read_text())
    assert "flags" in report or "n_train" in report


def test_python_dash_m_chemsplit_list():
    """Exercises chemsplit/__main__.py's ``if __name__ == "__main__":`` guard directly (calling
    ``cli.main()`` in-process, as the other tests here do, never actually runs that module as
    ``__main__``)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "chemsplit", "list"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == cli.EXIT_OK
    assert "random" in result.stdout


def test_cli_split_unknown_splitter_exits_validation(tmp_path, mols_csv):
    rc = cli.main(
        [
            "split",
            "--splitter",
            "not_a_real_splitter",
            "--input",
            str(mols_csv),
            "--smiles-col",
            "smiles",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == cli.EXIT_VALIDATION
