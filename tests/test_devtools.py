"""Coverage for chemsplit/_devtools.py's actual regeneration machinery (the gate, the write path,
the failure-reporting path, the tolerance-tier payload branch, and the CLI entry point) -- as
opposed to tests/test_golden.py, which only exercises the plan/fixture/payload helper functions
via the already-committed goldens.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from chemsplit import _devtools


class TestRegenerateGoldensGate:
    def test_refuses_without_confirm(self, monkeypatch):
        monkeypatch.setenv("CHEMSPLIT_ALLOW_GOLDEN_REGEN", "1")
        with pytest.raises(RuntimeError, match="confirm=True"):
            _devtools.regenerate_goldens(confirm=False)

    def test_refuses_without_env_var_even_with_confirm(self, monkeypatch):
        monkeypatch.delenv("CHEMSPLIT_ALLOW_GOLDEN_REGEN", raising=False)
        with pytest.raises(RuntimeError, match="CHEMSPLIT_ALLOW_GOLDEN_REGEN"):
            _devtools.regenerate_goldens(confirm=True)

    def test_refuses_with_wrong_env_var_value(self, monkeypatch):
        monkeypatch.setenv("CHEMSPLIT_ALLOW_GOLDEN_REGEN", "0")
        with pytest.raises(RuntimeError):
            _devtools.regenerate_goldens(confirm=True)


class TestRegenerateGoldensWrite:
    def test_writes_a_real_golden_into_a_temp_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_devtools, "_GOLDEN_DIR", tmp_path)
        monkeypatch.setenv("CHEMSPLIT_ALLOW_GOLDEN_REGEN", "1")
        written = _devtools.regenerate_goldens(confirm=True, splitter_ids=["random"])
        assert "random" in written
        out_path = Path(written["random"])
        assert out_path.exists()
        payload = json.loads(out_path.read_text())
        assert payload["tier"] == "exact"

    def test_writes_interaction_and_mood_and_lineage_plan_entries(self, tmp_path, monkeypatch):
        """Exercises _interactions_X_y/_binary_y (cold_drug/cold_target/cold_pair,
        simpd/source/party) and the MOODSplitter plan closure (mood), none of which the
        random-only tests above touch."""
        monkeypatch.setattr(_devtools, "_GOLDEN_DIR", tmp_path)
        monkeypatch.setenv("CHEMSPLIT_ALLOW_GOLDEN_REGEN", "1")
        ids = ["cold_drug", "cold_target", "cold_pair", "ave", "mood", "source", "party"]
        written = _devtools.regenerate_goldens(confirm=True, splitter_ids=ids)
        assert set(written) == set(ids)

    def test_unknown_splitter_id_reported_as_failure_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_devtools, "_GOLDEN_DIR", tmp_path)
        monkeypatch.setenv("CHEMSPLIT_ALLOW_GOLDEN_REGEN", "1")
        written = _devtools.regenerate_goldens(confirm=True, splitter_ids=["not_a_real_splitter"])
        assert written == {}

    def test_builder_exception_reported_as_failure_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_devtools, "_GOLDEN_DIR", tmp_path)
        monkeypatch.setenv("CHEMSPLIT_ALLOW_GOLDEN_REGEN", "1")

        def _boom():
            raise ValueError("synthetic failure for coverage")

        real_build_plan = _devtools._build_plan

        def _patched_build_plan(fixtures):
            plan, names = real_build_plan(fixtures)
            plan["random"] = _boom
            return plan, names

        monkeypatch.setattr(_devtools, "_build_plan", _patched_build_plan)
        written = _devtools.regenerate_goldens(confirm=True, splitter_ids=["random"])
        assert written == {}


class TestToleranceTierPayload:
    def test_tolerance_tier_branch(self):
        """A splitter with metadata["nondeterministic_method"]=True takes the tolerance-tier
        payload path (sizes + group-size histogram as a multiset, no exact index arrays)."""
        result = SimpleNamespace(
            metadata={"nondeterministic_method": True},
            groups=__import__("numpy").array([0, 0, 1, 1, 1], dtype="int64"),
            train=__import__("numpy").array([0, 1], dtype="int64"),
            valid=__import__("numpy").array([], dtype="int64"),
            test=__import__("numpy").array([2, 3, 4], dtype="int64"),
            discard=__import__("numpy").array([], dtype="int64"),
        )
        payload = _devtools._to_golden_payload(result)
        assert payload["tier"] == "tolerance"
        assert payload["n_train"] == 2
        assert payload["n_test"] == 3
        assert payload["group_size_histogram"] == [2, 3]

    def test_tolerance_tier_without_groups(self):
        import numpy as np

        result = SimpleNamespace(
            metadata={"nondeterministic_method": True},
            groups=None,
            train=np.array([0], dtype="int64"),
            valid=np.array([], dtype="int64"),
            test=np.array([1], dtype="int64"),
            discard=np.array([], dtype="int64"),
        )
        payload = _devtools._to_golden_payload(result)
        assert payload["group_size_histogram"] is None


class TestMain:
    def test_main_gate_error_propagates_uncaught(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_devtools, "_GOLDEN_DIR", tmp_path)
        monkeypatch.delenv("CHEMSPLIT_ALLOW_GOLDEN_REGEN", raising=False)
        # No --confirm -> regenerate_goldens raises RuntimeError -> propagates (main() does not
        # itself catch the gate's RuntimeError; this documents that contract).
        with pytest.raises(RuntimeError):
            _devtools.main(["regenerate_goldens", "--splitter", "random"])

    def test_main_returns_1_when_nothing_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_devtools, "_GOLDEN_DIR", tmp_path)
        monkeypatch.setenv("CHEMSPLIT_ALLOW_GOLDEN_REGEN", "1")
        rc = _devtools.main(["regenerate_goldens", "--confirm", "--splitter", "not_a_real_splitter"])
        assert rc == 1

    def test_main_returns_0_on_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_devtools, "_GOLDEN_DIR", tmp_path)
        monkeypatch.setenv("CHEMSPLIT_ALLOW_GOLDEN_REGEN", "1")
        rc = _devtools.main(["regenerate_goldens", "--confirm", "--splitter", "random"])
        assert rc == 0

    def test_main_missing_subcommand_exits_via_argparse(self):
        with pytest.raises(SystemExit):
            _devtools.main([])


def test_cli_entrypoint_via_subprocess():
    """Exercises the ``if __name__ == "__main__":`` guard for real, and confirms the gate refuses
    without CHEMSPLIT_ALLOW_GOLDEN_REGEN set even when invoked as a real CLI."""
    import os

    env = dict(os.environ)
    env.pop("CHEMSPLIT_ALLOW_GOLDEN_REGEN", None)
    result = subprocess.run(
        [sys.executable, "-m", "chemsplit._devtools", "regenerate_goldens", "--splitter", "random"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode != 0
    assert "CHEMSPLIT_ALLOW_GOLDEN_REGEN" in result.stderr or "confirm" in result.stderr.lower()
