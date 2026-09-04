"""Every committed golden file in tests/golden/ is reproducible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemsplit._devtools import _build_plan, _fixture_cache, _to_golden_payload
from chemsplit.registry import get_splitter

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_GOLDEN_FILES = sorted(_GOLDEN_DIR.glob("*__*__seed0.json")) if _GOLDEN_DIR.is_dir() else []


def _parse_golden_filename(path: Path) -> str:
    # "<splitter_id>__<fixture_name>__seed0.json" -- splitter_id is everything before the first
    # "__" (ids themselves never contain "__").
    return path.name.split("__", 1)[0]


@pytest.fixture(scope="module")
def fixtures():
    return _fixture_cache()


@pytest.fixture(scope="module")
def plan(fixtures):
    p, _names = _build_plan(fixtures)
    return p


@pytest.mark.golden
@pytest.mark.parametrize("golden_path", _GOLDEN_FILES, ids=[p.name for p in _GOLDEN_FILES])
def test_golden_reproducible(golden_path, plan):
    splitter_id = _parse_golden_filename(golden_path)
    assert splitter_id in plan, f"{golden_path.name}: no plan entry for {splitter_id!r}"

    with open(golden_path, encoding="utf-8") as fh:
        expected = json.load(fh)

    X, y, split_kwargs, ctor_kwargs = plan[splitter_id]()
    splitter = get_splitter(splitter_id, random_state=0, **ctor_kwargs)
    result = splitter.split_result(X, y, **split_kwargs)[0]
    got = _to_golden_payload(result)

    assert got["tier"] == expected["tier"], (
        f"{splitter_id}: tier changed from {expected['tier']!r} to {got['tier']!r} -- "
        "this is itself a behavioural change (deterministic_method / nondeterministic_method "
        "flipped), not just a value drift"
    )

    if expected["tier"] == "exact":
        assert got["split_result_json"] == expected["split_result_json"], (
            f"{splitter_id}: SplitResult.to_json() output no longer matches the committed golden. "
            "If this is a deliberate, disclosed behavioural change, regenerate goldens via "
            "`CHEMSPLIT_ALLOW_GOLDEN_REGEN=1 python -m chemsplit._devtools regenerate_goldens "
            f"--confirm --splitter {splitter_id}`."
        )
    else:
        assert got["n_train"] == expected["n_train"]
        assert got["n_valid"] == expected["n_valid"]
        assert got["n_test"] == expected["n_test"]
        assert got["n_discard"] == expected["n_discard"]
        # group-size histogram compared as a multiset, not position-for-position.
        assert sorted(got["group_size_histogram"] or []) == sorted(
            expected["group_size_histogram"] or []
        )


def test_every_splitter_has_a_golden():
    from chemsplit.registry import SPLITTER_REGISTRY

    covered = {_parse_golden_filename(p) for p in _GOLDEN_FILES}
    missing = sorted(set(SPLITTER_REGISTRY.keys()) - covered)
    assert not missing, f"splitters with no golden file: {missing}"


def test_golden_dir_has_exactly_52_files():
    assert len(_GOLDEN_FILES) == 52
