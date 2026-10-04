"""Every committed golden file in tests/golden/ is reproducible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemsplit._devtools import _build_plan, _fixture_cache, _to_golden_payload
from chemsplit.registry import get_splitter

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_GOLDEN_FILES = sorted(_GOLDEN_DIR.glob("*__*__seed0.json")) if _GOLDEN_DIR.is_dir() else []

# Splitters committed at the "tolerance" tier rather than byte-exact. Eigendecomposition,
# k-means, UMAP's numba JIT and torch BLAS products all vary by build, and `complex_joint`'s
# default sequence grouper depends on whether parasail is installed. `_assert_within_tolerance`
# still compares their goldens, just not byte for byte.
_TOLERANCE_TIER_SPLITTERS = {"spectral", "k_means_cluster", "umap_cluster", "complex_joint",
                             "support_points", "self_organizing_map"}

#: Partition sizes may drift by this fraction of n, matching `GroupSplitter`'s `size_tolerance`
#: default: whole groups move between partitions when a cluster boundary shifts.
_SIZE_TOLERANCE = 0.05
#: Cluster count may drift by this fraction (minimum 1 group).
_N_GROUPS_TOLERANCE = 0.20
#: The largest group's share of the dataset may drift by this much, absolutely.
_LARGEST_GROUP_FRAC_TOLERANCE = 0.10


def _assert_within_tolerance(splitter_id: str, got: dict, expected: dict) -> None:
    """Compare a tolerance-tier payload: exact record count, partition sizes and grouping summary
    within tolerance. Catches a splitter that changes what it does (wrong partition sizes, a
    collapsed or exploded cluster count) while tolerating the few-records-move jitter that a
    different BLAS or numba build produces."""
    assert got["n_records"] == expected["n_records"], f"{splitter_id}: n_records changed"
    size_slack = max(1, round(_SIZE_TOLERANCE * expected["n_records"]))
    for key in ("n_train", "n_valid", "n_test", "n_discard"):
        drift = abs(got[key] - expected[key])
        assert drift <= size_slack, (
            f"{splitter_id}: {key} = {got[key]}, expected {expected[key]} "
            f"+/- {size_slack} (drift {drift})"
        )
    if expected["n_groups"] is None:
        assert got["n_groups"] is None, f"{splitter_id}: started forming groups"
        return
    assert got["n_groups"] is not None, f"{splitter_id}: stopped forming groups"
    group_slack = max(1, round(_N_GROUPS_TOLERANCE * expected["n_groups"]))
    drift = abs(got["n_groups"] - expected["n_groups"])
    assert drift <= group_slack, (
        f"{splitter_id}: n_groups = {got['n_groups']}, expected {expected['n_groups']} "
        f"+/- {group_slack} (drift {drift})"
    )
    frac_drift = abs(got["largest_group_frac"] - expected["largest_group_frac"])
    assert frac_drift <= _LARGEST_GROUP_FRAC_TOLERANCE, (
        f"{splitter_id}: largest group covers {got['largest_group_frac']:.3f} of records, "
        f"expected {expected['largest_group_frac']:.3f} "
        f"+/- {_LARGEST_GROUP_FRAC_TOLERANCE}"
    )


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
        _assert_within_tolerance(splitter_id, got, expected)


def test_every_splitter_has_a_golden():
    from chemsplit.registry import SPLITTER_REGISTRY

    covered = {_parse_golden_filename(p) for p in _GOLDEN_FILES}
    missing = sorted(set(SPLITTER_REGISTRY.keys()) - covered)
    assert not missing, f"splitters with no golden file: {missing}"


def test_golden_dir_has_one_file_per_splitter():
    assert len(_GOLDEN_FILES) == 64


def test_tolerance_tier_splitters_are_committed_at_that_tier(plan):
    """The known-unstable splitters must stay at the tolerance tier.

    If one is ever committed as "exact" again its golden becomes a cross-build flake, and if one
    silently becomes deterministic we want to notice and promote it rather than keep the weaker
    check.
    """
    import json as _json

    for path in _GOLDEN_FILES:
        splitter_id = _parse_golden_filename(path)
        if splitter_id in _TOLERANCE_TIER_SPLITTERS:
            with open(path, encoding="utf-8") as fh:
                assert _json.load(fh)["tier"] == "tolerance", (
                    f"{splitter_id} is listed as tolerance-tier but its golden is byte-exact"
                )
