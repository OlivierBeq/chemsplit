"""Developer tooling: golden-file regeneration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "golden"

#: A zero-arg callable returning (X, y, split_kwargs, ctor_kwargs) for one splitter's golden case.
PlanBuilder = Callable[[], tuple[Any, Any, dict[str, Any], dict[str, Any]]]


def _fixture_cache() -> dict[str, Any]:
    from chemsplit import datasets as ds

    return {
        "linear_series": ds.make_linear_series(n=80, seed=0),
        "scaffold_families": ds.make_scaffold_families(n_scaffolds=8, per_scaffold=10, seed=0),
        "two_clusters": ds.make_two_clusters(n=80, seed=0),
        "activity_cliffs": ds.make_activity_cliffs(n_pairs=20, seed=0),
        "dated_series": ds.make_dated_series(n=100, seed=0),
        "multitask_sparse": ds.make_multitask_sparse(n=100, n_tasks=4, seed=0),
        "interactions": ds.make_interactions(n_compounds=40, n_targets=10, seed=0),
        "sequences": ds.make_sequences(n=20, families=4, seed=0),
        "pathological": ds.make_pathological(),
        "all_identical": ds.make_all_identical(n=30),
        "singletons": ds.make_singletons(n=20, seed=0),
        "label_extremes": ds.make_label_extremes(n=100, seed=0),
        # SIMPDSplitter (simpd) needs n >= 200 to meaningfully run its GA.
        "simpd_series": ds.make_scaffold_families(n_scaffolds=10, per_scaffold=20, seed=1),
        # A parallel sequences array for ComplexJointSplitter (complex_joint), which wants
        # ctx.sequences to derive its default sequence grouper even though its own `accepts`
        # doesn't include "sequences" (it's a second axis alongside the ligand SMILES, not the
        # primary X).
        "aux_sequences_80": ds.make_sequences(n=80, families=4, seed=1),
    }


def _interactions_X_y(fixture: Any) -> tuple[list[tuple[str, str]], list[float]]:
    X = [(fixture.smiles[ci], fixture.targets[ti]) for ci, ti, _y in fixture.interactions]
    y = [float(_y) for _ci, _ti, _y in fixture.interactions]
    return X, y


def _binary_y(y: np.ndarray) -> np.ndarray:
    med = float(np.median(y))
    return (np.asarray(y) > med).astype(np.int64)


# Each entry: splitter_id -> a zero-arg callable returning (X, y, split_kwargs, ctor_kwargs).
# split_kwargs are passed to split_result(X, y, **split_kwargs) (e.g. dates=, X_kind=,
# sequences=). ctor_kwargs are passed to the splitter's constructor alongside random_state=0.
def _build_plan(fixtures: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    F = fixtures
    plan: dict[str, Any] = {}
    fixture_name_of: dict[str, str] = {}

    def smiles_case(fid: str, *, y: np.ndarray | None = None, **ctor: Any) -> PlanBuilder:
        return lambda: (F[fid].smiles, y, {}, ctor)

    # -- baseline --
    plan["random"] = smiles_case("linear_series")
    plan["stratified_random"] = lambda: (F["linear_series"].smiles, F["linear_series"].y, {}, {})
    plan["k_fold"] = lambda: (F["linear_series"].smiles, None, {}, {"n_splits": 3})
    plan["monte_carlo"] = smiles_case("linear_series")
    plan["predefined"] = lambda: (
        F["linear_series"].smiles,
        None,
        {},
        {
            "assignment": {
                "train": list(range(0, 60)),
                "test": list(range(60, 80)),
            }
        },
    )

    # -- scaffold --
    plan["murcko_scaffold"] = smiles_case("scaffold_families")
    plan["generic_scaffold"] = smiles_case("scaffold_families")
    plan["scaffold_tree"] = smiles_case("scaffold_families")
    plan["ring_system"] = smiles_case("scaffold_families")
    plan["matched_molecular_series"] = smiles_case("scaffold_families")
    plan["activity_cliff"] = lambda: (
        F["activity_cliffs"].smiles,
        F["activity_cliffs"].y,
        {},
        {},
    )

    # -- similarity --
    for sid in ["similarity_threshold", "butina", "k_means_cluster", "density_cluster", "spectral",
                "max_min", "max_dissimilarity", "perimeter", "leave_one_cluster_out"]:
        plan[sid] = smiles_case("two_clusters")
    plan["balanced_multi_task"] = lambda: (
        F["multitask_sparse"].smiles,
        F["multitask_sparse"].y,
        {},
        {},
    )

    # -- embedding --
    plan["umap_cluster"] = smiles_case("two_clusters")
    plan["projection"] = smiles_case("two_clusters")
    plan["latent_space"] = lambda: (
        np.random.default_rng(0).standard_normal((60, 12)),
        None,
        {},
        {},
    )

    # -- property --
    plan["property"] = smiles_case("linear_series")
    plan["label_extrapolation"] = lambda: (F["linear_series"].smiles, F["linear_series"].y, {}, {})
    plan["stratified_distribution"] = lambda: (F["linear_series"].smiles, F["linear_series"].y, {}, {})
    def _moodsplitter_case() -> tuple[list[str], None, dict[str, Any], dict[str, Any]]:
        from chemsplit.registry import get_splitter

        candidates = [
            get_splitter("random", random_state=0),
            get_splitter("butina", random_state=0),
        ]
        return (
            F["two_clusters"].smiles,
            None,
            {},
            {"candidates": candidates, "deployment_set": F["singletons"].smiles},
        )

    plan["mood"] = _moodsplitter_case
    plan["adversarial"] = smiles_case("two_clusters")

    # -- lineage --
    plan["temporal"] = lambda: (
        F["dated_series"].smiles,
        None,
        {"dates": F["dated_series"].dates},
        {},
    )
    plan["simpd"] = lambda: (
        F["simpd_series"].smiles,
        _binary_y(np.arange(len(F["simpd_series"].smiles)) % 2),
        {},
        {},
    )
    plan["source"] = lambda: (
        F["scaffold_families"].smiles,
        None,
        {},
        {"source": F["scaffold_families"].groups_true.tolist()},
    )
    plan["party"] = lambda: (
        F["scaffold_families"].smiles,
        None,
        {},
        {"party": F["scaffold_families"].groups_true.tolist(), "synthesis": "given"},
    )

    # -- task --
    plan["hi"] = smiles_case("two_clusters")
    plan["lo"] = lambda: (F["linear_series"].smiles, F["linear_series"].y, {}, {})
    plan["scaffold_hop"] = lambda: (
        F["scaffold_families"].smiles,
        _binary_y(np.arange(len(F["scaffold_families"].smiles)) % 3 == 0),
        {},
        {"pharmacophore_similarity": "none"},
    )
    for sid in ["cold_drug", "cold_target", "cold_pair"]:
        def _make(sid: str = sid) -> tuple[list[tuple[str, str]], list[float], dict[str, Any], dict[str, Any]]:
            X, y = _interactions_X_y(F["interactions"])
            return (X, y, {}, {})
        plan[sid] = _make
    plan["ave"] = lambda: (
        F["two_clusters"].smiles,
        _binary_y(np.arange(len(F["two_clusters"].smiles)) % 2),
        {},
        {},
    )
    plan["decoy_benchmark"] = lambda: (
        F["scaffold_families"].smiles,
        _binary_y(np.arange(len(F["scaffold_families"].smiles)) % 5 == 0),
        {},
        {"scheme": "spatial_random"},
    )

    # -- biomolecular --
    plan["sequence_identity"] = lambda: (
        F["sequences"].sequences,
        None,
        {"X_kind": "sequences", "sequences": F["sequences"].sequences},
        {},
    )
    plan["protein_family"] = lambda: (
        F["sequences"].sequences,
        None,
        {"X_kind": "sequences", "sequences": F["sequences"].sequences},
        {"family_labels": F["sequences"].groups_true.tolist()},
    )
    plan["binding_site"] = lambda: (
        F["sequences"].sequences,
        None,
        {"X_kind": "sequences", "sequences": F["sequences"].sequences},
        {"representation": "pocket_sequence"},
    )
    plan["deposition_date"] = lambda: (
        F["dated_series"].smiles,
        None,
        {"dates": F["dated_series"].dates},
        {"cut_date": str(np.median(F["dated_series"].dates.astype("datetime64[D]").astype("int64")).astype("datetime64[D]"))},
    )
    plan["complex_joint"] = lambda: (
        F["two_clusters"].smiles,
        None,
        {"sequences": F["aux_sequences_80"].sequences},
        {},
    )

    # -- protocol --
    plan["group_k_fold"] = smiles_case("scaffold_families", **{"grouper": "murcko_scaffold"})
    plan["three_way"] = lambda: (
        F["two_clusters"].smiles,
        None,
        {},
        {"base_splitter": "random", "train_size": 0.6, "valid_size": 0.2, "test_size": 0.2},
    )
    plan["repeated"] = lambda: (
        F["linear_series"].smiles,
        None,
        {},
        {"base_splitter": "random", "n_repeats": 3},
    )
    plan["nested_cv"] = lambda: (
        F["linear_series"].smiles,
        None,
        {},
        {"outer_splitter": "random", "inner_splitter": "random"},
    )
    plan["external_holdout"] = lambda: (
        F["two_clusters"].smiles[:60],
        None,
        {},
        {"X_external": F["two_clusters"].smiles[60:70]},
    )
    plan["applicability_domain"] = lambda: (F["two_clusters"].smiles, None, {}, {"base_splitter": "random"})

    fixture_name_of.update(
        {
            "random": "linear_series",
            "stratified_random": "linear_series",
            "k_fold": "linear_series",
            "monte_carlo": "linear_series",
            "predefined": "linear_series",
            "murcko_scaffold": "scaffold_families",
            "generic_scaffold": "scaffold_families",
            "scaffold_tree": "scaffold_families",
            "ring_system": "scaffold_families",
            "matched_molecular_series": "scaffold_families",
            "activity_cliff": "activity_cliffs",
            "similarity_threshold": "two_clusters",
            "butina": "two_clusters",
            "k_means_cluster": "two_clusters",
            "density_cluster": "two_clusters",
            "spectral": "two_clusters",
            "max_min": "two_clusters",
            "max_dissimilarity": "two_clusters",
            "perimeter": "two_clusters",
            "leave_one_cluster_out": "two_clusters",
            "balanced_multi_task": "multitask_sparse",
            "umap_cluster": "two_clusters",
            "projection": "two_clusters",
            "latent_space": "synthetic_matrix",
            "property": "linear_series",
            "label_extrapolation": "linear_series",
            "stratified_distribution": "linear_series",
            "mood": "two_clusters",
            "adversarial": "two_clusters",
            "temporal": "dated_series",
            "simpd": "scaffold_families",
            "source": "scaffold_families",
            "party": "scaffold_families",
            "hi": "two_clusters",
            "lo": "linear_series",
            "scaffold_hop": "scaffold_families",
            "cold_drug": "interactions",
            "cold_target": "interactions",
            "cold_pair": "interactions",
            "ave": "two_clusters",
            "decoy_benchmark": "scaffold_families",
            "sequence_identity": "sequences",
            "protein_family": "sequences",
            "binding_site": "sequences",
            "deposition_date": "dated_series",
            "complex_joint": "two_clusters",
            "group_k_fold": "scaffold_families",
            "three_way": "two_clusters",
            "repeated": "linear_series",
            "nested_cv": "linear_series",
            "external_holdout": "two_clusters",
            "applicability_domain": "two_clusters",
        }
    )

    return plan, fixture_name_of


def _stabilize_for_golden(obj: Any, *, key: str | None = None) -> Any:
    """Round floats (BLAS/build ULP noise) and mask ``umap_versions`` before golden comparison."""
    if isinstance(obj, float):
        return float(f"{obj:.9g}")
    if isinstance(obj, dict):
        if key == "umap_versions":
            return dict.fromkeys(obj, "<version>")
        return {k: _stabilize_for_golden(v, key=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stabilize_for_golden(v, key=key) for v in obj]
    return obj


def _to_golden_payload(result: Any, ctx_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """``result`` is a SplitResult. Chooses the exact or tolerance tier based on
    ``metadata.get("nondeterministic_method")`` (per-instance; a couple of splitters, e.g. the
    embedding family's t-SNE/MDS projection modes, can only know this after actually running)."""
    nondeterministic = bool(result.metadata.get("nondeterministic_method", False))
    if not nondeterministic:
        stabilized = _stabilize_for_golden(json.loads(result.to_json()))
        split_result_json = json.dumps(
            stabilized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return {"tier": "exact", "split_result_json": split_result_json}

    # Tolerance tier: sizes exact, group-size histogram as a multiset.
    if result.groups is not None:
        group_sizes = np.bincount(result.groups).tolist()
    else:
        group_sizes = None
    return {
        "tier": "tolerance",
        "n_train": int(len(result.train)),
        "n_valid": int(len(result.valid)),
        "n_test": int(len(result.test)),
        "n_discard": int(len(result.discard)),
        "group_size_histogram": group_sizes,
    }


def regenerate_goldens(
    *, confirm: bool = False, splitter_ids: list[str] | None = None
) -> dict[str, str]:
    if not confirm or os.environ.get("CHEMSPLIT_ALLOW_GOLDEN_REGEN") != "1":
        raise RuntimeError(
            "regenerate_goldens() refuses to run: pass confirm=True AND set "
            "CHEMSPLIT_ALLOW_GOLDEN_REGEN=1 in the environment, so goldens are "
            "never silently rewritten by an unrelated test run."
        )

    from chemsplit.registry import SPLITTER_REGISTRY, _ensure_built, get_splitter

    _ensure_built()
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = _fixture_cache()
    plan, fixture_name_of = _build_plan(fixtures)

    written: dict[str, str] = {}
    failures: dict[str, str] = {}
    ids = splitter_ids if splitter_ids is not None else [
        sid for sid in SPLITTER_REGISTRY if sid in plan
    ]
    for sid in ids:
        if sid not in plan:
            failures[sid] = "no plan entry"
            continue
        builder = plan[sid]
        try:
            X, y, split_kwargs, ctor_kwargs = builder()
            splitter = get_splitter(sid, random_state=0, **ctor_kwargs)
            results = splitter.split_result(X, y, **split_kwargs)
            result = results[0]
            payload = _to_golden_payload(result)
            fixture_name = fixture_name_of.get(sid, "custom")
            out_path = _GOLDEN_DIR / f"{sid}__{fixture_name}__seed0.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            written[sid] = str(out_path)
            print(f"OK {sid:16s} -> {out_path.name}")
        except Exception as exc:  # noqa: BLE001 - devtool, report and continue
            failures[sid] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {sid:16s} -> {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"\n{len(written)}/{len(ids)} golden files written; {len(failures)} failures.")
    if failures:
        print("Failures:", file=sys.stderr)
        for sid, msg in failures.items():
            print(f"  {sid}: {msg}", file=sys.stderr)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chemsplit._devtools")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("regenerate_goldens")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--splitter", action="append", dest="splitter_ids", default=None)
    args = parser.parse_args(argv)

    if args.command == "regenerate_goldens":
        written = regenerate_goldens(confirm=args.confirm, splitter_ids=args.splitter_ids)
        return 0 if written else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
