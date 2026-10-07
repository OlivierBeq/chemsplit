"""The parallel layer is a throughput knob: ``n_jobs`` may never change a result.

``README.md`` and ``BaseSplitter`` have always promised this; nothing checked it until now.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit import _parallel
from chemsplit.preprocess import (
    StandardizeConfig,
    _digest_mols,
    _digest_smiles_chunk,
    _parse_all,
    run_pipeline,
)
from chemsplit.registry import get_splitter

# clears MIN_ITEMS_FOR_WORKERS, so the parallel path is actually exercised
_N = _parallel.MIN_ITEMS_FOR_WORKERS + 250


def _library(n: int) -> list[str]:
    """A deterministic, duplicate-free SMILES set big enough to trigger the worker path."""
    out: list[str] = []
    subs = ["", "C", "CC", "CCC", "F", "Cl", "Br", "O", "N", "OC", "NC", "C(=O)O"]
    cores = ["c1cc({a})ccc1{b}", "c1cc({a})ncc1{b}", "C1CC({a})CCC1{b}", "c1cc({a})sc1{b}"]
    i = 0
    while len(out) < n:
        core = cores[i % len(cores)]
        a = subs[(i // len(cores)) % len(subs)] or "C"
        out.append(core.format(a=a, b="C" * (1 + i % 9) + "O" * (i % 3)))
        i += 1
    return out[:n]


def test_effective_n_jobs_follows_sklearn() -> None:
    n_cpus = _parallel.effective_n_jobs(-1)
    assert n_cpus >= 1
    assert _parallel.effective_n_jobs(None) == 1
    assert _parallel.effective_n_jobs(1) == 1
    assert _parallel.effective_n_jobs(3) == 3
    assert _parallel.effective_n_jobs(-2) == max(1, n_cpus - 1)


def test_chunk_bounds_cover_every_index_exactly_once() -> None:
    for n_items in (0, 1, 7, 100):
        for n_chunks in (1, 2, 3, 50):
            bounds = _parallel.chunk_bounds(n_items, n_chunks)
            covered = [i for start, stop in bounds for i in range(start, stop)]
            assert covered == list(range(n_items))
            # contiguous and ascending, which is what preserves input order
            assert all(b[0] < b[1] for b in bounds)
            assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))


def _double(chunk: list[int]) -> list[int]:
    return [x * 2 for x in chunk]


@pytest.mark.parametrize("n_jobs", [1, 2, 4, -1])
def test_ordered_map_preserves_input_order(n_jobs: int) -> None:
    items = list(range(5000))
    assert _parallel.ordered_map(_double, items, n_jobs=n_jobs, min_items=10) == _double(items)


def test_digest_formulations_agree() -> None:
    """The SMILES and molecule digests must agree exactly.

    ``run_pipeline`` picks between them on whether workers are used, so divergence would make
    results depend on ``n_jobs``.
    """
    smiles = _library(400)
    mols, _ = _parse_all(smiles)
    cfg = StandardizeConfig()
    for want_differs in (False, True):
        assert _digest_smiles_chunk(
            smiles, config=cfg, want_differs=want_differs
        ) == _digest_mols(mols, cfg, want_differs=want_differs)


@pytest.mark.parametrize("n_jobs", [2, 4, -1])
def test_run_pipeline_is_n_jobs_invariant(n_jobs: int) -> None:
    smiles = _library(_N)
    ref = run_pipeline(smiles, None, None, x_kind="smiles", n_jobs=1)
    got = run_pipeline(smiles, None, None, x_kind="smiles", n_jobs=n_jobs)
    assert got.forced_discard == ref.forced_discard
    assert (got.dedup_group_labels is None) == (ref.dedup_group_labels is None)
    if ref.dedup_group_labels is not None:
        assert np.array_equal(got.dedup_group_labels, ref.dedup_group_labels)


@pytest.mark.parametrize("splitter_id", ["random", "murcko_scaffold", "butina", "max_min"])
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_split_result_is_n_jobs_invariant(splitter_id: str, n_jobs: int) -> None:
    smiles = _library(_N)

    def run(nj: int) -> list:
        splitter = get_splitter(
            splitter_id, random_state=0, train_size=0.8, valid_size=0.0, test_size=0.2, n_jobs=nj
        )
        return splitter.split_result(smiles)

    for ref, got in zip(run(1), run(n_jobs), strict=True):
        for part in ("train", "valid", "test", "discard"):
            assert np.array_equal(getattr(got, part), getattr(ref, part)), part
        assert (got.groups is None) == (ref.groups is None)
        if ref.groups is not None:
            assert np.array_equal(got.groups, ref.groups)
        # metadata carries computed diagnostics, so it is part of the contract
        assert got.metadata == ref.metadata


def test_cache_hit_is_indistinguishable_from_a_cold_call(recwarn: pytest.WarningsRecorder) -> None:
    """A cached call must reproduce the result *and* the warnings of an uncached one.

    The cache holds digests, not the finished ``PipelineResult``, so warnings stay downstream of it.
    Otherwise a second split of the same data would silently stop warning about duplicates.
    """
    import warnings as _warnings

    from chemsplit import preprocess as pp

    smiles = ["CC(=O)O.[Na+]", "CCO", "OCC", "CCN", "c1ccccc1"]

    def run() -> tuple[object, list[tuple[type, str]]]:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            res = pp.run_pipeline(smiles, None, None, x_kind="smiles", on_duplicates="warn")
        return res.dedup_group_labels, [(type(w.message), str(w.message)) for w in caught]

    pp.clear_digest_cache()
    cold_labels, cold_warnings = run()
    warm_labels, warm_warnings = run()  # served from the cache

    assert warm_warnings == cold_warnings
    assert any(issubclass(t, Warning) and "duplicate" in m for t, m in cold_warnings)
    assert (warm_labels is None) == (cold_labels is None)
    if cold_labels is not None:
        assert np.array_equal(warm_labels, cold_labels)


def test_cache_misses_when_standardisation_settings_change() -> None:
    from chemsplit import preprocess as pp

    smiles = ["CC(=O)O.[Na+]", "CCO", "CCN"]
    pp.clear_digest_cache()
    keep = pp._digest_cache_key(smiles, pp.StandardizeConfig(), True)
    strip = pp._digest_cache_key(smiles, pp.StandardizeConfig(stereo="strip"), True)
    no_differs = pp._digest_cache_key(smiles, pp.StandardizeConfig(), False)
    other_input = pp._digest_cache_key([*smiles, "CCC"], pp.StandardizeConfig(), True)
    assert len({keep, strip, no_differs, other_input}) == 4


def test_cache_can_be_disabled_without_changing_results() -> None:
    from chemsplit import preprocess as pp

    smiles = _library(600)
    try:
        pp.set_digest_cache_enabled(False)
        uncached = pp.run_pipeline(smiles, None, None, x_kind="smiles")
        pp.set_digest_cache_enabled(True)
        pp.clear_digest_cache()
        cached = pp.run_pipeline(smiles, None, None, x_kind="smiles")
    finally:
        pp.set_digest_cache_enabled(True)
        pp.clear_digest_cache()
    assert uncached.forced_discard == cached.forced_discard
    assert (uncached.dedup_group_labels is None) == (cached.dedup_group_labels is None)


@pytest.mark.parametrize("splitter_id", ["murcko_scaffold", "generic_scaffold"])
def test_scaffold_keys_resolve_in_worker_processes(splitter_id: str) -> None:
    """Key functions must be importable by a worker that never imported the splitter module.

    A spawned worker has an empty registry, so resolving by name alone raised KeyError and broke
    every scaffold splitter under ``n_jobs>1``.
    """
    smiles = _library(_N)

    def run(n_jobs: int) -> list:
        splitter = get_splitter(
            splitter_id,
            random_state=0,
            train_size=0.8,
            valid_size=0.0,
            test_size=0.2,
            n_jobs=n_jobs,
        )
        return splitter.split_result(smiles)

    for ref, got in zip(run(1), run(-1), strict=True):
        assert np.array_equal(got.train, ref.train)
        assert np.array_equal(got.test, ref.test)
        assert np.array_equal(got.groups, ref.groups)


def test_molmap_chunk_works_without_prior_registration() -> None:
    """The worker entry point must not depend on anything the parent registered."""
    from chemsplit import _molmap

    target = _molmap._TARGETS["scaffold.murcko"]
    saved = dict(_molmap._FUNCS)
    try:
        _molmap._FUNCS.clear()  # emulate a fresh worker interpreter
        keys = _molmap._chunk(["c1ccccc1CC", "CCO", None], target=target, params=(False,))
    finally:
        _molmap._FUNCS.update(saved)
    assert keys[0] == "c1ccccc1"
    assert keys[1] == ""  # no ring system, so no Murcko scaffold
    assert keys[2] == ""  # absent molecule
