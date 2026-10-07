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
