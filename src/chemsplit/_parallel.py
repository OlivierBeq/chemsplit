"""Deterministic parallel helpers.

Determinism: ``n_jobs`` MUST NOT change any returned value. Every helper maps over contiguous,
index-ordered chunks and reassembles them in input order, so every downstream reduction sees the
serial order. Worker count is a throughput knob, never a semantic one.

``joblib`` is imported lazily: ``import chemsplit`` must stay cheap and rdkit/sklearn-free.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import TypeVar

__all__ = ["MIN_ITEMS_FOR_WORKERS", "effective_n_jobs", "ordered_map"]

_T = TypeVar("_T")
_R = TypeVar("_R")

MIN_ITEMS_FOR_WORKERS = 2000
"""Below this many items, :func:`ordered_map` runs serially: pool startup and input shipping cost
more than the work.

A frozen constant, never auto-tuned from the host. Both paths return identical values, so the
boundary only affects speed -- but fixing it keeps every machine on the same code path.
"""

_CHUNKS_PER_WORKER = 4
"""More than one chunk per worker evens out uneven chunk costs; too many wastes shipping time."""


def effective_n_jobs(n_jobs: int | None) -> int:
    """Resolve an ``n_jobs`` spec to a positive worker count, following scikit-learn.

    :param n_jobs: ``None`` or ``1`` for serial, ``-1`` for every CPU, and ``-k`` for
        ``n_cpus + 1 - k``.
    :return: the worker count, at least ``1``.
    """
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs > 0:
        return n_jobs
    n_cpus = os.cpu_count() or 1
    return max(1, n_cpus + 1 + n_jobs)


def will_parallelize(
    n_items: int, n_jobs: int | None, min_items: int = MIN_ITEMS_FOR_WORKERS
) -> bool:
    """Report whether :func:`ordered_map` would actually use workers for this problem.

    Lets a caller pick the cheaper of two *equivalent* formulations -- reusing parsed objects
    in-process, or re-deriving them from strings that ship cheaply to a worker.

    :param n_items: the number of items to map over.
    :param n_jobs: worker count, as for :func:`effective_n_jobs`.
    :param min_items: the serial-path threshold.
    :return: ``True`` if workers would be used.
    """
    return effective_n_jobs(n_jobs) > 1 and n_items >= min_items


def chunk_bounds(n_items: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split ``range(n_items)`` into at most ``n_chunks`` contiguous ``(start, stop)`` spans.

    :param n_items: how many items to cover.
    :param n_chunks: the desired number of chunks, at least ``1``.
    :return: the spans, in ascending order, covering ``range(n_items)`` exactly once.
    """
    if n_items <= 0:
        return []
    n_chunks = max(1, min(n_chunks, n_items))
    size = -(-n_items // n_chunks)
    return [(s, min(s + size, n_items)) for s in range(0, n_items, size)]


def ordered_map(
    fn: Callable[[Sequence[_T]], list[_R]],
    items: Sequence[_T],
    *,
    n_jobs: int | None = 1,
    min_items: int = MIN_ITEMS_FOR_WORKERS,
) -> list[_R]:
    """Apply a chunk-wise function across ``items``, returning results in input order.

``fn`` receives a contiguous slice and returns one result per element, in order. Chunks are
    reassembled by index, so the result is identical for every ``n_jobs``.

    :param fn: the chunk worker, picklable for the process backend.
    :param items: the inputs.
    :param n_jobs: worker count, as for :func:`effective_n_jobs`. Results do not depend on it.
    :param min_items: below this many items, run serially regardless of ``n_jobs``.
    :return: one result per input, in input order.
    """
    n_items = len(items)
    workers = effective_n_jobs(n_jobs)
    if workers == 1 or n_items < min_items:
        return fn(items)

    bounds = chunk_bounds(n_items, workers * _CHUNKS_PER_WORKER)
    if len(bounds) == 1:
        return fn(items)

    from joblib import Parallel, delayed

    # joblib preserves submission order, and the chunks are contiguous and ascending, so
    # concatenating them rebuilds the input order
    parts: list[list[_R]] = Parallel(n_jobs=workers, prefer="processes")(
        delayed(fn)(items[start:stop]) for start, stop in bounds
    )
    out: list[_R] = []
    for part in parts:
        out.extend(part)
    if len(out) != n_items:
        raise AssertionError(
            f"ordered_map: worker returned {len(out)} results for {n_items} items"
        )
    return out


