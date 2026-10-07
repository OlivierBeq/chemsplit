"""Cached, parallelisable per-molecule key computation.

Scaffold-style keys are pure functions of the molecule and the splitter's parameters, yet are
recomputed per fold by the protocol wrappers and twice over by ``compute_groups`` plus
``_partition``.

Determinism: keyed on input *content* plus parameters, mapped over contiguous ordered chunks, so
neither caching nor ``n_jobs`` changes a key. Workers exchange SMILES strings, never molecules.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import threading
from collections.abc import Callable, Sequence
from typing import Any

from chemsplit import _parallel

__all__ = ["KEY_CACHE_MAX_RECORDS", "clear_key_cache", "mapped_keys", "register"]

_FUNCS: dict[str, Callable[..., str]] = {}

_CACHE: collections.OrderedDict[str, list[str]] = collections.OrderedDict()
_LOCK = threading.Lock()

KEY_CACHE_MAX_RECORDS = 2_000_000
"""Keys the cache may hold before evicting oldest. Bounded by records, not entries, so one large
dataset cannot pin the budget."""


def register(name: str, fn: Callable[..., str]) -> str:
    """Register a per-molecule key function under a name workers can resolve.

    :param name: a stable identifier; two functions must never share one.
    :param fn: called as ``fn(mol, *params)``, returning a string. Must be pure, or caching is
        unsound.
    :raises ValueError: if ``name`` is already registered to a different function.
    :return: ``name``, so this can be used at module scope.
    """
    existing = _FUNCS.get(name)
    if existing is not None and existing is not fn:
        raise ValueError(f"_molmap name {name!r} is already registered to a different function")
    _FUNCS[name] = fn
    return name


def clear_key_cache() -> None:
    """Drop every cached key list."""
    with _LOCK:
        _CACHE.clear()


def _cache_key(name: str, params: tuple[Any,...], smiles: Sequence[str | None]) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((name, params)).encode())
    h.update(b"\x00")
    for smi in smiles:
        h.update(b"\xff" if smi is None else smi.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _chunk(
    smiles_chunk: Sequence[str | None], *, name: str, params: tuple[Any,...]
) -> list[str]:
    """Compute keys for a chunk of SMILES. The unit of work shipped to a worker.

    :param smiles_chunk: the SMILES, ``None`` for a record with no molecule.
    :param name: the registered function name.
    :param params: extra positional arguments for the function.
    :return: one key per input; ``""`` where there is no molecule or it did not parse.
    """
    from rdkit import Chem

    fn = _FUNCS[name]
    out: list[str] = []
    for smi in smiles_chunk:
        if smi is None:
            out.append("")
            continue
        mol = Chem.MolFromSmiles(smi)
        out.append("" if mol is None else fn(mol, *params))
    return out


def mapped_keys(
    name: str,
    params: tuple[Any,...],
    *,
    smiles: Sequence[str | None] | None,
    mols: Sequence[Any],
    n_jobs: int | None = 1,
) -> list[str]:
    """Per-molecule keys, cached across calls and computed in parallel when worthwhile.

    :param name: a function name registered with :func:`register`.
    :param params: extra positional arguments for that function. Part of the cache key.
    :param smiles: the input SMILES, enabling caching and the worker path. ``None`` when molecules
        were supplied directly, which forces the serial path: they have no cheap content key and
        cost more to pickle than they save.
    :param mols: the parsed molecules, used by the serial path.
    :param n_jobs: worker count. Results do not depend on it.
    :return: one key per record; ``""`` where the molecule is missing or unparseable.
    """
    fn = _FUNCS[name]
    if smiles is None:
        return ["" if m is None else fn(m, *params) for m in mols]

    key = _cache_key(name, params, smiles)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return list(hit)

    if _parallel.will_parallelize(len(smiles), n_jobs):
        keys = _parallel.ordered_map(
            functools.partial(_chunk, name=name, params=params), smiles, n_jobs=n_jobs
        )
    else:
        # in-process: reuse the already-parsed molecules rather than re-parsing the strings
        keys = ["" if m is None else fn(m, *params) for m in mols]

    with _LOCK:
        _CACHE[key] = list(keys)
        _CACHE.move_to_end(key)
        total = sum(len(v) for v in _CACHE.values())
        while total > KEY_CACHE_MAX_RECORDS and len(_CACHE) > 1:
            _, evicted = _CACHE.popitem(last=False)
            total -= len(evicted)
    return keys


def params_of(obj: Any, names: Sequence[str]) -> tuple[Any,...]:
    """Collect attributes into a hashable parameter tuple for the cache key.

    :param obj: usually the splitter instance.
    :param names: the attribute names that change the computed keys.
    :return: the attribute values, in order.
    """
    out = []
    for attr in names:
        value = getattr(obj, attr)
        out.append(dataclasses.astuple(value) if dataclasses.is_dataclass(value) else value)
    return tuple(out)
