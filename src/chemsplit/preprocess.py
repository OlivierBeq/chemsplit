"""Input handling, standardisation, deduplication, replicate aggregation.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import threading
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from chemsplit import _parallel
from chemsplit.exceptions import (
    ColumnError,
    ConfigurationError,
    DuplicateRecordError,
    DuplicateWarning,
    InputKindError,
    InvariantError,
    MoleculeParseError,
    ParseWarning,
    StandardizationWarning,
    warn_with_details,
)

__all__ = [
    "DIGEST_CACHE_MAX_RECORDS",
    "StandardizeConfig",
    "ParseFailure",
    "PipelineResult",
    "detect_input_kind",
    "input_length",
    "resolve_dataframe_columns",
    "parse_smiles",
    "standardize",
    "dedup_key",
    "find_duplicates",
    "aggregate_replicates",
    "run_pipeline",
    "clear_digest_cache",
    "set_digest_cache_enabled",
]



_DATAFRAME_SELECTORS = (
    "smiles_col",
    "label_col",
    "date_col",
    "target_col",
    "sequence_col",
    "group_col",
    "weight_col",
)


def detect_input_kind(X: Any, x_kind: str | None = None) -> str:
    """Detect what kind of records ``X`` holds.

    The only genuine ambiguity is SMILES against protein sequences, since both arrive as
    ``Sequence[str]``; ``x_kind`` resolves it.

    :param X: the records.
    :param x_kind: the caller's ``X_kind`` hint, or ``None`` to infer.
    :raises InputKindError: if ``X`` is of no recognised kind, or ``x_kind`` names an unknown
        one.
    :raises EmptyInputError: if ``X`` is empty.
    :return: one of ``"smiles"``, ``"mol"``, ``"features"``, ``"sequences"`` or
        ``"interactions"``.
    """
    if x_kind is not None:
        return x_kind

    if isinstance(X, pd.DataFrame):
        return "dataframe"

    import scipy.sparse

    if isinstance(X, np.ndarray) and X.ndim == 2:
        return "features"
    if scipy.sparse.issparse(X):
        return "features"

    try:
        items = list(X)
    except TypeError as exc:
        raise InputKindError(f"cannot determine the kind of X: {exc}") from exc

    if len(items) == 0:
        from chemsplit.exceptions import EmptyInputError

        raise EmptyInputError("X is empty")

    from rdkit import Chem

    if all(isinstance(item, Chem.rdchem.Mol) for item in items):
        return "mol"
    if all(isinstance(item, str) for item in items):
        return "smiles"
    if all(isinstance(item, tuple) and len(item) == 2 for item in items):
        return "interactions"

    raise InputKindError(
        f"X has a mixed or unrecognised element type; first element type is "
        f"{type(items[0]).__name__}"
    )


def input_length(X: Any, x_kind: str) -> int:
    """Count the records in ``X``.

    :param X: the records.
    :param x_kind: the detected input kind, which decides how length is read.
    :return: the record count.
    """
    if x_kind == "dataframe":
        return len(X)
    import scipy.sparse

    if isinstance(X, np.ndarray) or scipy.sparse.issparse(X):
        return X.shape[0]
    return len(X)


def resolve_dataframe_columns(df: pd.DataFrame, **selectors: str | None) -> dict[str, Any]:
    """Resolve column selectors against a frame.

    A missing ``smiles_col`` raises rather than guessing a column named ``"smiles"``.

    :param df: the frame.
    :param selectors: selector name to column name, e.g. ``smiles_col="SMILES"``.
    :raises ColumnError: if a named column is absent from ``df``.
    :return: selector name to the resolved column, with unset selectors omitted.
    """
    if "smiles_col" not in selectors or selectors.get("smiles_col") is None:
        raise ColumnError(
            "DataFrame input requires an explicit smiles_col= keyword; chemsplit never guesses "
            "column names"
        )
    out: dict[str, Any] = {}
    for key, col in selectors.items():
        if col is None:
            out[key] = None
            continue
        if col not in df.columns:
            raise ColumnError(f"{key}={col!r} is not a column of the input DataFrame")
        out[key] = df[col].to_numpy()
    return out




@dataclasses.dataclass(frozen=True, slots=True)
class ParseFailure:
    index: int
    smiles: str
    rdkit_error_log: str = ""


def parse_smiles(s: str) -> Any:
    """Parse one SMILES string.

    Per-record failures are the caller's to record as a :class:`ParseFailure`, so this never
    raises.

    :param s: the SMILES string.
    :return: the molecule, or ``None`` if it did not parse.
    """
    from rdkit import Chem

    return Chem.MolFromSmiles(s, sanitize=True)


def _parse_all(smiles_list: Sequence[str]) -> tuple[list[Any], list[ParseFailure]]:
    mols: list[Any] = []
    failures: list[ParseFailure] = []
    for i, s in enumerate(smiles_list):
        mol = parse_smiles(s)
        mols.append(mol)
        if mol is None:
            failures.append(ParseFailure(index=i, smiles=s))
    return mols, failures




@dataclasses.dataclass(frozen=True, slots=True)
class StandardizeConfig:
    strip_salts: bool = True
    normalize_charges: bool = True
    canonical_tautomer: bool = False
    strip_isotopes: bool = False
    stereo: Literal["keep", "strip", "strip_unassigned"] = "keep"


_HELPERS = threading.local()


def _helpers() -> dict[str, Any]:
    """Return this thread's ``rdMolStandardize`` helper instances, building them on first use.

    Constructing these is expensive (``Normalizer`` loads a transform catalogue) and they carry no
    per-molecule state, so reuse makes standardisation ~3x faster at identical output.

    Thread-local, not module-global: RDKit does not document them as thread-safe, and the code this
    replaced built a fresh set per call.

    :return: a mapping with the ``chooser``, ``normalizer``, ``uncharger`` and ``tautomer``
        helpers.
    """
    cached: dict[str, Any] | None = getattr(_HELPERS, "value", None)
    if cached is None:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        tautomer = rdMolStandardize.TautomerEnumerator()
        tautomer.SetMaxTautomers(1000)
        tautomer.SetMaxTransforms(1000)
        cached = {
            "chooser": rdMolStandardize.LargestFragmentChooser(preferOrganic=True),
            "normalizer": rdMolStandardize.Normalizer(),
            "uncharger": rdMolStandardize.Uncharger(),
            "tautomer": tautomer,
        }
        _HELPERS.value = cached
    return cached


def standardize(mol: Any, config: StandardizeConfig | None = None) -> Any:
    """Run the six-step standardisation pipeline, in order.

    :param mol: the molecule.
    :param config: which steps to apply, or ``None`` for the defaults.
    :return: the standardised molecule, or ``None`` if a step failed.
    """
    if config is None:
        config = StandardizeConfig()
    from rdkit import Chem

    helpers = _helpers()
    m = Chem.Mol(mol)
    Chem.SanitizeMol(m)

    if config.strip_salts:
        m = helpers["chooser"].choose(m)
        Chem.SanitizeMol(m)

    if config.normalize_charges:
        m = helpers["normalizer"].normalize(m)
        m = helpers["uncharger"].uncharge(m)
        Chem.SanitizeMol(m)

    if config.canonical_tautomer:
        m = helpers["tautomer"].Canonicalize(m)
        Chem.SanitizeMol(m)

    if config.strip_isotopes:
        for atom in m.GetAtoms():
            atom.SetIsotope(0)
        Chem.SanitizeMol(m)

    if config.stereo == "strip":
        Chem.RemoveStereochemistry(m)
    elif config.stereo == "strip_unassigned":
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
        for atom in m.GetAtoms():
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and not atom.HasProp(
                "_ChiralityPossible"
            ):
                continue
        # remove stereo only on centres RDKit reports as unassigned (possible but not set)
        unassigned_atoms = Chem.FindMolChiralCenters(
            m, includeUnassigned=True, useLegacyImplementation=False
        )
        for atom_idx, label in unassigned_atoms:
            if label == "?":
                m.GetAtomWithIdx(atom_idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    return m




def dedup_key(
    mol: Any, config: StandardizeConfig | None = None
) -> tuple[str, bool]:
    """Compute a molecule's deduplication key.

    :param mol: the molecule.
    :param config: standardisation settings, or ``None`` for the defaults.
    :return: the key, and whether the InChIKey fallback was used because the primary key could
        not be computed.
    """
    return _key_of_standardized(standardize(mol, config), config)


def _key_of_standardized(
    std: Any, config: StandardizeConfig | None = None
) -> tuple[str, bool]:
    """Derive a deduplication key from an already-standardised molecule.

    Split out of :func:`dedup_key` so ``run_pipeline`` standardises once and feeds both consumers.

    :param std: the standardised molecule.
    :param config: standardisation settings, or ``None`` for the defaults.
    :return: the key, and whether the canonical-SMILES fallback was used.
    """
    from rdkit import Chem

    key = Chem.MolToInchiKey(std)
    if key:
        return key, False
    cfg = config or StandardizeConfig()
    key = Chem.MolToSmiles(std, canonical=True, isomericSmiles=(cfg.stereo != "strip"))
    return key, True


@dataclasses.dataclass(frozen=True, slots=True)
class _Digest:
    """Everything ``run_pipeline`` needs from one molecule's standardised form.

    Plain scalars only: this crosses a process boundary, and returning a molecule would cost more
    than the standardisation it saves.
    """

    key: str
    used_fallback: bool
    differs: bool


def _digest_smiles_chunk(
    smiles_chunk: Sequence[str],
    *,
    config: StandardizeConfig,
    want_differs: bool,
) -> list[_Digest | None]:
    """Standardise a chunk of SMILES and reduce each to a :class:`_Digest`.

    The unit of work for :func:`chemsplit._parallel.ordered_map`.

    :param smiles_chunk: the SMILES strings.
    :param config: standardisation settings.
    :param want_differs: also report whether standardisation changed the molecule; costs two extra
        canonicalisations and is only needed for :class:`StandardizationWarning`.
    :return: one digest per input, or ``None`` where it did not parse or standardisation failed.
    """
    from rdkit import Chem

    out: list[_Digest | None] = []
    for smi in smiles_chunk:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            out.append(None)
            continue
        try:
            std = standardize(mol, config)
            key, used_fallback = _key_of_standardized(std, config)
            differs = bool(want_differs and Chem.MolToSmiles(std) != Chem.MolToSmiles(mol))
        except Exception:
            out.append(None)
            continue
        out.append(_Digest(key=key, used_fallback=used_fallback, differs=differs))
    return out


def _digest_mols(
    mols: Sequence[Any], config: StandardizeConfig, *, want_differs: bool
) -> list[_Digest | None]:
    """Serial :func:`_digest_smiles_chunk` equivalent for molecules supplied directly.

    Not parallelised: pickling molecules to a worker costs more than it saves.

    :param mols: the molecules, possibly containing ``None``.
    :param config: standardisation settings.
    :param want_differs: as for :func:`_digest_smiles_chunk`.
    :return: one digest per molecule, or ``None`` where it was absent or failed.
    """
    from rdkit import Chem

    out: list[_Digest | None] = []
    for mol in mols:
        if mol is None:
            out.append(None)
            continue
        try:
            std = standardize(mol, config)
            key, used_fallback = _key_of_standardized(std, config)
            differs = bool(want_differs and Chem.MolToSmiles(std) != Chem.MolToSmiles(mol))
        except Exception:
            out.append(None)
            continue
        out.append(_Digest(key=key, used_fallback=used_fallback, differs=differs))
    return out


_DIGEST_CACHE: collections.OrderedDict[str, list[_Digest | None]] = collections.OrderedDict()
_DIGEST_CACHE_LOCK = threading.Lock()
_DIGEST_CACHE_ENABLED = True

DIGEST_CACHE_MAX_RECORDS = 1_000_000
"""Records the digest cache may hold before evicting oldest. Bounded by records, not entries, so
one large dataset cannot pin the budget."""


def set_digest_cache_enabled(enabled: bool) -> None:
    """Turn the standardisation cache on or off, and clear it when turning it off.

    :param enabled: whether to cache.
    """
    global _DIGEST_CACHE_ENABLED
    with _DIGEST_CACHE_LOCK:
        _DIGEST_CACHE_ENABLED = enabled
        if not enabled:
            _DIGEST_CACHE.clear()


def clear_digest_cache() -> None:
    """Drop every cached standardisation result."""
    with _DIGEST_CACHE_LOCK:
        _DIGEST_CACHE.clear()


def _digest_cache_key(
    smiles: Sequence[str], config: StandardizeConfig, want_differs: bool
) -> str:
    """Content hash identifying a standardisation result.

    Keyed on content, not ``id()``: the wrappers pass an equal-but-not-identical list per fold, so
    an identity key would never hit. Hashing 200k SMILES costs milliseconds.

    :param smiles: the input SMILES.
    :param config: standardisation settings, which change the result.
    :param want_differs: also part of the result, so part of the key.
    :return: a hex digest.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((dataclasses.astuple(config), want_differs)).encode())
    h.update(b"\x00")
    for smi in smiles:
        h.update(smi.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cached_digests(
    smiles: Sequence[str],
    config: StandardizeConfig,
    *,
    want_differs: bool,
    n_jobs: int | None,
) -> list[_Digest | None]:
    """Standardisation digests for ``smiles``, reusing an earlier identical computation.

    Caching the *digests*, not the whole :class:`PipelineResult`, keeps every warning downstream of
    the cache, so a hit still re-emits them and is indistinguishable from a cold call.

    :param smiles: the input SMILES.
    :param config: standardisation settings.
    :param want_differs: whether the diagnostic comparison is needed.
    :param n_jobs: worker count for a cold computation.
    :return: one digest per input, or ``None`` where the record did not parse or failed.
    """
    if not _DIGEST_CACHE_ENABLED:
        return _compute_digests(smiles, config, want_differs=want_differs, n_jobs=n_jobs)

    key = _digest_cache_key(smiles, config, want_differs)
    with _DIGEST_CACHE_LOCK:
        hit = _DIGEST_CACHE.get(key)
        if hit is not None:
            _DIGEST_CACHE.move_to_end(key)
            return list(hit)

    digests = _compute_digests(smiles, config, want_differs=want_differs, n_jobs=n_jobs)

    with _DIGEST_CACHE_LOCK:
        _DIGEST_CACHE[key] = list(digests)
        _DIGEST_CACHE.move_to_end(key)
        total = sum(len(v) for v in _DIGEST_CACHE.values())
        while total > DIGEST_CACHE_MAX_RECORDS and len(_DIGEST_CACHE) > 1:
            _, evicted = _DIGEST_CACHE.popitem(last=False)
            total -= len(evicted)
    return digests


def _compute_digests(
    smiles: Sequence[str],
    config: StandardizeConfig,
    *,
    want_differs: bool,
    n_jobs: int | None,
) -> list[_Digest | None]:
    """Standardise ``smiles``, in worker processes when worthwhile.

    :param smiles: the input SMILES.
    :param config: standardisation settings.
    :param want_differs: whether the diagnostic comparison is needed.
    :param n_jobs: worker count. Results do not depend on it.
    :return: one digest per input.
    """
    return _parallel.ordered_map(
        functools.partial(_digest_smiles_chunk, config=config, want_differs=want_differs),
        smiles,
        n_jobs=n_jobs,
    )


def _group_by_key(digests: Sequence[_Digest | None]) -> dict[str, list[int]]:
    """Group record indices by deduplication key, in first-appearance order.

    :param digests: the per-record digests, ``None`` where there is no key.
    :return: key to the ascending indices sharing it.
    """
    keyed: dict[str, list[int]] = {}
    for i, digest in enumerate(digests):
        if digest is None:
            continue
        keyed.setdefault(digest.key, []).append(i)
    return keyed


def find_duplicates(
    mols: Sequence[Any], config: StandardizeConfig | None = None
) -> tuple[dict[str, list[int]], list[int]]:
    """Group molecules by deduplication key.

    Every key is returned, not only the shared ones, so callers filter for themselves.

    :param mols: the molecules.
    :param config: standardisation settings, or ``None`` for the defaults.
    :return: key to the sorted indices sharing it, plus the indices that fell back to an
        InChIKey.
    """
    return _find_duplicates_from_standardized(
        [None if m is None else standardize(m, config) for m in mols], config
    )


def _find_duplicates_from_standardized(
    std_mols: Sequence[Any], config: StandardizeConfig | None = None
) -> tuple[dict[str, list[int]], list[int]]:
    """Group already-standardised molecules by deduplication key.

    :param std_mols: the standardised molecules, ``None`` where the input did not parse.
    :param config: standardisation settings, or ``None`` for the defaults.
    :return: key to the sorted indices sharing it, plus the indices that fell back to canonical
        SMILES.
    """
    keyed: dict[str, list[int]] = {}
    fallbacks: list[int] = []
    for i, std in enumerate(std_mols):
        if std is None:
            continue
        key, used_fallback = _key_of_standardized(std, config)
        keyed.setdefault(key, []).append(i)
        if used_fallback:
            fallbacks.append(i)
    return keyed, fallbacks




def aggregate_replicates(
    df: pd.DataFrame,
    key_col: str = "inchikey",
    value_col: str = "y",
    method: Literal["median", "mean", "max", "min", "first", "drop_conflicting"] = "median",
    max_spread: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse replicate measurements that share a key.

    A standalone utility: no splitter calls it.

    :param df: the frame to aggregate.
    :param key_col: the column identifying replicates.
    :param value_col: the column to aggregate.
    :param method: how replicates are combined, or ``"drop_conflicting"`` to drop any key whose
        spread exceeds ``max_spread``.
    :param max_spread: the spread above which replicates count as conflicting, or ``None`` for
        no limit.
    :raises ColumnError: if ``key_col`` or ``value_col`` is absent.
    :raises ParameterError: if ``method`` is unknown, or ``max_spread`` is negative.
    :return: the aggregated frame, and the rows dropped from it.
    """
    grouped = df.groupby(key_col, sort=False)
    spreads = grouped[value_col].agg(lambda s: s.max() - s.min())
    to_drop_keys: set[Any] = set()
    if max_spread is not None:
        to_drop_keys |= set(spreads[spreads > max_spread].index)
    if method == "drop_conflicting":
        nunique = grouped[value_col].nunique()
        to_drop_keys |= set(nunique[nunique > 1].index)

    dropped = df[df[key_col].isin(to_drop_keys)].copy()
    kept = df[~df[key_col].isin(to_drop_keys)]

    agg_func = {
        "median": "median",
        "mean": "mean",
        "max": "max",
        "min": "min",
        "first": "first",
        "drop_conflicting": "first",
    }[method]
    other_cols = [c for c in df.columns if c not in (key_col, value_col)]
    aggregated = kept.groupby(key_col, sort=False, as_index=False).agg(
        {value_col: agg_func, **{c: "first" for c in other_cols}}
    )
    return aggregated, dropped


@dataclasses.dataclass(frozen=True, slots=True)
class PipelineResult:
    mols: list[Any] | None
    smiles: list[str] | None
    forced_discard: list[int]
    dedup_group_labels: Any | None  # IndexArray | None, avoids a chemsplit.types import cycle


def run_pipeline(
    X: Any,
    y: Any,
    groups: Any,
    *,
    x_kind: str,
    on_parse_error: Literal["raise", "discard", "ignore"] = "raise",
    standardize: bool = False,
    on_duplicates: Literal["warn", "raise", "ignore", "group"] = "warn",
    group_forming: bool = False,
    config: StandardizeConfig | None = None,
    n_jobs: int | None = 1,
) -> PipelineResult:
    """Parse, standardise and deduplicate ``X`` before a split.

    Called by ``BaseSplitter._run``; splitters do not call it themselves.

    :param X: the records.
    :param y: labels, or ``None``.
    :param groups: precomputed group labels, or ``None``.
    :param x_kind: the detected input kind.
    :param on_parse_error: raise on an unparseable record, discard it, or leave it as ``None``
        for downstream code to treat as all-zero features.
    :param standardize: run the standardisation pipeline over the parsed molecules.
    :param on_duplicates: warn about duplicate records, raise, ignore them, or keep them
        together as one group.
    :param group_forming: whether the calling splitter forms groups, which decides whether
        duplicates can be grouped.
    :param config: standardisation settings, or ``None`` for the defaults.
    :param n_jobs: worker count for the per-molecule standardisation pass. Results do not
        depend on it: chunks are contiguous and reassembled in input order.
    :raises MoleculeParseError: if a record fails to parse and ``on_parse_error="raise"``.
    :raises DuplicateRecordError: if duplicates exist and ``on_duplicates="raise"``.
    :raises ConfigurationError: if ``on_duplicates="group"`` is asked of a splitter that forms
        no groups.
    :return: the parsed molecules and SMILES, the labels, the records forced into ``discard``,
        and the duplicate group labels where duplicates were grouped.
    """
    if on_duplicates == "group" and not group_forming:
        raise ConfigurationError(
            "on_duplicates='group' is only legal for group-forming splitters"
        )

    forced_discard: list[int] = []
    smiles_list: list[str] | None = None
    mols: list[Any] | None = None

    if x_kind == "smiles":
        smiles_list = list(X)
        mols, failures = _parse_all(smiles_list)
        if failures:
            if on_parse_error == "raise":
                shown = failures[:20]
                raise MoleculeParseError(
                    f"{len(failures)} SMILES failed to parse (showing up to 20): "
                    + ", ".join(f"({f.index}, {f.smiles!r})" for f in shown)
                )
            if on_parse_error == "discard":
                forced_discard.extend(f.index for f in failures)
                warn_with_details(
                    ParseWarning(
                        f"{len(failures)} record(s) failed to parse and were moved to discard",
                        details={"count": len(failures), "indices": [f.index for f in failures]},
                    )
                )
            # ignore: mols[i] stays None, and featurizers read that as all-zero
    elif x_kind == "mol":
        mols = list(X)
        smiles_list = None

    # Standardise once per molecule and share it with both consumers below: they used to
    # standardise independently, doubling the dominant cost of every split.
    digests: list[_Digest | None] | None = None
    if mols is not None and (not standardize or on_duplicates != "ignore"):
        cfg_once = config or StandardizeConfig()
        want_differs = not standardize
        if smiles_list is not None and (
            _DIGEST_CACHE_ENABLED or _parallel.will_parallelize(len(smiles_list), n_jobs)
        ):
            # Work from the SMILES: cheap to ship to workers and to hash for the cache. The
            # re-parse only pays off when one of those applies -- charging it to a plain
            # single-threaded uncached call was a 36% regression.
            digests = _cached_digests(
                smiles_list, cfg_once, want_differs=want_differs, n_jobs=n_jobs
            )
        else:
            digests = _digest_mols(mols, cfg_once, want_differs=want_differs)

        if on_duplicates != "ignore":
            # A digest is None for an unparseable record (fine) or a standardisation failure (not),
            # so re-run the offenders serially to raise exactly what the old code raised.
            for i, (mol, digest) in enumerate(zip(mols, digests, strict=True)):
                if mol is not None and digest is None:
                    globals()["standardize"](mol, cfg_once)
                    raise InvariantError(  # pragma: no cover - the line above must raise
                        f"record {i} failed standardisation in a worker but not in the parent"
                    )

    if mols is not None and not standardize:
        # standardisation is off by default, so warn once if stripping would change anything
        assert digests is not None
        n_differs = sum(1 for d in digests if d is not None and d.differs)
        if n_differs:
            warn_with_details(
                StandardizationWarning(
                    f"{n_differs} record(s)' standardised form differs from their input form, "
                    "but standardize=False so the input form was used",
                    details={"count": n_differs},
                )
            )

    dedup_group_labels = None
    if mols is not None and on_duplicates != "ignore":
        assert digests is not None
        keyed = _group_by_key(digests)
        dup_keys = {k: v for k, v in keyed.items() if len(v) > 1}
        if dup_keys:
            n_affected = sum(len(v) for v in dup_keys.values())
            if on_duplicates == "raise":
                raise DuplicateRecordError(
                    f"{len(dup_keys)} duplicate key(s) affecting {n_affected} record(s)"
                )
            if on_duplicates == "warn":
                warn_with_details(
                    DuplicateWarning(
                        f"{len(dup_keys)} duplicate key(s) affecting {n_affected} record(s)",
                        details={"n_duplicate_keys": len(dup_keys), "n_affected": n_affected},
                    )
                )
            elif on_duplicates == "group":
                from chemsplit._unionfind import dense_label_encode

                keys_per_record = [None] * len(mols)
                for key, idxs in keyed.items():
                    for i in idxs:
                        keys_per_record[i] = key
                # records with no key (unparsed) get a unique singleton key
                for i, k in enumerate(keys_per_record):
                    if k is None:
                        keys_per_record[i] = f"__unparsed_{i}__"
                dedup_group_labels = dense_label_encode(keys_per_record)

    return PipelineResult(
        mols=mols,
        smiles=smiles_list,
        forced_discard=forced_discard,
        dedup_group_labels=dedup_group_labels,
    )
