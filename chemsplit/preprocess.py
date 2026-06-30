"""Input handling, standardisation, deduplication, replicate aggregation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from chemsplit.exceptions import (
    ColumnError,
    ConfigurationError,
    DuplicateRecordError,
    DuplicateWarning,
    InputKindError,
    MoleculeParseError,
    ParameterError,
    ParseWarning,
    StandardizationWarning,
    warn_with_details,
)

__all__ = [
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
]


# ---------------------------------------------------------------------------
# / input kind detection
# ---------------------------------------------------------------------------

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
    """detect the kind of ``X``. ``x_kind`` (the ``X_kind`` splitter keyword) resolves the
    only genuine ambiguity — SMILES strings vs. protein sequences, both ``Sequence[str]``."""
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
    if x_kind == "dataframe":
        return len(X)
    import scipy.sparse

    if isinstance(X, np.ndarray) or scipy.sparse.issparse(X):
        return X.shape[0]
    return len(X)


def resolve_dataframe_columns(df: pd.DataFrame, **selectors: str | None) -> dict[str, Any]:
    """resolve DataFrame column selectors. Missing columns raise ``ColumnError``. If
    ``smiles_col`` is omitted, raise rather than guess a ``"smiles"``-named column."""
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


# ---------------------------------------------------------------------------
# Molecule parsing
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ParseFailure:
    index: int
    smiles: str
    rdkit_error_log: str = ""


def parse_smiles(s: str) -> Any:
    """Parse one SMILES string. Returns ``None`` on failure (the caller is responsible for
    recording a :class:`ParseFailure` — this function does not raise per-record)."""
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


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class StandardizeConfig:
    strip_salts: bool = True
    normalize_charges: bool = True
    canonical_tautomer: bool = False
    strip_isotopes: bool = False
    stereo: Literal["keep", "strip", "strip_unassigned"] = "keep"


def standardize(mol: Any, config: StandardizeConfig | None = None) -> Any:
    """the 6-step standardisation pipeline, applied in exact order."""
    if config is None:
        config = StandardizeConfig()
    from rdkit import Chem
    from rdkit.Chem import rdMolStandardize

    m = Chem.Mol(mol)
    Chem.SanitizeMol(m)

    if config.strip_salts:
        chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        m = chooser.choose(m)
        Chem.SanitizeMol(m)

    if config.normalize_charges:
        m = rdMolStandardize.Normalizer().normalize(m)
        m = rdMolStandardize.Uncharger().uncharge(m)
        Chem.SanitizeMol(m)

    if config.canonical_tautomer:
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(1000)
        enumerator.SetMaxTransforms(1000)
        m = enumerator.Canonicalize(m)
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


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def dedup_key(
    mol: Any, config: StandardizeConfig | None = None
) -> tuple[str, bool]:
    """Returns ``(key, used_fallback)``."""
    from rdkit import Chem

    m = standardize(mol, config)
    key = Chem.MolToInchiKey(m)
    if key:
        return key, False
    cfg = config or StandardizeConfig()
    key = Chem.MolToSmiles(m, canonical=True, isomericSmiles=(cfg.stereo != "strip"))
    return key, True


def find_duplicates(
    mols: Sequence[Any], config: StandardizeConfig | None = None
) -> tuple[dict[str, list[int]], list[int]]:
    """Returns ``(key -> sorted indices sharing that key, indices that used the InChIKey
    fallback)``. Only keys shared by >= 2 records are meaningful duplicates, but all keys are
    returned; callers filter."""
    keyed: dict[str, list[int]] = {}
    fallbacks: list[int] = []
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        key, used_fallback = dedup_key(mol, config)
        keyed.setdefault(key, []).append(i)
        if used_fallback:
            fallbacks.append(i)
    return keyed, fallbacks


# ---------------------------------------------------------------------------
# Replicate aggregation
# ---------------------------------------------------------------------------


def aggregate_replicates(
    df: pd.DataFrame,
    key_col: str = "inchikey",
    value_col: str = "y",
    method: Literal["median", "mean", "max", "min", "first", "drop_conflicting"] = "median",
    max_spread: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """aggregate replicate measurements sharing ``key_col``. Not part of any splitter — a
    standalone preprocessing utility. Returns ``(aggregated, dropped)``."""
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


# ---------------------------------------------------------------------------
# The pipeline glue used by BaseSplitter._run (not itself part of this project's public surface)
# ---------------------------------------------------------------------------


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
) -> PipelineResult:
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
            # "ignore": mols[i] stays None; downstream featurizers must treat None as all-zero
    elif x_kind == "mol":
        mols = list(X)
        smiles_list = None

    if mols is not None and not standardize:
        # standardisation is off by default; warn once if the stripped form differs.
        cfg = config or StandardizeConfig()
        n_differs = 0
        for mol in mols:
            if mol is None:
                continue
            try:
                from rdkit import Chem

                std = globals()["standardize"](mol, cfg)
                if Chem.MolToSmiles(std) != Chem.MolToSmiles(mol):
                    n_differs += 1
            except Exception:
                continue
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
        keyed, _fallbacks = find_duplicates(mols, config)
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
