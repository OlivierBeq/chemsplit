"""Splitter registry: ``SPLITTER_REGISTRY``, ``get_splitter``, ``list_splitters``.

This module is the single place that imports every concrete splitter class — no other module
should import a sibling family module directly except through here (or, for a splitter that needs
to *resolve* another splitter at run time from a string id, via a local import of
:func:`get_splitter` inside a method body, never at module scope, to avoid an import cycle).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from chemsplit.exceptions import UnknownSplitterError

if TYPE_CHECKING:
    from chemsplit.base import BaseSplitter

# ---------------------------------------------------------------------------
# Import every concrete splitter class, in family + design-ordinal order.
# ---------------------------------------------------------------------------

from chemsplit.splitters.baseline import (
    KFoldSplitter,
    MonteCarloSplitter,
    PredefinedSplitter,
    RandomSplitter,
    StratifiedRandomSplitter,
)
from chemsplit.splitters.scaffold import (
    ActivityCliffSplitter,
    GenericScaffoldSplitter,
    MatchedMolecularSeriesSplitter,
    MurckoScaffoldSplitter,
    RingSystemSplitter,
    ScaffoldTreeSplitter,
)
from chemsplit.splitters.similarity import (
    BalancedMultiTaskSplitter,
    ButinaSplitter,
    DensityClusterSplitter,
    KMeansClusterSplitter,
    LeaveOneClusterOutSplitter,
    MaxDissimilaritySplitter,
    MaxMinSplitter,
    PerimeterSplitter,
    SimilarityThresholdSplitter,
    SpectralSplitter,
)
from chemsplit.splitters.embedding import (
    LatentSpaceSplitter,
    ProjectionSplitter,
    UMAPClusterSplitter,
)
from chemsplit.splitters.property_ import (
    AdversarialSplitter,
    LabelExtrapolationSplitter,
    MOODSplitter,
    PropertySplitter,
    StratifiedDistributionSplitter,
)
from chemsplit.splitters.lineage import (
    PartySplitter,
    SIMPDSplitter,
    SourceSplitter,
    TemporalSplitter,
)
from chemsplit.splitters.task import (
    AVESplitter,
    ColdDrugSplitter,
    ColdPairSplitter,
    ColdTargetSplitter,
    DecoyBenchmarkSplitter,
    HiSplitter,
    LoSplitter,
    ScaffoldHopSplitter,
)
from chemsplit.splitters.biomolecular import (
    BindingSiteSplitter,
    ComplexJointSplitter,
    DepositionDateSplitter,
    ProteinFamilySplitter,
    SequenceIdentitySplitter,
)

# protocol.py is imported lazily inside _protocol_classes() below, not here: every protocol
# splitter resolves *other* splitters (by id) at construction or split time, and importing
# it eagerly here would be fine today (no cycle, since protocol.py itself only imports
# chemsplit.registry lazily inside method bodies) but keeping the import lazy documents the
# intended direction of the dependency (protocol -> registry, never registry -> protocol at
# module scope) so a future edit to protocol.py can't accidentally introduce a real cycle.


def _protocol_classes() -> list[type["BaseSplitter"]]:
    from chemsplit.splitters.protocol import (
        ApplicabilityDomainSplitter,
        ExternalHoldoutSplitter,
        GroupKFoldSplitter,
        NestedCVSplitter,
        RepeatedSplitter,
        ThreeWaySplitter,
    )

    return [
        GroupKFoldSplitter,
        ThreeWaySplitter,
        RepeatedSplitter,
        NestedCVSplitter,
        ExternalHoldoutSplitter,
        ApplicabilityDomainSplitter,
    ]


# Every splitter class, in declared order within each of the 9 families. This list is
# the single source of truth for SPLITTER_REGISTRY, list_splitters(), and
# chemsplit/__init__.py's __all__ ordering.
_ALL_SPLITTER_CLASSES: list[type["BaseSplitter"]] = [
    # baseline (the baseline family)
    RandomSplitter,
    StratifiedRandomSplitter,
    KFoldSplitter,
    MonteCarloSplitter,
    PredefinedSplitter,
    # scaffold (the scaffold family)
    MurckoScaffoldSplitter,
    GenericScaffoldSplitter,
    ScaffoldTreeSplitter,
    RingSystemSplitter,
    MatchedMolecularSeriesSplitter,
    ActivityCliffSplitter,
    # similarity (the similarity family)
    SimilarityThresholdSplitter,
    ButinaSplitter,
    KMeansClusterSplitter,
    DensityClusterSplitter,
    SpectralSplitter,
    MaxMinSplitter,
    MaxDissimilaritySplitter,
    PerimeterSplitter,
    LeaveOneClusterOutSplitter,
    BalancedMultiTaskSplitter,
    # embedding (the embedding family)
    UMAPClusterSplitter,
    ProjectionSplitter,
    LatentSpaceSplitter,
    # property (the property family)
    PropertySplitter,
    LabelExtrapolationSplitter,
    StratifiedDistributionSplitter,
    MOODSplitter,
    AdversarialSplitter,
    # lineage (the lineage family)
    TemporalSplitter,
    SIMPDSplitter,
    SourceSplitter,
    PartySplitter,
    # task (the task family)
    HiSplitter,
    LoSplitter,
    ScaffoldHopSplitter,
    ColdDrugSplitter,
    ColdTargetSplitter,
    ColdPairSplitter,
    AVESplitter,
    DecoyBenchmarkSplitter,
    # biomolecular (the biomolecular family)
    SequenceIdentitySplitter,
    ProteinFamilySplitter,
    BindingSiteSplitter,
    DepositionDateSplitter,
    ComplexJointSplitter,
    # protocol (the protocol family) appended by _build_registry() below, once importable.
]


def _camel_to_snake(name: str) -> str:
    """``"KMeansClusterSplitter"`` -> ``"k_means_cluster"``. Strips a trailing ``Splitter``."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    if s2.endswith("_splitter"):
        s2 = s2[: -len("_splitter")]
    return s2


def _build_registry() -> dict[str, type["BaseSplitter"]]:
    classes = list(_ALL_SPLITTER_CLASSES) + _protocol_classes()
    registry: dict[str, type["BaseSplitter"]] = {}
    seen_ids: set[str] = set()
    for cls in classes:
        sid = cls.splitter_id
        if sid in seen_ids:
            raise AssertionError(f"duplicate splitter_id {sid!r} on {cls.__name__}")
        if sid != _camel_to_snake(cls.__name__):
            raise AssertionError(
                f"splitter_id {sid!r} on {cls.__name__} does not match its class-name-derived "
                f"snake_case form {_camel_to_snake(cls.__name__)!r}"
            )
        seen_ids.add(sid)
        registry[sid] = cls
    return registry


SPLITTER_REGISTRY: dict[str, type["BaseSplitter"]] = {}


def _ensure_built() -> None:
    # Deferred build (rather than at module import time) so importing chemsplit.registry doesn't
    # force-import chemsplit.splitters.protocol before it exists during incremental development,
    # and — more importantly, permanently — so a first call to get_splitter()/list_splitters()
    # after chemsplit's own import-time budget has already elapsed doesn't pay for
    # importing every single splitter family until a caller actually asks the registry for one.
    if SPLITTER_REGISTRY:
        return
    SPLITTER_REGISTRY.update(_build_registry())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def get_splitter(name: str, **kwargs: Any) -> "BaseSplitter":
    """Instantiate a splitter by id (``"butina"``, case-insensitive) or class name
    (``"ButinaSplitter"``, case-sensitive).

    Unknown name raises :class:`~chemsplit.exceptions.UnknownSplitterError` naming the three
    closest ids by Levenshtein distance.
    """
    _ensure_built()
    if name in SPLITTER_REGISTRY:
        return SPLITTER_REGISTRY[name](**kwargs)
    for sid, cls in SPLITTER_REGISTRY.items():
        if cls.__name__ == name:
            return cls(**kwargs)
    lowered = name.lower()
    if lowered in SPLITTER_REGISTRY:
        return SPLITTER_REGISTRY[lowered](**kwargs)
    closest = sorted(SPLITTER_REGISTRY, key=lambda a: _levenshtein(lowered, a))[:3]
    raise UnknownSplitterError(
        f"unknown splitter {name!r}; closest ids: {closest}. "
        f"Use an id (e.g. 'butina') or a class name (e.g. 'ButinaSplitter')."
    )


def list_splitters(
    family: str | None = None,
    strictness: str | None = None,
    group_forming: bool | None = None,
) -> pd.DataFrame:
    """Return a DataFrame describing every registered splitter, one row per class.

    Columns: ``id, class_name, family, family_name, strictness,
    group_forming, requires_labels, requires_dates, requires_targets,
    deterministic_without_seed, extras``.
    """
    _ensure_built()
    rows = []
    for sid, cls in SPLITTER_REGISTRY.items():
        row = {
            "id": sid,
            "class_name": cls.__name__,
            "family": cls.family,
            "family_name": cls.family,
            "strictness": cls.strictness.value if hasattr(cls.strictness, "value") else cls.strictness,
            "group_forming": cls.group_forming,
            "requires_labels": cls.requires_labels,
            "requires_dates": cls.requires_dates,
            "requires_targets": cls.requires_targets,
            "deterministic_without_seed": cls.deterministic_without_seed,
            "extras": cls.extras,
        }
        rows.append(row)
    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "class_name",
            "family",
            "family_name",
            "strictness",
            "group_forming",
            "requires_labels",
            "requires_dates",
            "requires_targets",
            "deterministic_without_seed",
            "extras",
        ],
    )
    if family is not None:
        df = df[df["family"] == family]
    if strictness is not None:
        sval = strictness.value if hasattr(strictness, "value") else strictness
        df = df[df["strictness"] == sval]
    if group_forming is not None:
        df = df[df["group_forming"] == group_forming]
    return df.reset_index(drop=True)


__all__ = ["SPLITTER_REGISTRY", "get_splitter", "list_splitters"]
