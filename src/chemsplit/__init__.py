"""chemsplit: dataset-splitting strategies for cheminformatics machine learning.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

__all__ = [
    # -- 52 splitters across 9 families, in declared order: baseline (5), scaffold (6),
    # similarity (10), embedding (3), property (5), lineage (4), task (8), biomolecular (5),
    # protocol (6) --
    "RandomSplitter",
    "StratifiedRandomSplitter",
    "KFoldSplitter",
    "MonteCarloSplitter",
    "PredefinedSplitter",
    "MurckoScaffoldSplitter",
    "GenericScaffoldSplitter",
    "ScaffoldTreeSplitter",
    "RingSystemSplitter",
    "MatchedMolecularSeriesSplitter",
    "ActivityCliffSplitter",
    "SimilarityThresholdSplitter",
    "ButinaSplitter",
    "KMeansClusterSplitter",
    "DensityClusterSplitter",
    "SpectralSplitter",
    "MaxMinSplitter",
    "MaxDissimilaritySplitter",
    "PerimeterSplitter",
    "LeaveOneClusterOutSplitter",
    "BalancedMultiTaskSplitter",
    "UMAPClusterSplitter",
    "ProjectionSplitter",
    "LatentSpaceSplitter",
    "PropertySplitter",
    "LabelExtrapolationSplitter",
    "StratifiedDistributionSplitter",
    "MOODSplitter",
    "AdversarialSplitter",
    "TemporalSplitter",
    "SIMPDSplitter",
    "SourceSplitter",
    "PartySplitter",
    "HiSplitter",
    "LoSplitter",
    "ScaffoldHopSplitter",
    "ColdDrugSplitter",
    "ColdTargetSplitter",
    "ColdPairSplitter",
    "AVESplitter",
    "DecoyBenchmarkSplitter",
    "SequenceIdentitySplitter",
    "ProteinFamilySplitter",
    "BindingSiteSplitter",
    "DepositionDateSplitter",
    "ComplexJointSplitter",
    "GroupKFoldSplitter",
    "ThreeWaySplitter",
    "RepeatedSplitter",
    "NestedCVSplitter",
    "ExternalHoldoutSplitter",
    "ApplicabilityDomainSplitter",
    # -- core types --
    "BaseSplitter",
    "GroupSplitter",
    "SplitResult",
    "Strictness",
    # -- registry --
    "get_splitter",
    "list_splitters",
    "SPLITTER_REGISTRY",
    # -- audit --
    "audit_split",
    "LeakageReport",
    "adversarial_validation",
    "nn_similarity_profile",
    "y_scramble_control",
    # -- featurizers --
    "get_featurizer",
    "Featurizer",
    # -- exceptions --
    "ChemSplitError",
    "ParameterError",
    "ConfigurationError",
    "UnknownSplitterError",
    "UnknownFeaturizerError",
    "UnknownMetricError",
    "InputError",
    "InputKindError",
    "ColumnError",
    "EmptyInputError",
    "MoleculeParseError",
    "DuplicateRecordError",
    "LabelError",
    "InfeasibleSplitError",
    "DegenerateGroupingError",
    "ConstraintUnsatisfiableError",
    "EmptyPartitionError",
    "ScalabilityError",
    "MissingDependencyError",
    "InvariantError",
    # -- warnings --
    "ChemSplitWarning",
    "SizeToleranceWarning",
    "DuplicateWarning",
    "ParseWarning",
    "StandardizationWarning",
    "DeterminismWarning",
    "DegenerateClusterWarning",
    "SmallPartitionWarning",
    "CircularityWarning",
    "HomologyLeakWarning",
    # -- version --
    "__version__",
]

# name -> the submodule that actually defines it. Every entry not listed here (there are none
# left over once this dict is complete) would fall through to AttributeError in __getattr__.
_LAZY_SOURCE: dict[str, str] = {
    # splitters/baseline.py
    "RandomSplitter": "chemsplit.splitters.baseline",
    "StratifiedRandomSplitter": "chemsplit.splitters.baseline",
    "KFoldSplitter": "chemsplit.splitters.baseline",
    "MonteCarloSplitter": "chemsplit.splitters.baseline",
    "PredefinedSplitter": "chemsplit.splitters.baseline",
    # splitters/scaffold.py
    "MurckoScaffoldSplitter": "chemsplit.splitters.scaffold",
    "GenericScaffoldSplitter": "chemsplit.splitters.scaffold",
    "ScaffoldTreeSplitter": "chemsplit.splitters.scaffold",
    "RingSystemSplitter": "chemsplit.splitters.scaffold",
    "MatchedMolecularSeriesSplitter": "chemsplit.splitters.scaffold",
    "ActivityCliffSplitter": "chemsplit.splitters.scaffold",
    # splitters/similarity.py
    "SimilarityThresholdSplitter": "chemsplit.splitters.similarity",
    "ButinaSplitter": "chemsplit.splitters.similarity",
    "KMeansClusterSplitter": "chemsplit.splitters.similarity",
    "DensityClusterSplitter": "chemsplit.splitters.similarity",
    "SpectralSplitter": "chemsplit.splitters.similarity",
    "MaxMinSplitter": "chemsplit.splitters.similarity",
    "MaxDissimilaritySplitter": "chemsplit.splitters.similarity",
    "PerimeterSplitter": "chemsplit.splitters.similarity",
    "LeaveOneClusterOutSplitter": "chemsplit.splitters.similarity",
    "BalancedMultiTaskSplitter": "chemsplit.splitters.similarity",
    # splitters/embedding.py
    "UMAPClusterSplitter": "chemsplit.splitters.embedding",
    "ProjectionSplitter": "chemsplit.splitters.embedding",
    "LatentSpaceSplitter": "chemsplit.splitters.embedding",
    # splitters/property_.py
    "PropertySplitter": "chemsplit.splitters.property_",
    "LabelExtrapolationSplitter": "chemsplit.splitters.property_",
    "StratifiedDistributionSplitter": "chemsplit.splitters.property_",
    "MOODSplitter": "chemsplit.splitters.property_",
    "AdversarialSplitter": "chemsplit.splitters.property_",
    # splitters/lineage.py
    "TemporalSplitter": "chemsplit.splitters.lineage",
    "SIMPDSplitter": "chemsplit.splitters.lineage",
    "SourceSplitter": "chemsplit.splitters.lineage",
    "PartySplitter": "chemsplit.splitters.lineage",
    # splitters/task.py
    "HiSplitter": "chemsplit.splitters.task",
    "LoSplitter": "chemsplit.splitters.task",
    "ScaffoldHopSplitter": "chemsplit.splitters.task",
    "ColdDrugSplitter": "chemsplit.splitters.task",
    "ColdTargetSplitter": "chemsplit.splitters.task",
    "ColdPairSplitter": "chemsplit.splitters.task",
    "AVESplitter": "chemsplit.splitters.task",
    "DecoyBenchmarkSplitter": "chemsplit.splitters.task",
    # splitters/biomolecular.py
    "SequenceIdentitySplitter": "chemsplit.splitters.biomolecular",
    "ProteinFamilySplitter": "chemsplit.splitters.biomolecular",
    "BindingSiteSplitter": "chemsplit.splitters.biomolecular",
    "DepositionDateSplitter": "chemsplit.splitters.biomolecular",
    "ComplexJointSplitter": "chemsplit.splitters.biomolecular",
    # splitters/protocol.py
    "GroupKFoldSplitter": "chemsplit.splitters.protocol",
    "ThreeWaySplitter": "chemsplit.splitters.protocol",
    "RepeatedSplitter": "chemsplit.splitters.protocol",
    "NestedCVSplitter": "chemsplit.splitters.protocol",
    "ExternalHoldoutSplitter": "chemsplit.splitters.protocol",
    "ApplicabilityDomainSplitter": "chemsplit.splitters.protocol",
    # base.py
    "BaseSplitter": "chemsplit.base",
    "GroupSplitter": "chemsplit.base",
    "SplitResult": "chemsplit.base",
    "Strictness": "chemsplit.base",
    # registry.py
    "get_splitter": "chemsplit.registry",
    "list_splitters": "chemsplit.registry",
    "SPLITTER_REGISTRY": "chemsplit.registry",
    # audit.py
    "audit_split": "chemsplit.audit",
    "LeakageReport": "chemsplit.audit",
    "adversarial_validation": "chemsplit.audit",
    "nn_similarity_profile": "chemsplit.audit",
    "y_scramble_control": "chemsplit.audit",
    # featurizers/__init__.py (cheap: no rdkit/sklearn at its own module scope)
    "get_featurizer": "chemsplit.featurizers",
    "Featurizer": "chemsplit.featurizers",
    # exceptions.py (cheap: no rdkit/sklearn at its own module scope)
    "ChemSplitError": "chemsplit.exceptions",
    "ParameterError": "chemsplit.exceptions",
    "ConfigurationError": "chemsplit.exceptions",
    "UnknownSplitterError": "chemsplit.exceptions",
    "UnknownFeaturizerError": "chemsplit.exceptions",
    "UnknownMetricError": "chemsplit.exceptions",
    "InputError": "chemsplit.exceptions",
    "InputKindError": "chemsplit.exceptions",
    "ColumnError": "chemsplit.exceptions",
    "EmptyInputError": "chemsplit.exceptions",
    "MoleculeParseError": "chemsplit.exceptions",
    "DuplicateRecordError": "chemsplit.exceptions",
    "LabelError": "chemsplit.exceptions",
    "InfeasibleSplitError": "chemsplit.exceptions",
    "DegenerateGroupingError": "chemsplit.exceptions",
    "ConstraintUnsatisfiableError": "chemsplit.exceptions",
    "EmptyPartitionError": "chemsplit.exceptions",
    "ScalabilityError": "chemsplit.exceptions",
    "MissingDependencyError": "chemsplit.exceptions",
    "InvariantError": "chemsplit.exceptions",
    "ChemSplitWarning": "chemsplit.exceptions",
    "SizeToleranceWarning": "chemsplit.exceptions",
    "DuplicateWarning": "chemsplit.exceptions",
    "ParseWarning": "chemsplit.exceptions",
    "StandardizationWarning": "chemsplit.exceptions",
    "DeterminismWarning": "chemsplit.exceptions",
    "DegenerateClusterWarning": "chemsplit.exceptions",
    "SmallPartitionWarning": "chemsplit.exceptions",
    "CircularityWarning": "chemsplit.exceptions",
    "HomologyLeakWarning": "chemsplit.exceptions",
}


def __getattr__(name: str) -> Any:
    if name == "__version__":
        return __version__
    source = _LAZY_SOURCE.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(source)
    value = getattr(module, name)
    globals()[name] = value  # cache on the package module so repeat access is free
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
