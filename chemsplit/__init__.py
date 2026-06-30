"""chemsplit: dataset-splitting strategies for cheminformatics machine learning.
"""

from __future__ import annotations

__version__ = "0.1.0"

# The full public surface (splitter classes, BaseSplitter, GroupSplitter, SplitResult,
# Strictness, get_splitter, list_splitters, SPLITTER_REGISTRY, audit_split, LeakageReport,
# adversarial_validation, nn_similarity_profile, y_scramble_control, get_featurizer, Featurizer,
# every exception class) is added to __all__ once every splitter family module exists.

__all__: list[str] = [
    "__version__",
]
