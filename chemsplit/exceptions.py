"""Error and warning taxonomy for chemsplit.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = [
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
    "warn_with_details",
]


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class ChemSplitError(Exception):
    """Root of every exception chemsplit's public API can raise."""


class ParameterError(ChemSplitError, ValueError):
    """Raised in ``__init__`` when a constructor argument fails its documented validation.

    Validation is always eager (performed at construction time), never deferred to ``split()`` —
    this deliberately does not follow sklearn's lazy ``check_estimator`` convention.
    """


class ConfigurationError(ParameterError):
    """Raised when individually-valid parameters are jointly impossible.

    Examples: ``valid_size > 0`` combined with a call to ``split()`` (use
    ``split_with_validation()``/``split_result()`` instead); ``n_splits > 1`` with
    ``valid_size > 0``; ``on_duplicates="group"`` on a non-group-forming splitter.
    """


class UnknownSplitterError(ParameterError):
    """Raised by the registry when a splitter id/class-name cannot be resolved."""


class UnknownFeaturizerError(ParameterError):
    """Raised by ``get_featurizer`` when a featurizer alias cannot be resolved."""


class UnknownMetricError(ParameterError):
    """Raised when an unrecognised ``metric`` string is supplied."""


class InputError(ChemSplitError, ValueError):
    """Root of every input-shape/content validation error."""


class InputKindError(InputError):
    """Raised when the detected kind of ``X`` is not in the splitter's ``accepts`` set.

    The message names both the detected kind and the accepted set.
    """


class ColumnError(InputError):
    """Raised when a DataFrame column selector (``smiles_col`` etc.) names a missing column."""


class EmptyInputError(InputError):
    """Raised when ``X`` is empty, or ``n < 2``."""


class MoleculeParseError(InputError):
    """Raised when ``on_parse_error="raise"`` (the default) and >= 1 SMILES failed to parse.

    The message lists at most the first 20 ``(index, smiles)`` pairs and the total failure count.
    """


class DuplicateRecordError(InputError):
    """Raised when ``on_duplicates="raise"`` and duplicate records were found."""


class LabelError(InputError):
    """Raised when ``requires_labels`` is True and ``y`` is missing, mis-shaped, or contains
    forbidden NaNs; also raised for date/target presence-and-length failures driven by
    ``requires_dates``/``requires_targets``.
    """


class InfeasibleSplitError(ChemSplitError, RuntimeError):
    """Root of every error raised when a requested split cannot be produced at all."""


class DegenerateGroupingError(InfeasibleSplitError):
    """Raised when a grouping produced exactly 1 group, or exactly ``n`` groups (every record its
    own group) and the splitter documents that configuration as degenerate.

    The message includes the first five entries of the group-count histogram.
    """


class ConstraintUnsatisfiableError(InfeasibleSplitError):
    """Raised when a hard constraint cannot be met (e.g. a similarity-threshold constraint, or an
    infeasible balance-assignment problem).

    The message states the binding constraint and the closest achievable value.
    """


class EmptyPartitionError(InfeasibleSplitError):
    """Raised when a non-``discard`` partition with a positive target size came out empty."""


class ScalabilityError(ChemSplitError, RuntimeError):
    """Raised by the memory guard, or any other documented ``n`` ceiling."""


class MissingDependencyError(ChemSplitError, ImportError):
    """Raised when an optional extra required by a splitter/feature is not installed."""

    def __init__(self, splitter_name: str, extra: str) -> None:
        self.splitter_name = splitter_name
        self.extra = extra
        message = (
            f"{splitter_name} requires the '{extra}' extra. "
            f"Install with: pip install 'chemsplit[{extra}]'"
        )
        super().__init__(message)


class InvariantError(ChemSplitError, AssertionError):
    """Raised when a ``SplitResult`` invariant fails.

    This always indicates a bug in chemsplit itself, never a data condition. The message instructs
    the caller to file an issue and includes ``splitter_id``, ``params``, and ``n_records``.
    """

    def __init__(
        self,
        message: str,
        *,
        splitter_id: str,
        params: dict[str, Any],
        n_records: int,
    ) -> None:
        self.splitter_id = splitter_id
        self.params = params
        self.n_records = n_records
        full_message = (
            f"{message}\n"
            "This is a bug in chemsplit — please file an issue at "
            "https://github.com/OlivierBeq/chemsplit/issues including the details below.\n"
            f"splitter_id={splitter_id!r} n_records={n_records!r} params={params!r}"
        )
        super().__init__(full_message)


# ---------------------------------------------------------------------------
# Warning hierarchy
# ---------------------------------------------------------------------------


class ChemSplitWarning(UserWarning):
    """Root of every warning chemsplit's public API can emit.

    Every warning instance carries a ``.details: dict`` attribute of machine-readable fields, and
    must be emitted with ``stacklevel=2`` (relative to the code raising it) — use
    :func:`warn_with_details` rather than a bare ``warnings.warn`` call to get this right
    automatically.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = details if details is not None else {}


class SizeToleranceWarning(ChemSplitWarning):
    """A group-forming splitter's realised partition sizes deviated from their targets by more
    than ``size_tolerance``."""


class DuplicateWarning(ChemSplitWarning):
    """``on_duplicates="warn"`` (the default) and duplicate records were found."""


class ParseWarning(ChemSplitWarning):
    """``on_parse_error="discard"`` and >= 1 SMILES failed to parse."""


class StandardizationWarning(ChemSplitWarning):
    """``standardize=False`` and >= 1 record's stripped form differs from its input form."""


class DeterminismWarning(ChemSplitWarning):
    """A computation that is nominally deterministic hit a condition that weakens that guarantee
    (e.g. degenerate eigenvalues in spectral clustering, or a user forcing ``n_jobs>1`` on UMAP)."""


class DegenerateClusterWarning(ChemSplitWarning):
    """A clustering/grouping step produced a near-degenerate result (e.g. one dominant cluster)
    that is not yet bad enough to raise :class:`DegenerateGroupingError`."""


class SmallPartitionWarning(ChemSplitWarning):
    """A non-empty partition ended with < 10 records, or < 1% of ``n``, whichever is larger."""


class CircularityWarning(ChemSplitWarning):
    """The embedding-family latent-space splitter's ``independence_declared=False`` warning."""


class HomologyLeakWarning(ChemSplitWarning):
    """The task-family cold-target splitter's ``target_grouper=None`` warning."""


def warn_with_details(warning: ChemSplitWarning) -> None:
    """Emit a :class:`ChemSplitWarning` instance with the correct stack level.

    Callers should use this helper rather than a bare ``warnings.warn(...)`` call: this function
    adds one stack frame relative to the code that actually detected the condition, so it uses
    ``stacklevel=3`` internally to make the warning point at the caller's caller.
    """
    warnings.warn(warning, stacklevel=3)
