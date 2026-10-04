"""Core types and the splitter API: ``SplitResult``, ``BaseSplitter``, ``GroupSplitter``, and
the shared group-to-partition routines.

A splitter id is the snake_case form of its class name, e.g. ``"butina"``, which
``SplitResult`` validates on construction.
"""

from __future__ import annotations

import abc
import dataclasses
import inspect
import json
import re
from collections.abc import Iterator
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import sklearn.base

from chemsplit.determinism import (
    SeedBundle,
    argmax_tiebreak,
    argmin_tiebreak,
    floor_round,
    make_seed_bundle,
    stable_sort,
)
from chemsplit.exceptions import (
    ConfigurationError,
    InvariantError,
    ParameterError,
)
from chemsplit.types import IndexArray

__all__ = [
    "FAMILY_NAMES",
    "Strictness",
    "SplitResult",
    "BaseSplitter",
    "GroupSplitter",
    "check_group_splitter_design",
    "resolve_group_splitter",
    "resolve_sizes",
]

#: The nine family names splitter ids are grouped under.
FAMILY_NAMES: frozenset[str] = frozenset(
    {
        "baseline",
        "scaffold",
        "similarity",
        "embedding",
        "property",
        "lineage",
        "task",
        "biomolecular",
        "protocol",
    }
)

_SPLITTER_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class Strictness(str, __import__("enum").Enum):
    """Ordinal metadata describing expected train-to-test distance.

    Descriptive only; it never affects computation.
    """

    OPTIMISTIC = "optimistic"
    MODERATE = "moderate"
    STRICT = "strict"
    EXTRAPOLATIVE = "extrapolative"


def _encode_index_array(arr: IndexArray) -> list[int | list[int]]:
    """Run-length encode a sorted-ascending int64 index array: runs of >= 3 consecutive integers
    become a ``[start, end]`` pair (inclusive); shorter runs stay literal ints."""
    out: list[int | list[int]] = []
    values = arr.tolist()
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[j + 1] == values[j] + 1:
            j += 1
        run_len = j - i + 1
        if run_len >= 3:
            out.append([values[i], values[j]])
        else:
            out.extend(values[i: j + 1])
        i = j + 1
    return out


def _decode_index_array(encoded: list[int | list[int]]) -> IndexArray:
    values: list[int] = []
    for item in encoded:
        if isinstance(item, list):
            start, end = item
            values.extend(range(start, end + 1))
        else:
            values.append(item)
    return np.asarray(values, dtype=np.int64)


def _canonicalize_params(value: Any) -> Any:
    """Coerce a ``get_params()`` value tree into JSON-native shapes.

    Tuples become lists and numpy scalars native ``int``/``float``, so splitters with
    ``tuple[int, int]`` parameters need not avoid tuples by hand.

    ``SplitResult.params`` is an audit record, not a reconstruction mechanism --
    ``get_params()`` and ``clone()`` serve that and are untouched here -- so a value
    json.dumps cannot represent, such as a live
    :class:`~chemsplit.featurizers.Featurizer`, a callable ``embedding=`` or an array, is
    stringified. ``"<function...>"`` is still useful provenance.

    :param value: any node of a parameter value tree.
    :return: the same tree with every node replaced by a JSON-native equivalent.
    """
    if isinstance(value, (type(None), bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [_canonicalize_params(v) for v in value]
    if isinstance(value, list):
        return [_canonicalize_params(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonicalize_params(v) for k, v in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return f"<ndarray shape={value.shape} dtype={value.dtype}>"
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class SplitResult:
    """One ``(train, test)`` -- or ``(train, valid, test)`` -- outcome of a splitter, plus full
    provenance.

    Every partition is a sorted, strictly ascending int64 index array into the records passed to
    the splitter, and the four of them together cover ``range(n_records)`` exactly once.

    :param train: indices of the training records.
    :param test: indices of the test records.
    :param valid: indices of the validation records; empty when no validation set was asked for.
    :param discard: indices the splitter deliberately dropped, e.g. records straddling a
        boundary or belonging to a fold other than this one.
    :param groups: dense ``0..n_groups-1`` group label per record, in order of first appearance,
        or ``None`` for splitters that form no groups.
    :param splitter_id: snake_case id of the splitter that produced this result.
    :param params: the producing splitter's fully resolved ``get_params()``, canonicalized to
        JSON-native values.
    :param n_records: number of records the split was computed over.
    :param metadata: splitter-specific diagnostics, always including ``"realised_sizes"``.
    :raises InvariantError: if the partitions are malformed, miss or repeat a record, carry
        non-canonical group labels, hold params that do not round-trip through JSON, or name a
        non-snake_case ``splitter_id``.
    """

    train: IndexArray
    test: IndexArray
    valid: IndexArray
    discard: IndexArray
    groups: IndexArray | None
    splitter_id: str
    params: dict[str, Any]
    n_records: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        # Frozen dataclass, so this is the only hook every construction site passes through.
        # metadata is canonicalized too: only params is checked below, but to_json() writes
        # both, and a float32 there would fail at serialisation instead of here.
        object.__setattr__(self, "params", _canonicalize_params(self.params))
        object.__setattr__(self, "metadata", _canonicalize_params(self.metadata))
        self._check_index_arrays()
        self._check_partition_cover()
        self._check_group_labels()
        self._check_params_json()
        self._check_splitter_id()

    def _fail(self, message: str) -> None:
        """Raise :class:`InvariantError` with this result's provenance attached.

        :param message: what went wrong.
        :raises InvariantError: always.
        """
        raise InvariantError(
            message,
            splitter_id=self.splitter_id,
            params=self.params,
            n_records=self.n_records,
        )

    def _check_index_arrays(self) -> None:
        for name in ("train", "valid", "test", "discard"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray) or arr.dtype != np.int64 or arr.ndim != 1:
                self._fail(f"index arrays: {name!r} must be a 1-D int64 ndarray, got {arr!r}")
            if arr.size > 1 and not np.all(np.diff(arr) > 0):
                self._fail(f"index arrays: {name!r} must be strictly ascending, got {arr!r}")

    def _check_partition_cover(self) -> None:
        parts = [self.train, self.valid, self.test, self.discard]
        total = sum(p.size for p in parts)
        if total != self.n_records:
            self._fail(
                f"partition cover: train+valid+test+discard has {total} records, "
                f"expected n_records={self.n_records}"
            )
        union = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        if union.size and (
            np.unique(union).size != union.size
            or union.min() < 0
            or union.max() >= self.n_records
        ):
            self._fail("partition cover: train/valid/test/discard are not disjoint or not complete")

    def _check_group_labels(self) -> None:
        if self.groups is None:
            return
        g = self.groups
        if g.shape != (self.n_records,) or g.dtype != np.int64:
            self._fail(
                f"group labels: groups must have shape ({self.n_records},) and dtype int64, "
                f"got shape={g.shape} dtype={g.dtype}"
            )
        distinct = np.unique(g)
        if distinct.size and (distinct.min() < 0 or distinct.max() != distinct.size - 1):
            self._fail("group labels: groups must be 0..n_groups-1 with no gaps")
        first_seen: dict[int, int] = {}
        next_expected = 0
        for i, label in enumerate(g.tolist()):
            if label not in first_seen:
                if label != next_expected:
                    self._fail(
                        "group labels: group ids must be assigned in order of first appearance"
                    )
                first_seen[label] = i
                next_expected += 1

    def _check_params_json(self) -> None:
        try:
            round_tripped = json.loads(json.dumps(self.params))
        except (TypeError, ValueError) as exc:
            self._fail(f"params: not JSON-serialisable: {exc}")
            return
        if round_tripped != self.params:
            self._fail("params: does not round-trip through json.dumps/json.loads")

    def _check_splitter_id(self) -> None:
        if not _SPLITTER_ID_RE.match(self.splitter_id):
            self._fail(
                f"splitter_id: {self.splitter_id!r} does not match "
                f"{_SPLITTER_ID_RE.pattern!r}"
            )

    def as_tuple(self) -> tuple[IndexArray, IndexArray]:
        """Return the sklearn-shaped pair.

        :return: ``(train, test)``.
        """
        return self.train, self.test

    def as_triple(self) -> tuple[IndexArray, IndexArray, IndexArray]:
        """Return all three model-fitting partitions.

        :return: ``(train, valid, test)``.
        """
        return self.train, self.valid, self.test

    def to_frame(self) -> pd.DataFrame:
        """Flatten the split into one row per record.

        :return: a frame indexed 0..n-1 with columns ``index``, ``partition`` and ``group``
            (``None`` throughout when the splitter formed no groups).
        """
        rows: list[tuple[int, str, int | None]] = []
        groups = self.groups
        for partition_name in ("train", "valid", "test", "discard"):
            for idx in getattr(self, partition_name).tolist():
                rows.append((idx, partition_name, int(groups[idx]) if groups is not None else None))
        frame = pd.DataFrame(rows, columns=["index", "partition", "group"])
        return frame.sort_values("index").reset_index(drop=True)

    def to_json(self) -> str:
        """Serialise the result, with index arrays run-length encoded.

        :return: a compact, key-sorted JSON string tagged ``schema="chemsplit/split/1"``.
        """
        payload = {
            "schema": "chemsplit/split/1",
            "splitter_id": self.splitter_id,
            "n_records": self.n_records,
            "train": _encode_index_array(self.train),
            "valid": _encode_index_array(self.valid),
            "test": _encode_index_array(self.test),
            "discard": _encode_index_array(self.discard),
            "groups": self.groups.tolist() if self.groups is not None else None,
            "params": self.params,
            "metadata": self.metadata,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_json(cls, s: str) -> SplitResult:
        """Rebuild a result from :meth:`to_json` output.

        :param s: a JSON string produced by :meth:`to_json`.
        :raises InvariantError: if the payload does not satisfy the usual checks.
        :return: the reconstructed :class:`SplitResult`.
        """
        payload = json.loads(s)
        groups = payload["groups"]
        return cls(
            train=_decode_index_array(payload["train"]),
            valid=_decode_index_array(payload["valid"]),
            test=_decode_index_array(payload["test"]),
            discard=_decode_index_array(payload["discard"]),
            groups=np.asarray(groups, dtype=np.int64) if groups is not None else None,
            splitter_id=payload["splitter_id"],
            params=payload["params"],
            n_records=payload["n_records"],
            metadata=payload["metadata"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _ResolvedSizes:
    n_train: int
    n_valid: int
    n_test: int


_UNSET = object()


def _resolve_one(value: Any, n: int, field_name: str) -> Any:
    """Resolve one ``*_size`` value to ``_UNSET``, an int count, or a rounded fraction of ``n``.

    :param value: ``None``, an int count, or a float fraction in ``[0, 1]``.
    :param n: number of records.
    :param field_name: parameter name, used in error messages.
    :raises ParameterError: if ``value`` is a bool, an out-of-range number, or another type.
    :return: ``_UNSET`` for ``None``, otherwise a record count.
    """
    if value is None:
        return _UNSET
    if isinstance(value, bool):
        raise ParameterError(f"{field_name}: bool is not a valid size spec, got {value!r}")
    if isinstance(value, (int, np.integer)):
        if not (1 <= value <= n):
            raise ParameterError(
                f"{field_name}: int size spec must satisfy 1 <= value <= n ({n}), got {value!r}"
            )
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not (0.0 <= value <= 1.0):
            raise ParameterError(
                f"{field_name}: float size spec must satisfy 0.0 <= value <= 1.0, got {value!r}"
            )
        if value == 0.0:
            return 0
        return floor_round(value * n)
    raise ParameterError(f"{field_name}: invalid size spec {value!r} ({type(value).__name__})")


def resolve_sizes(
    n: int,
    train_size: Any,
    valid_size: Any,
    test_size: Any,
) -> _ResolvedSizes:
    """Resolve ``train_size``/``valid_size``/``test_size`` into concrete record counts.

    At most one of the three may be left unset, in which case it absorbs the remainder. An
    integer ``0`` is rejected; pass ``0.0`` or ``None`` for a deliberately empty partition.

    :param n: number of records.
    :param train_size: int count, float fraction, or ``None`` to infer.
    :param valid_size: int count, float fraction, or ``None`` to infer.
    :param test_size: int count, float fraction, or ``None`` to infer.
    :raises ParameterError: if the sizes are ambiguous, exceed ``n``, or resolve to an empty
        train or test partition.
    :return: the resolved ``n_train``/``n_valid``/``n_test`` counts.
    """
    resolved = {
        "train": _resolve_one(train_size, n, "train_size"),
        "valid": _resolve_one(valid_size, n, "valid_size"),
        "test": _resolve_one(test_size, n, "test_size"),
    }
    n_unset = sum(1 for v in resolved.values() if v is _UNSET)

    if n_unset == 3:
        resolved["train"] = floor_round(0.8 * n)
        resolved["test"] = floor_round(0.2 * n)
        resolved["valid"] = 0
    elif n_unset == 1:
        known_sum = sum(v for v in resolved.values() if v is not _UNSET)
        remainder = n - known_sum
        if remainder < 0:
            raise ParameterError(
                f"size spec exceeds n={n}: known partitions already sum to {known_sum}"
            )
        for key, value in resolved.items():
            if value is _UNSET:
                resolved[key] = remainder
    elif n_unset == 2:
        raise ParameterError(
            "ambiguous size spec: at most one of train_size/valid_size/test_size "
            "may be None"
        )
    # n_unset == 0: nothing to absorb yet.

    count_tr, count_va, count_te = resolved["train"], resolved["valid"], resolved["test"]
    if count_tr + count_va + count_te > n:
        raise ParameterError(
            f"train_size + valid_size + test_size ({count_tr + count_va + count_te}) exceeds "
            f"n ({n})"
        )
    leftover = n - (count_tr + count_va + count_te)
    if leftover > 0 and n_unset == 0:
        count_tr += leftover

    if count_tr < 1:
        raise ParameterError(f"resolved train size must be >= 1, got {count_tr}")
    if count_te < 1:
        raise ParameterError(f"resolved test size must be >= 1, got {count_te}")

    return _ResolvedSizes(n_train=count_tr, n_valid=count_va, n_test=count_te)


@dataclasses.dataclass(frozen=True, slots=True)
class _Context:
    n: int
    mols: list[Any] | None
    smiles: list[str] | None
    y: np.ndarray | None
    groups_in: IndexArray | None
    dates: np.ndarray | None
    targets: np.ndarray | None
    sequences: list[str] | None
    sizes: _ResolvedSizes
    rng_seeds: SeedBundle
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)
    raw_features: Any | None = None
    """When ``X`` was already a feature matrix (detected kind ``"features"``), the matrix itself:
    there is no ``Featurizer`` to run in that case, so :meth:`get_features` needs somewhere to
    return it from."""
    _feature_cache: dict[Any, Any] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )

    def get_features(self, featurizer: Any = None) -> Any:
        """Featurize the context's molecules, memoised for the lifetime of one ``_run`` call.

        Takes an already-constructed featurizer rather than an alias, so that this module need
        not import :mod:`chemsplit.featurizers`.

        :param featurizer: a ``Featurizer`` instance; optional, and ignored, when ``X`` was
            already a feature matrix, in which case :attr:`raw_features` is returned directly.
        :raises ValueError: if ``X`` was not a feature matrix and no featurizer was given.
        :return: the feature matrix.
        """
        if self.raw_features is not None:
            return self.raw_features
        if featurizer is None:
            raise ValueError("get_features() requires a featurizer when X was not already features")
        key = (
            id(self.mols),
            featurizer.name,
            tuple(sorted(featurizer.get_params().items())),
        )
        if key not in self._feature_cache:
            self._feature_cache[key] = featurizer.transform(self.mols)
        return self._feature_cache[key]


class BaseSplitter(sklearn.base.BaseEstimator, abc.ABC):
    """Base class for every splitter: parameter handling, preprocessing and the sklearn API.

    Subclasses declare their identity through the class attributes above and implement
    :meth:`_partition`. These constructor parameters are accepted by every splitter in the
    library and are forwarded up from each subclass's ``**kwargs``.

    :param n_splits: number of splits to yield. Splitters that produce a single holdout ignore
        anything but ``1``; CV splitters accept an int, and some also ``"auto"`` or ``"loo"``.
    :param train_size: training partition size, as an int count or a float fraction of ``n``.
        ``None`` means "whatever is left over".
    :param valid_size: validation partition size, same forms as ``train_size``.
    :param test_size: test partition size, same forms as ``train_size``.
    :param random_state: seed for every stochastic choice. An int reproduces, ``None`` draws
        OS entropy, and a ``numpy.random.Generator`` leaves the split unreproducible from
        ``params`` alone.
    :param n_jobs: worker count for parallelisable internals. Results do not depend on it.
    :param verbose: verbosity level; ``0`` is silent.
    :raises ParameterError: if any size spec, ``n_splits`` or ``n_jobs`` is invalid.
    """

    splitter_id: ClassVar[str]
    family: ClassVar[str]
    strictness: ClassVar[Strictness]
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = False
    requires_dates: ClassVar[bool] = False
    requires_targets: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        self.n_splits = n_splits
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._validate_base_params()

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Collect constructor parameter names across the whole MRO, not just ``cls.__init__``.

        Splitters forward ``**base`` upwards, and sklearn's ``_get_param_names`` introspects
        only ``cls.__init__``, so it drops every inherited parameter and breaks ``clone()``.
        Every class defining its own ``__init__`` is walked, so mixins such as
        ``SimilarityParamsMixin`` contribute their parameters too.

        :return: the sorted parameter names, excluding ``self``, ``*args`` and ``**kwargs``.
        """
        names: set[str] = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            for p in inspect.signature(init).parameters.values():
                if p.name == "self":
                    continue
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                names.add(p.name)
        return sorted(names)

    @staticmethod
    def _sanitize_params(raw: dict[str, Any]) -> dict[str, Any]:
        """Coerce a ``get_params()`` dict into the round-trip-safe form ``params`` needs.

        Thin wrapper around :func:`_canonicalize_params`. ``SplitResult.__post_init__`` applies
        the same thing unconditionally, so calling this first is optional.

        :param raw: a ``get_params()`` dict.
        :return: the same mapping with JSON-native values.
        """
        return {k: _canonicalize_params(v) for k, v in raw.items()}

    def _validate_base_params(self) -> None:
        """Check the parameters shared by every splitter.

        :raises ParameterError: if ``n_splits``, a size spec or ``n_jobs`` is invalid.
        """
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, (int, np.integer)):
            if self.n_splits not in ("auto", "loo"):  # subclasses may special-case these
                raise ParameterError(f"n_splits must be a positive int, got {self.n_splits!r}")
        elif self.n_splits < 1:
            raise ParameterError(f"n_splits must be >= 1, got {self.n_splits!r}")
        for name in ("train_size", "valid_size", "test_size"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool):
                raise ParameterError(f"{name}: bool is not a valid size spec, got {value!r}")
            if isinstance(value, (float, np.floating)) and not (0.0 <= value <= 1.0):
                raise ParameterError(f"{name} must satisfy 0.0 <= value <= 1.0, got {value!r}")
            if isinstance(value, (int, np.integer)) and value < 0:
                raise ParameterError(f"{name} must be >= 0, got {value!r}")
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, (int, np.integer)):
            raise ParameterError(f"n_jobs must be an int, got {self.n_jobs!r}")

    def split(
        self, X: Any, y: Any = None, groups: Any = None, **kw: Any
    ) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Yield ``(train, test)`` index pairs, as sklearn cross-validators do.

        :param X: the records: SMILES strings, RDKit molecules, a feature matrix, sequences or
            interaction tuples, depending on the splitter's ``accepts``.
        :param y: labels, required by splitters with ``requires_labels``.
        :param groups: precomputed group labels, for splitters that consume rather than form
            them.
        :param kw: per-call extras such as ``X_kind``, ``dates``, ``targets``, ``sequences``,
            ``standardize``, ``on_parse_error`` and ``on_duplicates``.
        :raises ConfigurationError: if ``valid_size`` is non-empty, since those records have
            nowhere to go in a two-way contract.
        :return: an iterator over ``(train_indices, test_indices)``.
        """
        if self._resolved_valid_size_nonzero():
            raise ConfigurationError(
                "split() cannot be used when valid_size is non-empty -- records assigned to "
                "'valid' would silently vanish from sklearn's (train, test) contract. Use "
                "split_with_validation() or split_result() instead."
            )
        for result in self._run(X, y, groups, **kw):
            yield result.train, result.test

    def split_with_validation(
        self, X: Any, y: Any = None, groups: Any = None, **kw: Any
    ) -> Iterator[tuple[IndexArray, IndexArray, IndexArray]]:
        """Yield ``(train, valid, test)`` index triples.

        :param X: the records, as for :meth:`split`.
        :param y: labels, if the splitter needs them.
        :param groups: precomputed group labels.
        :param kw: per-call extras, as for :meth:`split`.
        :return: an iterator over ``(train_indices, valid_indices, test_indices)``.
        """
        for result in self._run(X, y, groups, **kw):
            yield result.as_triple()

    def split_result(
        self, X: Any, y: Any = None, groups: Any = None, **kw: Any
    ) -> list[SplitResult]:
        """Split, keeping the discard pile, group labels, params and metadata.

        :param X: the records, as for :meth:`split`.
        :param y: labels, if the splitter needs them.
        :param groups: precomputed group labels.
        :param kw: per-call extras, as for :meth:`split`.
        :return: one :class:`SplitResult` per split, in fold order.
        """
        return self._run(X, y, groups, **kw)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Report how many splits will be yielded, for sklearn's cross-validator protocol.

        :param X: ignored, as are ``y`` and ``groups``; the signature is sklearn\'s.
        :return: ``n_splits`` when it is an int, else ``1``.
        """
        if isinstance(self.n_splits, (int, np.integer)) and not isinstance(self.n_splits, bool):
            return int(self.n_splits)
        return 1

    def compute_groups(self, X: Any, y: Any = None, **kw: Any) -> IndexArray:
        """Expose the group labels without performing a split.

        :param X: the records, as for :meth:`split`.
        :param y: labels, if the splitter needs them.
        :param kw: per-call extras, as for :meth:`split`.
        :raises NotImplementedError: on splitters that form no groups.
        :return: one dense group label per record.
        """
        if not self.group_forming:
            raise NotImplementedError(
                f"{type(self).__name__} is not group-forming; compute_groups() is only "
                "available on GroupSplitter subclasses"
            )
        raise NotImplementedError  # overridden by GroupSplitter

    def _resolved_valid_size_nonzero(self) -> bool:
        """Report whether ``valid_size`` asks for a non-empty validation partition.

        :return: ``True`` when records would be routed to ``valid``.
        """
        vs = self.valid_size
        if vs is None:
            return False
        if isinstance(vs, bool):
            return False
        return bool(vs)

    def _run(self, X: Any, y: Any, groups: Any, **kw: Any) -> list[SplitResult]:
        from chemsplit import preprocess  # local import: avoids a hard import-time coupling

        x_kind = preprocess.detect_input_kind(X, kw.get("X_kind"))
        if x_kind not in self.accepts:
            from chemsplit.exceptions import InputKindError

            raise InputKindError(
                f"{type(self).__name__} accepts {self.accepts}, but the detected kind of X is "
                f"{x_kind!r}"
            )

        n = preprocess.input_length(X, x_kind)
        if n < 2:
            from chemsplit.exceptions import EmptyInputError

            raise EmptyInputError(f"X has {n} record(s); chemsplit requires n >= 2")

        if self.requires_labels and y is None:
            from chemsplit.exceptions import LabelError

            raise LabelError(f"{type(self).__name__} requires labels (y) but y is None")
        if y is not None and len(y) != n:
            from chemsplit.exceptions import LabelError

            raise LabelError(f"len(y)={len(y)} does not match n={n}")

        sizes = resolve_sizes(n, self.train_size, self.valid_size, self.test_size)

        pipeline_result = preprocess.run_pipeline(
            X,
            y,
            groups,
            x_kind=x_kind,
            on_parse_error=kw.get("on_parse_error", "raise"),
            standardize=kw.get("standardize", False),
            on_duplicates=kw.get("on_duplicates", "warn"),
            group_forming=self.group_forming,
        )

        bundle = make_seed_bundle(self.random_state)
        ctx = _Context(
            n=n,
            mols=pipeline_result.mols,
            smiles=pipeline_result.smiles,
            y=y,
            groups_in=np.asarray(groups, dtype=np.int64) if groups is not None else None,
            dates=kw.get("dates"),
            targets=kw.get("targets"),
            sequences=kw.get("sequences") or (list(X) if x_kind == "sequences" else None),
            sizes=sizes,
            rng_seeds=bundle,
            extra={
                "resolved_seed": bundle.resolved_seed,
                "forced_discard": pipeline_result.forced_discard,
                "dedup_group_labels": pipeline_result.dedup_group_labels,
                # Interaction records name entities by key, not by molecule, so run_pipeline
                # leaves mols/smiles None for them and nothing else on _Context carries the
                # tuples. Stash them here.
                "raw_interactions": list(X) if x_kind == "interactions" else None,
            },
            raw_features=X if x_kind == "features" else None,
        )

        self._check_preconditions(ctx)
        results = self._partition(ctx)
        for result in results:
            if result.splitter_id != self.splitter_id:
                raise InvariantError(
                    "SplitResult.splitter_id does not match the splitter that produced it",
                    splitter_id=result.splitter_id,
                    params=result.params,
                    n_records=result.n_records,
                )
        return results

    def _check_preconditions(self, ctx: _Context) -> None:
        """Splitter-specific precondition hook; no-op by default.

        :param ctx: the prepared split context.
        :raises ChemSplitError: in subclasses, if the inputs cannot support this splitter.
        """

    @abc.abstractmethod
    def _partition(self, ctx: _Context) -> list[SplitResult]:
        """Produce the splits. The one method every splitter must implement.

        :param ctx: the prepared split context.
        :return: one :class:`SplitResult` per split.
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, slots=True)
class _Bucket:
    name: str
    capacity: int


def assign_groups(
    labels: IndexArray,
    sizes: _ResolvedSizes,
    mode: Literal["greedy_desc", "balanced", "random"],
    rng: np.random.Generator,
) -> dict[str, IndexArray]:
    """Assign whole groups to partitions. Shared by every group-forming splitter.

    Groups are visited largest-first, or in seeded random order, and each goes to the partition
    with the most room, so no group is ever split.

    :param labels: dense ``0..g-1`` label per record. Callers must drop their ``forced_discard``
        records beforehand, since this routine never discards.
    :param sizes: the target record counts.
    :param mode: ``"greedy_desc"`` fills the emptiest partition, ``"balanced"`` minimises
        relative overflow, ``"random"`` shuffles the visit order.
    :param rng: generator used by ``mode="random"``.
    :return: a mapping from partition name to member indices, omitting zero-capacity partitions.
    """
    n = len(labels)
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(int(labels[i]), []).append(i)
    for lst in members.values():
        lst.sort()

    groups_by_min_index = sorted(members.keys(), key=lambda g: members[g][0])

    if mode == "random":
        order = [int(g) for g in rng.permutation(np.asarray(groups_by_min_index, dtype=np.int64))]
    else:
        order = stable_sort(groups_by_min_index, key=lambda g: len(members[g]), desc=True)

    buckets = [
        _Bucket("train", sizes.n_train),
        _Bucket("valid", sizes.n_valid),
        _Bucket("test", sizes.n_test),
    ]
    buckets = [b for b in buckets if b.capacity > 0]
    filled: dict[str, list[int]] = {b.name: [] for b in buckets}
    counts: dict[str, int] = {b.name: 0 for b in buckets}

    for g in order:
        m = len(members[g])
        if mode in ("greedy_desc", "random"):
            cand = argmax_tiebreak(lambda b: b.capacity - counts[b.name], buckets)
        else:  # "balanced"
            cand = argmin_tiebreak(
                lambda b: max(0, counts[b.name] + m - b.capacity) / max(1, b.capacity),
                buckets,
            )
        filled[cand.name].extend(members[g])
        counts[cand.name] += m

    return {
        name: np.sort(np.asarray(idx, dtype=np.int64))
        for name, idx in filled.items()
    }


def assign_groups_kfold(
    labels: IndexArray,
    n_splits: int,
    rng: np.random.Generator,
) -> list[IndexArray]:
    """Distribute whole groups over ``n_splits`` folds of near-equal size.

    Deficit-first, with the same tie rules as :func:`assign_groups`.

    :param labels: dense ``0..g-1`` label per record.
    :param n_splits: number of folds.
    :param rng: unused; the fold order is deterministic.
    :return: one array of member indices per fold.
    """
    n = len(labels)
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(int(labels[i]), []).append(i)
    for lst in members.values():
        lst.sort()
    groups_by_min_index = sorted(members.keys(), key=lambda g: members[g][0])
    order = stable_sort(groups_by_min_index, key=lambda g: len(members[g]), desc=True)

    target = n / n_splits
    counts = [0] * n_splits
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for g in order:
        m = len(members[g])
        fold_idx = argmax_tiebreak(lambda f: target - counts[f], range(n_splits))
        folds[fold_idx].extend(members[g])
        counts[fold_idx] += m
    return [np.sort(np.asarray(f, dtype=np.int64)) for f in folds]


class GroupSplitter(BaseSplitter):
    """Base class for splitters that form groups and assign them whole to partitions.

    Subclasses implement :meth:`_group_labels` and inherit the rest: the group-to-partition
    assignment, the k-fold-over-groups path, metadata and the size-tolerance check.

    :param size_tolerance: drift from a target size, as a fraction of ``n``, that triggers
        :class:`SizeToleranceWarning`. Whole-group assignment cannot hit an arbitrary ratio, so
        some drift is normal.
    :param group_assignment: how groups are handed to partitions; see :func:`assign_groups`.
    :param kw: forwarded to :class:`BaseSplitter`.
    :raises ParameterError: if ``size_tolerance`` is negative or ``group_assignment`` is
        unknown.
    """

    group_forming: ClassVar[bool] = True

    def __init__(
        self,
        *,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.size_tolerance = size_tolerance
        self.group_assignment = group_assignment
        if not (0.0 <= size_tolerance):
            raise ParameterError(f"size_tolerance must be >= 0, got {size_tolerance!r}")
        if group_assignment not in ("greedy_desc", "balanced", "random"):
            raise ParameterError(f"invalid group_assignment: {group_assignment!r}")

    @abc.abstractmethod
    def _group_labels(self, ctx: _Context) -> IndexArray:
        """Compute the grouping. The one method a group splitter must implement.

        :param ctx: the prepared split context.
        :return: a dense ``0..g-1`` label per record, numbered by first appearance.
        """
        raise NotImplementedError

    def compute_groups(self, X: Any, y: Any = None, **kw: Any) -> IndexArray:
        """Compute the grouping without performing a split.

        :param X: the records, as for :meth:`split`.
        :param y: labels, if the splitter needs them.
        :param kw: per-call extras, as for :meth:`split`.
        :return: one dense group label per record, numbered by first appearance.
        """
        from chemsplit import preprocess

        x_kind = preprocess.detect_input_kind(X, kw.get("X_kind"))
        n = preprocess.input_length(X, x_kind)
        sizes = resolve_sizes(n, self.train_size, self.valid_size, self.test_size)
        pipeline_result = preprocess.run_pipeline(
            X,
            y,
            None,
            x_kind=x_kind,
            on_parse_error=kw.get("on_parse_error", "raise"),
            standardize=kw.get("standardize", False),
            on_duplicates=kw.get("on_duplicates", "ignore"),
            group_forming=True,
        )
        bundle = make_seed_bundle(self.random_state)
        ctx = _Context(
            n=n,
            mols=pipeline_result.mols,
            smiles=pipeline_result.smiles,
            y=y,
            groups_in=None,
            dates=kw.get("dates"),
            targets=kw.get("targets"),
            sequences=kw.get("sequences") or (list(X) if x_kind == "sequences" else None),
            sizes=sizes,
            rng_seeds=bundle,
            raw_features=X if x_kind == "features" else None,
        )
        return self._group_labels(ctx)

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        """Splitter-specific metadata, e.g. ``n_clusters`` or algorithm diagnostics.

        Called once per :meth:`_partition`, before any record is discarded. The keys
        ``"fold_index"``, ``"n_splits"`` and ``"realised_sizes"`` belong to
        :meth:`_build_result` and the k-fold path.

        :param ctx: the prepared split context.
        :param labels: the full, pre-discard group labels.
        :return: entries to merge into ``SplitResult.metadata``; empty by default.
        """
        return {}

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        labels_full = self._group_labels(ctx)
        group_metadata = self._group_metadata(ctx, labels_full)
        forced_discard = np.asarray(
            sorted(ctx.extra.get("forced_discard", [])), dtype=np.int64
        )
        keep_mask = np.ones(ctx.n, dtype=bool)
        keep_mask[forced_discard] = False
        keep_idx = np.nonzero(keep_mask)[0]

        from chemsplit._unionfind import dense_label_encode

        labels_kept = dense_label_encode(labels_full[keep_idx].tolist())

        n_splits = self.get_n_splits()
        if n_splits > 1:
            if ctx.sizes.n_valid > 0:
                raise ConfigurationError(
                    "valid_size is not permitted with n_splits > 1 unless the splitter is "
                    "wrapped in the nested-CV protocol wrapper"
                )
            rng = self._rng_for_group_assignment(ctx)
            folds_local = assign_groups_kfold(labels_kept, n_splits, rng)
            results = []
            for k in range(n_splits):
                test_local = folds_local[k]
                train_local = np.sort(
                    np.concatenate([folds_local[j] for j in range(n_splits) if j != k])
                    if n_splits > 1
                    else np.array([], dtype=np.int64)
                )
                train = keep_idx[train_local]
                test = keep_idx[test_local]
                results.append(
                    self._build_result(
                        ctx,
                        train=train,
                        valid=np.array([], dtype=np.int64),
                        test=test,
                        discard=forced_discard,
                        groups=labels_full,
                        extra_metadata={
                            **group_metadata,
                            "fold_index": k,
                            "n_splits": n_splits,
                        },
                    )
                )
            return results

        rng = self._rng_for_group_assignment(ctx)
        buckets_local = assign_groups(labels_kept, ctx.sizes, self.group_assignment, rng)
        train = keep_idx[buckets_local["train"]]
        valid = keep_idx[buckets_local.get("valid", np.array([], dtype=np.int64))]
        test = keep_idx[buckets_local["test"]]
        result = self._build_result(
            ctx,
            train=train,
            valid=valid,
            test=test,
            discard=forced_discard,
            groups=labels_full,
            extra_metadata=group_metadata,
        )
        self._check_size_tolerance(result, ctx)
        return [result]

    def _rng_for_group_assignment(self, ctx: _Context) -> np.random.Generator:
        """Draw the generator used for group-to-partition assignment.

        :param ctx: the prepared split context.
        :return: a generator from the ``"group.assign"`` stream.
        """
        from chemsplit.determinism import seed_for

        return seed_for(ctx.rng_seeds, "group.assign", 0)

    def _build_result(
        self,
        ctx: _Context,
        *,
        train: IndexArray,
        valid: IndexArray,
        test: IndexArray,
        discard: IndexArray,
        groups: IndexArray,
        extra_metadata: dict[str, Any],
    ) -> SplitResult:
        """Assemble a :class:`SplitResult` with realised sizes recorded.

        :param ctx: the prepared split context.
        :param train: training indices.
        :param valid: validation indices.
        :param test: test indices.
        :param discard: discarded indices.
        :param groups: the group label per record.
        :param extra_metadata: splitter-specific metadata to merge in.
        :return: the validated result.
        """
        metadata = {
            "realised_sizes": {
                "train": int(train.size),
                "valid": int(valid.size),
                "test": int(test.size),
            },
            **extra_metadata,
        }
        return SplitResult(
            train=np.sort(train),
            valid=np.sort(valid),
            test=np.sort(test),
            discard=np.sort(discard),
            groups=groups,
            splitter_id=self.splitter_id,
            params=_canonicalize_params(self.get_params()),
            n_records=ctx.n,
            metadata=metadata,
        )

    def _check_size_tolerance(self, result: SplitResult, ctx: _Context) -> None:
        """Warn when a realised partition size drifts past ``size_tolerance``.

        :param result: the result to check.
        :param ctx: the prepared split context, holding the targets.
        """
        from chemsplit.exceptions import SizeToleranceWarning, warn_with_details

        targets = {
            "train": ctx.sizes.n_train,
            "valid": ctx.sizes.n_valid,
            "test": ctx.sizes.n_test,
        }
        realised = result.metadata["realised_sizes"]
        for name, target in targets.items():
            deviation = abs(realised[name] - target) / max(1, ctx.n)
            if deviation > self.size_tolerance:
                warn_with_details(
                    SizeToleranceWarning(
                        f"{type(self).__name__}: realised {name} size {realised[name]} deviates "
                        f"from target {target} by {deviation:.3f} of n (tolerance "
                        f"{self.size_tolerance})",
                        details={
                            "splitter_id": self.splitter_id,
                            "partition": name,
                            "target": target,
                            "realised": realised[name],
                            "deviation_frac": deviation,
                        },
                    )
                )


def check_group_splitter_design(
    design: Any,
    param_name: str,
    *,
    owner: str,
    design_kwargs: Any = None,
    design_kwargs_name: str = "",
) -> None:
    """Validate a ``clusterer=``-style parameter without touching the registry.

    Called from ``__init__``, so a wrong type fails at construction. Whether a registry id
    exists, and is group-forming, is only settled when :func:`resolve_group_splitter` resolves
    it at split time, since resolving needs :mod:`chemsplit.registry`, which imports every
    splitter module in turn.

    :param design: the value passed for ``param_name``.
    :param param_name: the parameter's name, for error messages.
    :param owner: the owning class name, for error messages.
    :param design_kwargs: kwargs intended for a string-resolved splitter, if any.
    :param design_kwargs_name: that parameter's name, for error messages.
    :raises ParameterError: if ``design`` is neither ``None``, a string, nor a
        :class:`GroupSplitter`.
    :raises ConfigurationError: if kwargs are supplied alongside an already-built instance.
    """
    if design is not None and not isinstance(design, (str, GroupSplitter)):
        raise ParameterError(
            f"{owner}: {param_name} must be None, a registry id/class-name string "
            f"(e.g. 'butina'), or a group-forming GroupSplitter instance; got "
            f"{type(design).__name__}"
        )
    if design_kwargs is not None and not isinstance(design, str):
        raise ConfigurationError(
            f"{owner}: {design_kwargs_name} only applies when {param_name} is a string to "
            f"instantiate; pass an already-configured splitter instead"
        )


def resolve_group_splitter(
    design: str | GroupSplitter | None,
    param_name: str,
    *,
    owner: str,
    design_kwargs: dict[str, Any] | None = None,
    random_state: int | None = None,
) -> GroupSplitter | None:
    """Resolve ``design`` to a group-forming splitter, or ``None`` for "use my own default".

    An instance passes through unchanged; a string goes through
    :func:`chemsplit.registry.get_splitter`. Resolving at split time rather than in ``__init__``
    keeps ``get_params()`` reporting the string the caller passed, and keeps splitter modules
    from importing the registry.

    :param design: ``None``, a registry id, or a :class:`GroupSplitter` instance.
    :param param_name: the parameter's name, for error messages.
    :param owner: the owning class name, for error messages.
    :param design_kwargs: kwargs for a string-resolved splitter.
    :param random_state: seed for a string-resolved splitter, needed because ``get_splitter``
        otherwise draws OS entropy. Ignored for an instance, or if ``design_kwargs`` seeds it.
    :raises ParameterError: if ``design`` resolves to something that is not group-forming.
    :return: the resolved splitter, or ``None``.
    """
    if design is None:
        return None
    if isinstance(design, str):
        from chemsplit.registry import get_splitter  # local: avoids a module-scope import cycle

        kwargs = dict(design_kwargs or {})
        if random_state is not None:
            kwargs.setdefault("random_state", random_state)
        resolved = get_splitter(design, **kwargs)
    else:
        resolved = design
    if not isinstance(resolved, GroupSplitter) or not resolved.group_forming:
        raise ParameterError(
            f"{owner}: {param_name}={design!r} resolves to {type(resolved).__name__}, which is "
            f"not group-forming and so cannot supply cluster labels"
        )
    return resolved
