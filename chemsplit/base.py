"""Core types and the splitter API: ``SplitResult``, ``BaseSplitter``, ``GroupSplitter``, plus the
canonical group-to-partition routine (``assign_groups``) and its k-fold-over-groups variant.

Splitter ids are the snake_case form of the class name (e.g. ``"butina"``), validated by
invariant I5.
"""

from __future__ import annotations

import abc
import dataclasses
import json
import re
from collections.abc import Iterator
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import sklearn.base

from chemsplit.determinism import (
    EPS,
    SeedBundle,
    argmax_tiebreak,
    argmin_tiebreak,
    floor_round,
    make_seed_bundle,
    stable_sort,
)
from chemsplit.exceptions import (
    ConfigurationError,
    EmptyPartitionError,
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
    "resolve_sizes",
]

#: The fixed 9-name family enum used for splitter ids.
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
    """Ordinal metadata describing expected train-to-test distance. Descriptive only — it
    never affects computation."""

    OPTIMISTIC = "optimistic"
    MODERATE = "moderate"
    STRICT = "strict"
    EXTRAPOLATIVE = "extrapolative"


# ---------------------------------------------------------------------------
# Run-length index encoding
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SplitResult
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SplitResult:
    """One ``(train, test)`` — or ``(train, valid, test)`` — outcome of a splitter, plus full
    provenance."""

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
        self._check_i1()
        self._check_i2()
        self._check_i3()
        self._check_i4()
        self._check_i5()

    # -- invariants ---------------------------------------------------------

    def _fail(self, message: str) -> None:
        raise InvariantError(
            message,
            splitter_id=self.splitter_id,
            params=self.params,
            n_records=self.n_records,
        )

    def _check_i1(self) -> None:
        for name in ("train", "valid", "test", "discard"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray) or arr.dtype != np.int64 or arr.ndim != 1:
                self._fail(f"I1 violated: {name!r} must be a 1-D int64 ndarray, got {arr!r}")
            if arr.size > 1 and not np.all(np.diff(arr) > 0):
                self._fail(f"I1 violated: {name!r} must be strictly ascending, got {arr!r}")

    def _check_i2(self) -> None:
        parts = [self.train, self.valid, self.test, self.discard]
        total = sum(p.size for p in parts)
        if total != self.n_records:
            self._fail(
                f"I2 violated: train+valid+test+discard has {total} records, "
                f"expected n_records={self.n_records}"
            )
        union = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        if union.size and (
            np.unique(union).size != union.size
            or union.min() < 0
            or union.max() >= self.n_records
        ):
            self._fail("I2 violated: train/valid/test/discard are not disjoint or not complete")

    def _check_i3(self) -> None:
        if self.groups is None:
            return
        g = self.groups
        if g.shape != (self.n_records,) or g.dtype != np.int64:
            self._fail(
                f"I3 violated: groups must have shape ({self.n_records},) and dtype int64, "
                f"got shape={g.shape} dtype={g.dtype}"
            )
        distinct = np.unique(g)
        if distinct.size and (distinct.min() < 0 or distinct.max() != distinct.size - 1):
            self._fail("I3 violated: groups must be 0..n_groups-1 with no gaps")
        first_seen: dict[int, int] = {}
        next_expected = 0
        for i, label in enumerate(g.tolist()):
            if label not in first_seen:
                if label != next_expected:
                    self._fail(
                        "I3 violated: group ids must be assigned in order of first appearance"
                    )
                first_seen[label] = i
                next_expected += 1

    def _check_i4(self) -> None:
        try:
            round_tripped = json.loads(json.dumps(self.params))
        except (TypeError, ValueError) as exc:
            self._fail(f"I4 violated: params is not JSON-serialisable: {exc}")
            return
        if round_tripped != self.params:
            self._fail("I4 violated: params does not round-trip through json.dumps/json.loads")

    def _check_i5(self) -> None:
        if not _SPLITTER_ID_RE.match(self.splitter_id):
            self._fail(
                f"I5 violated: splitter_id {self.splitter_id!r} does not match "
                f"{_SPLITTER_ID_RE.pattern!r}"
            )

    # -- accessors ------------------------------------------------------------

    def as_tuple(self) -> tuple[IndexArray, IndexArray]:
        return self.train, self.test

    def as_triple(self) -> tuple[IndexArray, IndexArray, IndexArray]:
        return self.train, self.valid, self.test

    def to_frame(self) -> pd.DataFrame:
        rows: list[tuple[int, str, int | None]] = []
        groups = self.groups
        for partition_name in ("train", "valid", "test", "discard"):
            for idx in getattr(self, partition_name).tolist():
                rows.append((idx, partition_name, int(groups[idx]) if groups is not None else None))
        frame = pd.DataFrame(rows, columns=["index", "partition", "group"])
        return frame.sort_values("index").reset_index(drop=True)

    def to_json(self) -> str:
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
    def from_json(cls, s: str) -> "SplitResult":
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


# ---------------------------------------------------------------------------
# resolve_sizes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _ResolvedSizes:
    n_train: int
    n_valid: int
    n_test: int


_UNSET = object()


def _resolve_one(value: Any, n: int, field_name: str) -> Any:
    """Resolve a single ``*_size`` value to ``_UNSET``, an int count, or a rounded fraction of ``n``."""
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

    Note: an integer size of ``0`` is rejected (the int branch requires ``1 <= value <= n``).
    Callers wanting an explicitly-empty partition should pass ``0.0`` or ``None``, not the int
    ``0``.
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


# ---------------------------------------------------------------------------
# _Context
# ---------------------------------------------------------------------------


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
        """Memoised featurization: cached on
        ``(id(mols), featurizer.name, sorted(featurizer.get_params().items()))`` within one
        ``_run`` call. Takes an already-constructed ``Featurizer`` instance — this module does not
        import ``chemsplit.featurizers`` itself, keeping core infra decoupled from it. If ``X`` was
        already a feature matrix, ``featurizer`` may be omitted (or is ignored) and
        :attr:`raw_features` is returned directly.
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


# ---------------------------------------------------------------------------
# BaseSplitter
# ---------------------------------------------------------------------------


class BaseSplitter(sklearn.base.BaseEstimator, abc.ABC):
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

    def _validate_base_params(self) -> None:
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, (int, np.integer)):
            if self.n_splits != "auto":  # subclasses may special-case "auto"/"loo"
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

    # -- public API -----------------------------------------------------------

    def split(
        self, X: Any, y: Any = None, groups: Any = None, **kw: Any
    ) -> Iterator[tuple[IndexArray, IndexArray]]:
        if self._resolved_valid_size_nonzero():
            raise ConfigurationError(
                "split() cannot be used when valid_size is non-empty — records assigned to "
                "'valid' would silently vanish from sklearn's (train, test) contract. Use "
                "split_with_validation() or split_result() instead."
            )
        for result in self._run(X, y, groups, **kw):
            yield result.train, result.test

    def split_with_validation(
        self, X: Any, y: Any = None, groups: Any = None, **kw: Any
    ) -> Iterator[tuple[IndexArray, IndexArray, IndexArray]]:
        for result in self._run(X, y, groups, **kw):
            yield result.as_triple()

    def split_result(self, X: Any, y: Any = None, groups: Any = None, **kw: Any) -> list[SplitResult]:
        return self._run(X, y, groups, **kw)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        if isinstance(self.n_splits, (int, np.integer)) and not isinstance(self.n_splits, bool):
            return int(self.n_splits)
        return 1

    def compute_groups(self, X: Any, y: Any = None, **kw: Any) -> IndexArray:
        if not self.group_forming:
            raise NotImplementedError(
                f"{type(self).__name__} is not group-forming; compute_groups() is only "
                "available on GroupSplitter subclasses"
            )
        raise NotImplementedError  # overridden by GroupSplitter

    def _resolved_valid_size_nonzero(self) -> bool:
        vs = self.valid_size
        if vs is None:
            return False
        if isinstance(vs, bool):
            return False
        return bool(vs)

    # -- template method --------------------------------------------------

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
            sequences=kw.get("sequences"),
            sizes=sizes,
            rng_seeds=bundle,
            extra={
                "resolved_seed": bundle.resolved_seed,
                "forced_discard": pipeline_result.forced_discard,
                "dedup_group_labels": pipeline_result.dedup_group_labels,
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
        """Splitter-specific precondition hook. No-op by default."""

    @abc.abstractmethod
    def _partition(self, ctx: _Context) -> list[SplitResult]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# GroupSplitter and the canonical assign_groups routine
# ---------------------------------------------------------------------------


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
    """The canonical group -> partition assignment routine, implemented once and shared
    by every group-forming splitter.

    ``labels`` is a dense ``0..g-1`` int64 array (one label per record still under consideration —
    callers must remove any ``forced_discard`` records from ``labels``/their own bookkeeping
    *before* calling this function; this routine never discards).
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
    """distribute groups over ``n_splits`` buckets of near-equal size (deficit-first, same
    tie rules as :func:`assign_groups`). Returns one IndexArray of member indices per fold."""
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
        raise NotImplementedError

    def compute_groups(self, X: Any, y: Any = None, **kw: Any) -> IndexArray:
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
            dates=None,
            targets=None,
            sequences=None,
            sizes=sizes,
            rng_seeds=bundle,
            raw_features=X if x_kind == "features" else None,
        )
        return self._group_labels(ctx)

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        labels_full = self._group_labels(ctx)
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
                        extra_metadata={"fold_index": k, "n_splits": n_splits},
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
            extra_metadata={},
        )
        self._check_size_tolerance(result, ctx)
        return [result]

    def _rng_for_group_assignment(self, ctx: _Context) -> np.random.Generator:
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
            params=self.get_params(),
            n_records=ctx.n,
            metadata=metadata,
        )

    def _check_size_tolerance(self, result: SplitResult, ctx: _Context) -> None:
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
