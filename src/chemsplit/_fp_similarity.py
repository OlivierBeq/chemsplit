"""Shared fingerprint/metric machinery for the ``similarity`` splitter family and ``hi``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chemsplit.exceptions import ParameterError, ScalabilityError
from chemsplit.featurizers import Featurizer, get_featurizer
from chemsplit.metrics import MetricName, is_bounded_metric, pairwise_distances

__all__ = [
    "EPS",
    "SimilarityParamsMixin",
    "compute_distance_matrix",
    "compute_neighbor_lists",
    "distance_range",
    "compute_similarity_matrix",
    "dense_matrix_fits",
    "guard_memory",
    "rectangular_distances",
    "resolve_featurizer",
]

#: Float-comparison tolerance for float32 similarity/distance matrices (~1.2e-7 rounding noise).
EPS = 1e-6


def resolve_featurizer(spec_or_instance: str | Featurizer, **kw: Any) -> Featurizer:
    """Resolve a featurizer alias or pass an instance through.

    A thin wrapper over :func:`chemsplit.featurizers.get_featurizer`, so callers that import
    only this module need no direct dependency on :mod:`chemsplit.featurizers`.

    :param spec_or_instance: an alias, or an already-built featurizer.
    :param kw: forwarded to the featurizer's constructor when an alias was given.
    :raises UnknownFeaturizerError: if the alias is not recognised.
    :return: the featurizer.
    """
    return get_featurizer(spec_or_instance, **kw)


_BYTES_PER_PAIR_IN_FLIGHT = 12
"""Peak bytes per entry for one :func:`~chemsplit.metrics.pairwise_distances` call: the float64
accumulator plus the float32 copy it returns, both live at the downcast."""


def guard_memory(
    n: int,
    max_memory_bytes: int,
    splitter_name: str,
    *,
    alternatives: list[str] | None = None,
    copies: int = 1,
) -> None:
    """Refuse to build an ``n x n`` matrix that would not fit.

    Every splitter that materialises a full dense matrix calls this first.

    :param n: the record count.
    :param max_memory_bytes: the ceiling.
    :param splitter_name: the caller, for the error message.
    :param alternatives: splitters to suggest instead, or ``None`` for the defaults.
    :param copies: how many ``n x n`` float64 matrices the caller holds live at once. The default
        of ``1`` already covers the two that :func:`~chemsplit.metrics.pairwise_distances` itself
        needs; pass more where the caller keeps additional ones, as ``SPXYSplitter`` does.
    :raises ScalabilityError: if the estimated peak would exceed the ceiling. The
        :class:`~chemsplit.exceptions.ScalabilityError` names the splitter, ``n``, the bytes
        needed and a few alternatives.
    """
    # The old n*n*4 estimate under-counted by ~3x, so a process could be OOM-killed well below the
    # nominal ceiling instead of raising a diagnosable ScalabilityError.
    required = n * n * (_BYTES_PER_PAIR_IN_FLIGHT + 8 * (copies - 1))
    if required <= max_memory_bytes:
        return
    if alternatives is None:
        alternatives = [
            "use a sparse/graph-based variant of this algorithm if one is offered "
            "(e.g. algorithm='sparse')",
            "use a mini-batch or streaming clustering algorithm instead of a full "
            "pairwise matrix",
            "subsample the input before splitting",
        ]
    alt_text = "; ".join(f"({chr(97 + i)}) {a}" for i, a in enumerate(alternatives))
    raise ScalabilityError(
        f"{splitter_name}: a dense {n}x{n} matrix would peak at {required:,} bytes, "
        f"exceeding max_memory_bytes={max_memory_bytes:,}. Alternatives: {alt_text}."
    )


class SimilarityParamsMixin:
    """Shared constructor parameters for fingerprint/metric-based splitters.

    Mix in alongside :class:`~chemsplit.base.BaseSplitter` or
    :class:`~chemsplit.base.GroupSplitter`; call ``self._validate_similarity_params()`` after
    both ``__init__``s have run. Set ``bounded_metric_required = True`` when the splitter treats
    ``cutoff``/``threshold`` as a similarity, so an unbounded metric (``"euclidean"``,
    ``"manhattan"``) raises :class:`~chemsplit.exceptions.ParameterError`.
    """

    bounded_metric_required: bool = False

    def __init__(
        self,
        *,
        featurizer: str | Featurizer = "ecfp4",
        metric: MetricName = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
    ) -> None:
        self.featurizer = featurizer
        self.metric = metric
        self.max_memory_bytes = max_memory_bytes
        self.n_jobs = n_jobs

    def _validate_similarity_params(self, *, bounded_metric_required: bool | None = None) -> None:
        """Check the shared ``featurizer``/``metric``/``max_memory_bytes`` block.

        :param bounded_metric_required: override the class's own ``bounded_metric_required``, or
            ``None`` to use it.
        :raises ParameterError: if ``max_memory_bytes`` is not a positive int, or the metric is
            unknown, or it is unbounded where a bounded one is required.
        """
        if bounded_metric_required is None:
            required = self.bounded_metric_required
        else:
            required = bounded_metric_required
        valid_metrics = {
            "tanimoto", "dice", "cosine", "euclidean", "manhattan", "tanimoto_count", "mahalanobis"
        }
        if self.metric not in valid_metrics:
            raise ParameterError(
                f"unknown metric {self.metric!r}; expected one of {sorted(valid_metrics)}"
            )
        if required and not is_bounded_metric(self.metric):
            raise ParameterError(
                f"metric={self.metric!r} is unbounded, but this splitter's cutoff/threshold is a "
                "similarity and requires a metric bounded in [0, 1] "
                "(tanimoto, dice, cosine, or tanimoto_count)."
            )
        if isinstance(self.max_memory_bytes, bool) or not isinstance(
            self.max_memory_bytes, (int, np.integer)
        ):
            raise ParameterError(
                f"max_memory_bytes must be an int, got {self.max_memory_bytes!r}"
            )
        if self.max_memory_bytes <= 0:
            raise ParameterError(f"max_memory_bytes must be > 0, got {self.max_memory_bytes!r}")


def _pairwise(
    ctx: Any,
    featurizer: str | Featurizer,
    metric: MetricName,
    max_memory_bytes: int,
    splitter_name: str,
    n_jobs: int,
) -> np.ndarray:
    guard_memory(ctx.n, max_memory_bytes, splitter_name)
    feat = resolve_featurizer(featurizer)
    F = ctx.get_features(feat)
    return pairwise_distances(F, metric=metric, n_jobs=n_jobs)


def dense_matrix_fits(n: int, max_memory_bytes: int, copies: int = 1) -> bool:
    """Report whether the dense path is within budget, without raising.

    Lets a splitter take the dense route when affordable -- faster, since the matrix is reused --
    and an equivalent blocked route otherwise, rather than failing outright. Both return identical
    values.

    :param n: the record count.
    :param max_memory_bytes: the ceiling.
    :param copies: live ``n x n`` matrices, as for :func:`guard_memory`.
    :return: ``True`` if the dense matrix fits.
    """
    return n * n * (_BYTES_PER_PAIR_IN_FLIGHT + 8 * (copies - 1)) <= max_memory_bytes


def rectangular_distances(
    ctx: Any,
    featurizer: str | Featurizer,
    metric: MetricName,
    rows: Any,
    cols: Any,
    n_jobs: int = 1,
) -> np.ndarray:
    """Distances between two index subsets, without building the full matrix.

    :param ctx: the split context supplying the records.
    :param featurizer: an alias or a featurizer instance.
    :param metric: the distance metric.
    :param rows: left-hand record indices.
    :param cols: right-hand record indices.
    :param n_jobs: worker count. Results never depend on it.
    :return: a ``(len(rows), len(cols))`` distance block.
    """
    feat = resolve_featurizer(featurizer)
    F = ctx.get_features(feat)
    return pairwise_distances(
        F[np.asarray(rows)], F[np.asarray(cols)], metric=metric, n_jobs=n_jobs
    )


def distance_range(
    ctx: Any,
    featurizer: str | Featurizer,
    metric: MetricName,
    *,
    n_jobs: int = 1,
    block_rows: int = 2048,
) -> tuple[float, float]:
    """Off-diagonal minimum and maximum distance, computed blockwise.

    Mirrors the dense ``D.max()`` and diagonal-masked ``D.min()``, so a
    ``radius_is="fraction_of_range"`` threshold is identical without an ``n x n`` matrix.

    :param ctx: the split context supplying the records.
    :param featurizer: an alias or a featurizer instance.
    :param metric: the distance metric.
    :param n_jobs: worker count. Results never depend on it.
    :param block_rows: rows per block.
    :return: ``(min, max)`` over the off-diagonal entries.
    """
    feat = resolve_featurizer(featurizer)
    F = ctx.get_features(feat)
    n = F.shape[0]
    d_min, d_max = float("inf"), float("-inf")
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        block = pairwise_distances(F[start:stop], F, metric=metric, n_jobs=n_jobs)
        d_max = max(d_max, float(block.max()))
        # hide this block's slice of the diagonal before taking the minimum
        rows = np.arange(stop - start)
        block[rows, rows + start] = np.inf
        d_min = min(d_min, float(block.min()))
    return d_min, d_max


def compute_neighbor_lists(
    ctx: Any,
    featurizer: str | Featurizer,
    metric: MetricName,
    cutoff: float,
    *,
    eps: float,
    n_jobs: int = 1,
    block_rows: int = 2048,
) -> list[np.ndarray]:
    """Radius-neighbour lists, built blockwise so no ``n x n`` matrix is materialised.

    Peak memory is ``block_rows * n``, not ``n * n``: at n=100000 the dense float64 matrix is 80 GB
    against 1.6 GB per block.

    Bit-identical to slicing the dense matrix -- the kernel is integer-exact, so blocking cannot
    shift a value across the cutoff, and the same comparison and ordering are applied.

    :param ctx: the split context supplying the records.
    :param featurizer: an alias or a featurizer instance.
    :param metric: the distance metric.
    :param cutoff: the neighbour radius, as a distance.
    :param eps: the caller's float-comparison tolerance, added to ``cutoff`` exactly as the dense
        path does.
    :param n_jobs: worker count. Results never depend on it.
    :param block_rows: rows per block.
    :return: per record, the ascending indices within the cutoff, excluding itself.
    """
    feat = resolve_featurizer(featurizer)
    F = ctx.get_features(feat)
    n = F.shape[0]
    out: list[np.ndarray] = []
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        block = pairwise_distances(F[start:stop], F, metric=metric, n_jobs=n_jobs)
        for row_offset in range(stop - start):
            i = start + row_offset
            idx = np.nonzero(block[row_offset] <= cutoff + eps)[0]
            out.append(np.sort(idx[idx != i]))
    return out


def compute_distance_matrix(
    ctx: Any,
    featurizer: str | Featurizer,
    metric: MetricName,
    max_memory_bytes: int,
    splitter_name: str,
    n_jobs: int = 1,
) -> np.ndarray:
    """Build the full pairwise distance matrix for a context's records.

    :param ctx: the split context supplying the records.
    :param featurizer: an alias or a featurizer instance.
    :param metric: the distance metric.
    :param max_memory_bytes: ceiling enforced by :func:`guard_memory` first.
    :param splitter_name: the caller, for error messages.
    :param n_jobs: worker count. Results never depend on it.
    :raises ScalabilityError: if the matrix would exceed ``max_memory_bytes``.
    :return: an ``(n, n)`` float32 distance matrix with a zero diagonal.
    """
    return _pairwise(ctx, featurizer, metric, max_memory_bytes, splitter_name, n_jobs)


def compute_similarity_matrix(
    ctx: Any,
    featurizer: str | Featurizer,
    metric: MetricName,
    max_memory_bytes: int,
    splitter_name: str,
    n_jobs: int = 1,
) -> np.ndarray:
    """Build the full pairwise similarity matrix for a context's records.

    :param ctx: the split context supplying the records.
    :param featurizer: an alias or a featurizer instance.
    :param metric: the metric, converted to a similarity as ``1 - distance``.
    :param max_memory_bytes: ceiling enforced by :func:`guard_memory` first.
    :param splitter_name: the caller, for error messages.
    :param n_jobs: worker count. Results never depend on it.
    :raises ScalabilityError: if the matrix would exceed ``max_memory_bytes``.
    :return: an ``(n, n)`` float32 similarity matrix. The diagonal is exactly ``1.0``, since
        :func:`chemsplit.metrics.pairwise_distances` forces a zero diagonal.
    """
    D = _pairwise(ctx, featurizer, metric, max_memory_bytes, splitter_name, n_jobs)
    return 1.0 - D
