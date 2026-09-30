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
    "compute_similarity_matrix",
    "guard_memory",
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


def guard_memory(
    n: int,
    max_memory_bytes: int,
    splitter_name: str,
    *,
    alternatives: list[str] | None = None,
) -> None:
    """Refuse to build an ``n x n`` matrix that would not fit.

    Every splitter that materialises a full dense matrix calls this first.

    :param n: the record count.
    :param max_memory_bytes: the ceiling.
    :param splitter_name: the caller, for the error message.
    :param alternatives: splitters to suggest instead, or ``None`` for the defaults.
    :raises ScalabilityError: if a float32 ``n x n`` matrix would exceed the ceiling. The
        :class:`~chemsplit.exceptions.ScalabilityError` names the splitter, ``n``, the bytes
        needed and a few alternatives.
    """
    required = n * n * 4
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
        f"{splitter_name}: a dense {n}x{n} matrix would require {required:,} bytes, "
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
