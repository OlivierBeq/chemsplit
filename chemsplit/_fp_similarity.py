"""Shared fingerprint/metric machinery for the ``similarity`` splitter family and ``hi``.
"""

from __future__ import annotations

from typing import Any, Literal

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

#: Fixed float-comparison tolerance. "within cutoff" means ``d <= cutoff + EPS``;
#: "exceeds threshold" means ``s > threshold + EPS``. Shared so no splitter hardcodes its own copy.
EPS = 1e-9


def resolve_featurizer(spec_or_instance: "str | Featurizer", **kw: Any) -> Featurizer:
    """Thin wrapper around :func:`chemsplit.featurizers.get_featurizer`.

    Lets callers that only import ``chemsplit._fp_similarity`` resolve a featurizer without a
    direct dependency on ``chemsplit.featurizers``.
    """
    return get_featurizer(spec_or_instance, **kw)


def guard_memory(
    n: int,
    max_memory_bytes: int,
    splitter_name: str,
    *,
    alternatives: list[str] | None = None,
) -> None:
    """Memory guard: any splitter materialising a full ``n x n`` matrix MUST call this first.

    Raises :class:`~chemsplit.exceptions.ScalabilityError` naming the splitter, ``n``, the
    required bytes, and 2-3 alternatives, if ``n**2 * 4`` bytes (a float32 dense matrix) exceeds
    ``max_memory_bytes``.
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
        featurizer: "str | Featurizer" = "ecfp4",
        metric: MetricName = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        n_jobs: int = 1,
    ) -> None:
        self.featurizer = featurizer
        self.metric = metric
        self.max_memory_bytes = max_memory_bytes
        self.n_jobs = n_jobs

    def _validate_similarity_params(self, *, bounded_metric_required: bool | None = None) -> None:
        required = (
            self.bounded_metric_required if bounded_metric_required is None else bounded_metric_required
        )
        valid_metrics = {"tanimoto", "dice", "cosine", "euclidean", "manhattan", "tanimoto_count"}
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
        if isinstance(self.max_memory_bytes, bool) or not isinstance(self.max_memory_bytes, (int, np.integer)):
            raise ParameterError(f"max_memory_bytes must be an int, got {self.max_memory_bytes!r}")
        if self.max_memory_bytes <= 0:
            raise ParameterError(f"max_memory_bytes must be > 0, got {self.max_memory_bytes!r}")


def _pairwise(
    ctx: Any,
    featurizer: "str | Featurizer",
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
    featurizer: "str | Featurizer",
    metric: MetricName,
    max_memory_bytes: int,
    splitter_name: str,
    n_jobs: int = 1,
) -> np.ndarray:
    """Full pairwise distance matrix for ``ctx``'s records, after the memory guard."""
    return _pairwise(ctx, featurizer, metric, max_memory_bytes, splitter_name, n_jobs)


def compute_similarity_matrix(
    ctx: Any,
    featurizer: "str | Featurizer",
    metric: MetricName,
    max_memory_bytes: int,
    splitter_name: str,
    n_jobs: int = 1,
) -> np.ndarray:
    """Full pairwise similarity matrix (``1 - distance``) for ``ctx``'s records.

    Diagonal is exactly ``1.0`` since :func:`chemsplit.metrics.pairwise_distances` forces
    ``D[i][i] == 0.0``.
    """
    D = _pairwise(ctx, featurizer, metric, max_memory_bytes, splitter_name, n_jobs)
    return 1.0 - D
