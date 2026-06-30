from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from chemsplit import _fp_similarity as fps
from chemsplit.exceptions import ParameterError, ScalabilityError


class _FakeContext:
    """Minimal stand-in for chemsplit.base._Context exposing only what _fp_similarity needs."""

    def __init__(self, features: np.ndarray) -> None:
        self.n = features.shape[0]
        self._features = features

    def get_features(self, featurizer=None):
        return self._features


def _toy_features() -> np.ndarray:
    rng = np.random.default_rng(0)
    dense = (rng.random((8, 32)) > 0.7).astype(np.uint8)
    return sp.csr_matrix(dense)


def test_compute_similarity_matrix_symmetric_and_unit_diagonal():
    ctx = _FakeContext(_toy_features())
    S = fps.compute_similarity_matrix(
        ctx, featurizer="ecfp4", metric="tanimoto", max_memory_bytes=2 * 1024**3,
        splitter_name="toy",
    )
    assert S.shape == (8, 8)
    assert np.allclose(np.diag(S), 1.0)
    assert np.allclose(S, S.T)


def test_compute_distance_matrix_zero_diagonal():
    ctx = _FakeContext(_toy_features())
    D = fps.compute_distance_matrix(
        ctx, featurizer="ecfp4", metric="tanimoto", max_memory_bytes=2 * 1024**3,
        splitter_name="toy",
    )
    assert np.allclose(np.diag(D), 0.0)


def test_guard_memory_raises_when_exceeded():
    with pytest.raises(ScalabilityError):
        fps.guard_memory(n=100_000, max_memory_bytes=1024, splitter_name="toy")


def test_guard_memory_passes_when_within_budget():
    fps.guard_memory(n=10, max_memory_bytes=2 * 1024**3, splitter_name="toy")  # no raise


def test_guard_memory_message_names_splitter_and_alternatives():
    with pytest.raises(ScalabilityError) as exc_info:
        fps.guard_memory(n=100_000, max_memory_bytes=1024, splitter_name="MySplitter")
    msg = str(exc_info.value)
    assert "MySplitter" in msg
    assert "100000" in msg or "100,000" in msg


class _ToySimilaritySplitter(fps.SimilarityParamsMixin):
    bounded_metric_required = True

    def __init__(self, **kw):
        super().__init__(**kw)


class _ToyUnboundedOkSplitter(fps.SimilarityParamsMixin):
    bounded_metric_required = False

    def __init__(self, **kw):
        super().__init__(**kw)


def test_validate_similarity_params_rejects_unbounded_metric_when_required():
    s = _ToySimilaritySplitter(metric="euclidean")
    with pytest.raises(ParameterError):
        s._validate_similarity_params()


def test_validate_similarity_params_accepts_bounded_metric_when_required():
    s = _ToySimilaritySplitter(metric="tanimoto")
    s._validate_similarity_params()  # no raise


def test_validate_similarity_params_accepts_unbounded_metric_when_not_required():
    s = _ToyUnboundedOkSplitter(metric="euclidean")
    s._validate_similarity_params()  # no raise


def test_validate_similarity_params_rejects_unknown_metric():
    s = _ToyUnboundedOkSplitter(metric="not_a_metric")
    with pytest.raises(ParameterError):
        s._validate_similarity_params()


def test_validate_similarity_params_rejects_bad_max_memory_bytes():
    s = _ToyUnboundedOkSplitter(metric="tanimoto", max_memory_bytes=0)
    with pytest.raises(ParameterError):
        s._validate_similarity_params()

    s2 = _ToyUnboundedOkSplitter(metric="tanimoto", max_memory_bytes=True)
    with pytest.raises(ParameterError):
        s2._validate_similarity_params()


def test_resolve_featurizer_passthrough_and_alias():
    feat = fps.resolve_featurizer("ecfp4")
    assert feat.name == "ecfp4"
    # passthrough of an already-constructed Featurizer instance
    assert fps.resolve_featurizer(feat) is feat
