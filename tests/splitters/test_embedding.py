"""Tests for chemsplit.splitters.embedding (umap_cluster UMAPClusterSplitter,
projection ProjectionSplitter, latent_space LatentSpaceSplitter)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from chemsplit.datasets import make_two_clusters
from chemsplit.exceptions import CircularityWarning, DeterminismWarning, ParameterError
from chemsplit.splitters.embedding import (
    LatentSpaceSplitter,
    ProjectionSplitter,
    UMAPClusterSplitter,
)


@pytest.fixture
def smiles_60():
    fx = make_two_clusters(n=60, seed=0)
    return fx.smiles


@pytest.fixture
def features_60():
    rng = np.random.default_rng(0)
    return rng.standard_normal((60, 16))


# ---------------------------------------------------------------------------
# UMAPClusterSplitter
# ---------------------------------------------------------------------------


def test_umap_splitter_basic(smiles_60):
    splitter = UMAPClusterSplitter(train_size=0.7, test_size=0.3, random_state=0, n_neighbors=5)
    [result] = splitter.split_result(smiles_60)
    assert result.splitter_id == "umap_cluster"
    assert result.metadata["n_clusters"] >= 2
    assert len(result.metadata["cluster_sizes"]) == result.metadata["n_clusters"]
    assert set(result.metadata["umap_versions"]) == {"umap", "numba", "pynndescent"}
    assert result.train.size + result.test.size + result.valid.size + result.discard.size == 60


def test_umap_splitter_deterministic(smiles_60):
    a = UMAPClusterSplitter(train_size=0.7, test_size=0.3, random_state=0, n_neighbors=5).split_result(smiles_60)[0]
    b = UMAPClusterSplitter(train_size=0.7, test_size=0.3, random_state=0, n_neighbors=5).split_result(smiles_60)[0]
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)
    assert np.array_equal(a.groups, b.groups)


def test_umap_splitter_n_jobs_warning(smiles_60):
    splitter = UMAPClusterSplitter(
        train_size=0.7, test_size=0.3, random_state=0, n_neighbors=5, n_jobs=4
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        splitter.split_result(smiles_60)
    assert any(issubclass(w.category, DeterminismWarning) for w in caught)


def test_umap_splitter_n_too_small(smiles_60):
    splitter = UMAPClusterSplitter(train_size=0.7, test_size=0.3, n_neighbors=100)
    with pytest.raises(ParameterError):
        splitter.split_result(smiles_60)


# ---------------------------------------------------------------------------
# ProjectionSplitter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pca", "svd", "kernel_pca"])
def test_projection_splitter_cluster_mode(features_60, method):
    splitter = ProjectionSplitter(
        method=method, mode="cluster", n_components=3, train_size=0.7, test_size=0.3, random_state=0,
    )
    [result] = splitter.split_result(features_60, X_kind="features")
    assert result.splitter_id == "projection"
    assert result.metadata["method"] == method
    assert result.metadata["mode"] == "cluster"
    assert result.metadata["nondeterministic_method"] is False


def test_projection_splitter_pca_deterministic(features_60):
    a = ProjectionSplitter(method="pca", train_size=0.7, test_size=0.3, random_state=0).split_result(
        features_60, X_kind="features"
    )[0]
    b = ProjectionSplitter(method="pca", train_size=0.7, test_size=0.3, random_state=0).split_result(
        features_60, X_kind="features"
    )[0]
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.groups, b.groups)


def test_projection_splitter_axis_cut(features_60):
    splitter = ProjectionSplitter(
        method="pca", mode="axis_cut", axis=0, n_components=2, train_size=0.7, test_size=0.3, random_state=0,
    )
    [result] = splitter.split_result(features_60, X_kind="features")
    assert result.train.size + result.test.size <= 60
    assert result.metadata["mode"] == "axis_cut"


def test_projection_splitter_grid(features_60):
    splitter = ProjectionSplitter(
        method="pca", mode="grid", n_components=2, grid_bins=3, train_size=0.7, test_size=0.3, random_state=0,
    )
    [result] = splitter.split_result(features_60, X_kind="features")
    assert result.metadata["mode"] == "grid"


def test_projection_splitter_tsne_nondeterministic_flag(features_60):
    splitter = ProjectionSplitter(
        method="tsne", mode="cluster", n_components=2, tsne_perplexity=5.0,
        train_size=0.7, test_size=0.3, random_state=0,
    )
    [result] = splitter.split_result(features_60, X_kind="features")
    assert result.metadata["nondeterministic_method"] is True
    assert result.metadata["kl_divergence"] is not None


def test_projection_splitter_mds(features_60):
    splitter = ProjectionSplitter(
        method="mds", mode="cluster", n_components=2, train_size=0.7, test_size=0.3, random_state=0,
    )
    [result] = splitter.split_result(features_60, X_kind="features")
    assert result.metadata["nondeterministic_method"] is True


def test_projection_splitter_perplexity_too_large(features_60):
    splitter = ProjectionSplitter(method="tsne", tsne_perplexity=50.0, train_size=0.7, test_size=0.3)
    with pytest.raises(ParameterError):
        splitter.split_result(features_60, X_kind="features")


def test_projection_splitter_invalid_method():
    with pytest.raises(ParameterError):
        ProjectionSplitter(method="bogus")


# ---------------------------------------------------------------------------
# LatentSpaceSplitter
# ---------------------------------------------------------------------------


def test_latent_space_splitter_warns_without_independence(features_60):
    splitter = LatentSpaceSplitter(train_size=0.7, test_size=0.3, random_state=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        splitter.split_result(features_60, X_kind="features")
    assert any(issubclass(w.category, CircularityWarning) for w in caught)


def test_latent_space_splitter_no_warning_when_declared(features_60):
    splitter = LatentSpaceSplitter(
        train_size=0.7, test_size=0.3, random_state=0, independence_declared=True
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        splitter.split_result(features_60, X_kind="features")
    assert not any(issubclass(w.category, CircularityWarning) for w in caught)


def test_latent_space_splitter_deterministic(features_60):
    a = LatentSpaceSplitter(train_size=0.7, test_size=0.3, random_state=0, independence_declared=True)
    b = LatentSpaceSplitter(train_size=0.7, test_size=0.3, random_state=0, independence_declared=True)
    ra = a.split_result(features_60, X_kind="features")[0]
    rb = b.split_result(features_60, X_kind="features")[0]
    assert np.array_equal(ra.train, rb.train)
    assert ra.metadata["embedding_hash"] == rb.metadata["embedding_hash"]


def test_latent_space_splitter_callable_embedding(features_60):
    def encoder(raw_features):
        assert raw_features.shape == features_60.shape
        rng = np.random.default_rng(1)
        return rng.standard_normal((60, 4))

    splitter = LatentSpaceSplitter(
        embedding=encoder, train_size=0.7, test_size=0.3, random_state=0, independence_declared=True,
    )
    [result] = splitter.split_result(features_60, X_kind="features")
    assert result.metadata["embedding_dim"] == 4


def test_latent_space_splitter_wrong_length_raises(features_60):
    bad = np.random.default_rng(0).standard_normal((59, 4))
    splitter = LatentSpaceSplitter(
        embedding=bad, train_size=0.7, test_size=0.3, independence_declared=True,
    )
    with pytest.raises(Exception):
        splitter.split_result(features_60, X_kind="features")


def test_latent_space_splitter_non_finite_raises():
    bad = np.full((60, 4), np.nan)
    splitter = LatentSpaceSplitter(
        embedding=bad, train_size=0.7, test_size=0.3, independence_declared=True,
    )
    with pytest.raises(Exception):
        splitter.split_result(bad, X_kind="features")
