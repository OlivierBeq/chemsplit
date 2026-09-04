"""Tests for chemsplit.splitters.protocol (group_k_fold-applicability_domain)."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit.registry import get_splitter
from chemsplit.splitters.baseline import RandomSplitter
from chemsplit.splitters.protocol import (
    ApplicabilityDomainSplitter,
    ExternalHoldoutSplitter,
    GroupKFoldSplitter,
    NestedCVSplitter,
    RepeatedSplitter,
    ThreeWaySplitter,
)

SMILES_POOL = [
    "CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1", "c1ccncc1", "c1ccccc1O",
    "c1ccccc1N", "CC(=O)O", "CC(=O)N", "CCCCO", "CCCCN", "CCCCC", "c1ccoc1",
    "c1ccsc1", "C1CCCCC1", "C1CCNCC1", "C1CCOCC1", "CC(C)O", "CC(C)N",
    "CCCCCCO", "CCCCCCN", "c1ccc2ccccc2c1", "CC(=O)OC", "CCOC(=O)C",
    "CCCCCCCCO", "CCCCCCCCN", "c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1",
    "CN1CCCCC1", "O=C1CCCCC1", "OC1CCCCC1", "NC1CCCCC1", "CC1CCCCC1",
    "c1cc2ccccc2cc1", "CCN(CC)CC", "CCOCC", "CC#N", "CC=O",
]


def _assert_valid_split(r, n):
    all_idx = np.concatenate([r.train, r.valid, r.test, r.discard])
    assert np.array_equal(np.sort(all_idx), np.arange(n))
    assert len(set(all_idx.tolist())) == n


class TestGroupKFoldSplitter:
    def test_folds_atomic_and_cover(self):
        s = GroupKFoldSplitter(grouper="murcko_scaffold", n_splits=3, random_state=0)
        results = s.split_result(SMILES_POOL)
        assert len(results) == 3
        for r in results:
            _assert_valid_split(r, len(SMILES_POOL))
        # each test fold disjoint
        test_sets = [set(r.test.tolist()) for r in results]
        for i in range(3):
            for j in range(i + 1, 3):
                assert test_sets[i].isdisjoint(test_sets[j])

    def test_auto_n_splits_equals_group_count(self):
        s = GroupKFoldSplitter(grouper="murcko_scaffold", n_splits="auto", random_state=0)
        n_splits = s.get_n_splits(SMILES_POOL)
        results = s.split_result(SMILES_POOL)
        assert len(results) == n_splits

    def test_string_grouper_resolution_via_registry(self):
        s = GroupKFoldSplitter(grouper="butina", n_splits=2, random_state=0)
        results = s.split_result(SMILES_POOL)
        assert len(results) == 2

    def test_rejects_size_spec(self):
        with pytest.raises(Exception):
            GroupKFoldSplitter(grouper="murcko_scaffold", train_size=0.8)

    def test_determinism(self):
        s1 = GroupKFoldSplitter(grouper="murcko_scaffold", n_splits=3, random_state=42)
        s2 = GroupKFoldSplitter(grouper="murcko_scaffold", n_splits=3, random_state=42)
        r1 = s1.split_result(SMILES_POOL)
        r2 = s2.split_result(SMILES_POOL)
        for a, b in zip(r1, r2):
            assert np.array_equal(a.train, b.train)
            assert np.array_equal(a.test, b.test)


class TestThreeWaySplitter:
    def test_index_space_and_completeness(self):
        base = get_splitter("random", shuffle=True)
        s = ThreeWaySplitter(
            base_splitter=base, train_size=0.5, valid_size=0.25, test_size=0.25, random_state=0
        )
        r = s.split_result(SMILES_POOL)[0]
        _assert_valid_split(r, len(SMILES_POOL))
        assert r.train.max() < len(SMILES_POOL)
        assert r.valid.size > 0
        assert set(r.train.tolist()).isdisjoint(set(r.valid.tolist()))
        assert set(r.train.tolist()).isdisjoint(set(r.test.tolist()))
        assert set(r.valid.tolist()).isdisjoint(set(r.test.tolist()))

    def test_string_base_splitter(self):
        s = ThreeWaySplitter(
            base_splitter="random", train_size=0.5, valid_size=0.25, test_size=0.25,
            random_state=0,
        )
        r = s.split_result(SMILES_POOL)[0]
        _assert_valid_split(r, len(SMILES_POOL))


class TestRepeatedSplitter:
    def test_repeats_differ(self):
        s = RepeatedSplitter(base_splitter="random", n_repeats=4, random_state=0)
        results = s.split_result(SMILES_POOL)
        assert len(results) == 4
        for k, r in enumerate(results):
            assert r.metadata["repeat_index"] == k
            _assert_valid_split(r, len(SMILES_POOL))
        test_sets = [tuple(r.test.tolist()) for r in results]
        assert len(set(test_sets)) > 1  # not all identical

    def test_get_n_splits(self):
        s = RepeatedSplitter(base_splitter="random", n_repeats=7, random_state=0)
        assert s.get_n_splits() == 7


class TestNestedCVSplitter:
    def test_reindexing_correct(self):
        s = NestedCVSplitter(
            outer_splitter=get_splitter("k_fold", n_splits=3, shuffle=True, random_state=0),
            inner_splitter="random",
            random_state=0,
        )
        results = s.split_result(SMILES_POOL)
        assert len(results) == 3
        for r in results:
            _assert_valid_split(r, len(SMILES_POOL))
            assert set(r.train.tolist()).isdisjoint(set(r.valid.tolist()))
            assert set(r.train.tolist()).isdisjoint(set(r.test.tolist()))
            assert set(r.valid.tolist()).isdisjoint(set(r.test.tolist()))


class TestExternalHoldoutSplitter:
    def test_external_becomes_test(self):
        external = ["CCCCCCCCCC", "c1ccc(I)cc1"]
        s = ExternalHoldoutSplitter(X_external=external)
        r = s.split_result(SMILES_POOL)[0]
        n = len(SMILES_POOL)
        assert r.n_records == n + len(external)
        assert np.array_equal(r.train, np.arange(n))
        assert np.array_equal(r.test, np.arange(n, n + len(external)))

    def test_requires_external(self):
        with pytest.raises(Exception):
            ExternalHoldoutSplitter()


class TestApplicabilityDomainSplitter:
    def test_bands_ordered_and_complete(self):
        s = ApplicabilityDomainSplitter(n_bands=3, train_size=0.6, test_size=0.4, random_state=0)
        results = s.split_result(SMILES_POOL)
        assert len(results) <= 3
        seen_test = set()
        prev_max = -1.0
        for r in sorted(results, key=lambda x: x.metadata["band_index"]):
            _assert_valid_split(r, len(SMILES_POOL))
            lo, hi = r.metadata["distance_range"]
            assert lo <= hi
            assert lo >= prev_max - 1e-9
            prev_max = hi
            seen_test.update(r.test.tolist())
        # every band shares the same train set
        trains = {tuple(r.train.tolist()) for r in results}
        assert len(trains) == 1

    def test_get_n_splits(self):
        s = ApplicabilityDomainSplitter(n_bands=5, random_state=0)
        assert s.get_n_splits() == 5
