"""Direct coverage for chemsplit/registry.py's own logic, beyond the incidental exercise every
other test file already gives it via get_splitter()/SPLITTER_REGISTRY access.
"""

from __future__ import annotations

import pytest

from chemsplit import registry
from chemsplit.exceptions import UnknownSplitterError


class TestGetSplitterLookupModes:
    def test_by_id(self):
        s = registry.get_splitter("random")
        assert s.splitter_id == "random"

    def test_by_class_name(self):
        s = registry.get_splitter("RandomSplitter")
        assert type(s).__name__ == "RandomSplitter"

    def test_by_id_case_insensitive(self):
        s1 = registry.get_splitter("random")
        s2 = registry.get_splitter("RANDOM")
        s3 = registry.get_splitter("Random")
        assert type(s1).__name__ == type(s2).__name__ == type(s3).__name__ == "RandomSplitter"

    def test_kwargs_forwarded(self):
        s = registry.get_splitter("random", random_state=42)
        assert s.random_state == 42

    def test_unknown_name_raises_with_suggestions(self):
        with pytest.raises(UnknownSplitterError) as exc_info:
            registry.get_splitter("randmo")  # typo of "random"
        assert "random" in str(exc_info.value)


class TestListSplitters:
    def test_returns_all_52_by_default(self):
        df = registry.list_splitters()
        assert len(df) == 52
        assert set(df.columns) == {
            "id", "class_name", "family", "family_name", "strictness",
            "group_forming", "requires_labels", "requires_dates", "requires_targets",
            "deterministic_without_seed", "extras",
        }

    def test_family_filter(self):
        df = registry.list_splitters(family="protocol")
        assert len(df) == 6
        assert set(df["family"]) == {"protocol"}

    def test_strictness_filter(self):
        df = registry.list_splitters(strictness="strict")
        assert len(df) > 0
        assert set(df["strictness"]) == {"strict"}

    def test_group_forming_filter(self):
        df_grouping = registry.list_splitters(group_forming=True)
        df_non_grouping = registry.list_splitters(group_forming=False)
        assert len(df_grouping) + len(df_non_grouping) == 52
        assert set(df_grouping["group_forming"]) == {True}
        assert set(df_non_grouping["group_forming"]) == {False}

    def test_combined_filters(self):
        df = registry.list_splitters(family="scaffold", group_forming=True)
        assert len(df) > 0
        assert set(df["family"]) == {"scaffold"}
        assert set(df["group_forming"]) == {True}

    def test_no_matches_returns_empty(self):
        df = registry.list_splitters(family="scaffold", strictness="optimistic")
        assert len(df) == 0


class TestCamelToSnake:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("RandomSplitter", "random"),
            ("KFoldSplitter", "k_fold"),
            ("BalancedMultiTaskSplitter", "balanced_multi_task"),
            ("UMAPClusterSplitter", "umap_cluster"),
            ("MOODSplitter", "mood"),
        ],
    )
    def test_known_mappings(self, name, expected):
        assert registry._camel_to_snake(name) == expected


class TestLevenshtein:
    def test_identical_strings(self):
        assert registry._levenshtein("random", "random") == 0

    def test_empty_string_either_side(self):
        assert registry._levenshtein("", "abc") == 3
        assert registry._levenshtein("abc", "") == 3

    def test_one_substitution(self):
        assert registry._levenshtein("cat", "bat") == 1
