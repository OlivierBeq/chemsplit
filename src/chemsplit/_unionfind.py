"""Shared Union-Find and dense-label-encoding primitives.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from chemsplit.types import IndexArray

__all__ = ["UnionFind", "dense_label_encode", "merge_group_labels"]


class UnionFind:
    """Path-compressed Union-Find with union-by-smallest-index.

    Unlike textbook union-by-rank/size, this always keeps the *smallest original member index* as
    a component's representative, so it's cheap to query via :meth:`find`.
    """

    __slots__ = ("_parent",)

    def __init__(self, n: int) -> None:
        self._parent: list[int] = list(range(n))

    def find(self, i: int) -> int:
        root = i
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[i] != root:
            self._parent[i], i = root, self._parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        # union-by-smallest-index: the smaller root always becomes the parent
        lo, hi = (ri, rj) if ri < rj else (rj, ri)
        self._parent[hi] = lo

    def components(self) -> dict[int, list[int]]:
        """Return ``{representative: sorted_member_list}``, keyed by ascending representative and
        with each member list itself sorted ascending."""
        members: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            members.setdefault(self.find(i), []).append(i)
        return dict(sorted(members.items()))


def dense_label_encode(keys: Sequence[Hashable]) -> IndexArray:
    """Map distinct ``keys`` to dense ``0..g-1`` int64 ids, assigned in **first-appearance order**
    — not sorted order.
    """
    next_id = 0
    seen: dict[Hashable, int] = {}
    out = np.empty(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        label = seen.get(key)
        if label is None:
            label = next_id
            seen[key] = label
            next_id += 1
        out[i] = label
    return out


def merge_group_labels(a: IndexArray, b: IndexArray) -> IndexArray:
    """Union-find merge of two per-record label arrays into dense component labels.

    Two records end up in the same output group iff they share a label under ``a``, share a label
    under ``b``, or are chain-connected through other records via either array. Output ids are
    dense (first-appearance order of each record's Union-Find representative, scanning ascending
    record index).
    """
    n = len(a)
    if len(b) != n:
        raise ValueError(f"merge_group_labels: length mismatch ({n} vs {len(b)})")
    uf = UnionFind(n)
    by_a: dict[object, int] = {}
    by_b: dict[object, int] = {}
    for i in range(n):
        ka, kb = a[i].item(), b[i].item()
        if ka in by_a:
            uf.union(i, by_a[ka])
        else:
            by_a[ka] = i
        if kb in by_b:
            uf.union(i, by_b[kb])
        else:
            by_b[kb] = i
    return dense_label_encode([uf.find(i) for i in range(n)])
