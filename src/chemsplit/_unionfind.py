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
        """Find ``i``'s representative, compressing the path on the way.

        :param i: a member index.
        :return: the component's representative, always its smallest member.
        """
        root = i
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        curr = i
        while self._parent[curr] != root:
            nxt = self._parent[curr]
            self._parent[curr] = root
            curr = nxt
        return root

    def union(self, i: int, j: int) -> None:
        """Merge the components holding ``i`` and ``j``.

        The smaller index becomes the merged representative.

        :param i: a member index.
        :param j: another member index.
        """
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        # union-by-smallest-index: the smaller root always becomes the parent
        lo, hi = (ri, rj) if ri < rj else (rj, ri)
        self._parent[hi] = lo

    def components(self) -> dict[int, list[int]]:
        """Collect the components.

        :return: representative to its sorted member list, keyed by ascending representative.
        """
        members: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            members.setdefault(self.find(i), []).append(i)
        return dict(sorted(members.items()))


def dense_label_encode(keys: Sequence[Hashable]) -> IndexArray:
    """Map distinct keys to dense ``0..g-1`` ids, in first-appearance order.

    :param keys: one hashable key per record.
    :return: the encoded labels, numbered by first appearance rather than sorted.
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
    """Merge two per-record label arrays into one dense grouping.

    Two records share an output group when they share a label under ``a``, share one under
    ``b``, or are chain-connected through other records via either array.

    :param a: the first per-record label array.
    :param b: the second, of the same length.
    :raises InputError: if the two arrays differ in length.
    :return: dense labels, numbered by the first appearance of each representative.
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
