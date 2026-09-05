"""Clustering and diversity-selection primitives shared by several splitter families.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import scipy.sparse as sp

from chemsplit.determinism import argmax_tiebreak, argmin_tiebreak

EPS = 1e-9

__all__ = [
    "butina",
    "kennard_stone",
    "leader",
    "maxmin_pick",
    "spectral_partition",
    "sphere_exclusion",
]


def _neighbor_lists(D: np.ndarray, cutoff: float) -> list[np.ndarray]:
    n = D.shape[0]
    neigh = []
    for i in range(n):
        row = D[i]
        idx = np.nonzero(row <= cutoff + EPS)[0]
        idx = idx[idx != i]
        neigh.append(np.sort(idx))
    return neigh


def butina(D: np.ndarray, cutoff: float, reorder: bool = False) -> list[list[int]]:
    """Taylor-Butina sphere-exclusion (leader) clustering.

    ``D`` is a dense symmetric distance matrix (``D[i][i] == 0``). Returns clusters in creation
    order; each cluster's first element is its centroid. Deterministic: ties always resolve to the
    smallest record index.
    """
    n = D.shape[0]
    neigh = _neighbor_lists(D, cutoff)
    assigned = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []

    if not reorder:
        counts = np.array([len(neigh[i]) for i in range(n)])
        # stable_sort by count descending, ties -> ascending index: sort by (-count, index).
        order = sorted(range(n), key=lambda i: (-counts[i], i))
        for i in order:
            if assigned[i]:
                continue
            members = [i] + [int(j) for j in neigh[i] if not assigned[j]]
            for m in members:
                assigned[m] = True
            clusters.append(members)
    else:
        while not np.all(assigned):
            live = [i for i in range(n) if not assigned[i]]
            counts = {i: int(np.sum(~assigned[neigh[i]])) if len(neigh[i]) else 0 for i in live}
            i = argmax_tiebreak(lambda idx: counts[idx], live)
            members = [i] + [int(j) for j in neigh[i] if not assigned[j]]
            for m in members:
                assigned[m] = True
            clusters.append(members)

    return clusters


def leader(D: np.ndarray, radius: float, order: Sequence[int] | None = None) -> list[list[int]]:
    """Greedy leader-follower clustering: index-order-deterministic, single pass.

    Each point either joins the nearest existing leader within ``radius``, or becomes a new
    leader itself. ``order`` (default ascending index) fixes the scan order and hence which points
    become leaders.
    """
    n = D.shape[0]
    scan = list(range(n)) if order is None else list(order)
    leaders: list[int] = []
    clusters: list[list[int]] = []
    for i in scan:
        if not leaders:
            leaders.append(i)
            clusters.append([i])
            continue
        dists = [D[i, ldr] for ldr in leaders]
        best = argmin_tiebreak(lambda k: dists[k], range(len(leaders)))
        if dists[best] <= radius + EPS:
            clusters[best].append(i)
        else:
            leaders.append(i)
            clusters.append([i])
    return clusters


def sphere_exclusion(
    D: np.ndarray, radius: float, order: Sequence[int] | None = None
) -> tuple[list[int], list[list[int]]]:
    """Greedy sphere-exclusion selection.

    Repeatedly selects the next unexcluded point (by ``order``, default ascending index) as a
    representative, then excludes every remaining point within ``radius`` of it. Returns
    ``(representatives, groups)`` where ``groups[k]`` are the points excluded by (and including)
    ``representatives[k]``.
    """
    n = D.shape[0]
    scan = list(range(n)) if order is None else list(order)
    excluded = np.zeros(n, dtype=bool)
    reps: list[int] = []
    groups: list[list[int]] = []
    for i in scan:
        if excluded[i]:
            continue
        reps.append(i)
        within = np.nonzero(D[i] <= radius + EPS)[0]
        group = [int(j) for j in within if not excluded[j]]
        for j in group:
            excluded[j] = True
        groups.append(group)
    return reps, groups


def maxmin_pick(
    D: np.ndarray,
    n_picks: int,
    init: Literal["random", "kennard_stone", "most_peripheral", "index_zero"] = "random",
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Greedy MaxMin (Kennard-Stone family) diversity selection.

    Iteratively picks the unpicked point maximising its minimum distance to the already-picked
    set, breaking ties by smallest index.
    """
    n = D.shape[0]
    if init == "kennard_stone":
        picked = list(kennard_stone(D, 2)) if n >= 2 else [0]
    elif init == "most_peripheral":
        mean_dist = D.mean(axis=1)
        picked = [argmax_tiebreak(lambda idx: mean_dist[idx], range(n))]
    elif init == "index_zero":
        picked = [0]
    else:  # "random"
        assert rng is not None, "rng is required for init='random'"
        picked = [int(rng.integers(0, n))]

    picked = picked[:n_picks]
    mind = np.min(D[:, picked], axis=1) if picked else np.full(n, np.inf)
    picked_set = set(picked)

    while len(picked) < n_picks:
        candidates = [i for i in range(n) if i not in picked_set]
        if not candidates:
            break
        next_i = argmax_tiebreak(lambda idx: mind[idx], candidates)
        picked.append(next_i)
        picked_set.add(next_i)
        mind = np.minimum(mind, D[:, next_i])

    return picked


def kennard_stone(D: np.ndarray, n_picks: int) -> list[int]:
    """Kennard-Stone selection: the first two picks are the maximally distant pair.

    Ties broken by lexicographically smallest ``(i, j)``.
    """
    n = D.shape[0]
    if n < 2:
        return list(range(n))[:n_picks]
    iu = np.triu_indices(n, k=1)
    dvals = D[iu]
    max_d = dvals.max()
    candidates = [(int(iu[0][k]), int(iu[1][k])) for k in range(len(dvals)) if dvals[k] >= max_d - EPS]
    i0, j0 = min(candidates)
    picked = [i0, j0]
    mind = np.minimum(D[:, i0], D[:, j0])
    picked_set = set(picked)
    while len(picked) < n_picks:
        candidates_idx = [i for i in range(n) if i not in picked_set]
        if not candidates_idx:
            break
        next_i = argmax_tiebreak(lambda idx: mind[idx], candidates_idx)
        picked.append(next_i)
        picked_set.add(next_i)
        mind = np.minimum(mind, D[:, next_i])
    return picked


def _fix_eigenvector_signs(U: np.ndarray) -> np.ndarray:
    """Deterministic sign fix: for each column, negate if the largest-|value| entry is negative."""
    U = U.copy()
    for c in range(U.shape[1]):
        col = U[:, c]
        abs_col = np.abs(col)
        idx = argmax_tiebreak(lambda k: abs_col[k], range(len(col)))
        if col[idx] < 0:
            U[:, c] = -col
    return U


def spectral_partition(
    W: np.ndarray | sp.spmatrix,
    n_clusters: int,
    laplacian: Literal["sym", "rw", "unnormalized"] = "sym",
    drop_first: bool = True,
    assign: Literal["kmeans", "discretize"] = "kmeans",
    rng: np.random.Generator | None = None,
    random_state: int = 0,
) -> np.ndarray:
    """Laplacian-eigenmap spectral partition. Returns a dense-label-encoded int array of length n.

    Isolated vertices (zero row-sum in ``W``) are excluded from the eigenproblem and reattached
    afterwards as their own singleton clusters, per design.
    """
    if sp.issparse(W):
        W = W.toarray()
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[0]
    deg = W.sum(axis=1)
    isolated = deg <= 0.0
    active = np.nonzero(~isolated)[0]

    labels = np.full(n, -1, dtype=np.int64)
    next_label = 0

    if active.size > 0:
        Wa = W[np.ix_(active, active)]
        dega = Wa.sum(axis=1)
        if laplacian == "unnormalized":
            L = np.diag(dega) - Wa
        elif laplacian == "rw":
            Dinv = np.diag(1.0 / dega)
            L = np.eye(len(active)) - Dinv @ Wa
        else:  # "sym"
            Dinv_sqrt = np.diag(1.0 / np.sqrt(dega))
            L = np.eye(len(active)) - Dinv_sqrt @ Wa @ Dinv_sqrt

        k = min(n_clusters + (1 if drop_first else 0), len(active) - 1)
        k = max(k, 1)

        if rng is None:
            rng = np.random.default_rng(random_state)
        v0 = rng.standard_normal(L.shape[0])

        if len(active) <= k + 1 or len(active) < 50:
            # Small enough for a dense eigensolve (also avoids ARPACK convergence issues on tiny
            # matrices); still deterministic (LAPACK's symmetric eigensolver, no v0 needed).
            vals, vecs = np.linalg.eigh(L)
        else:
            from scipy.sparse.linalg import eigsh

            vals, vecs = eigsh(L, k=k, which="SM", v0=v0, tol=0.0, maxiter=5000)

        order = np.argsort(vals, kind="stable")
        vecs = vecs[:, order]
        start = 1 if drop_first else 0
        U = vecs[:, start: start + n_clusters]
        U = _fix_eigenvector_signs(U)
        if laplacian == "sym":
            norms = np.linalg.norm(U, axis=1, keepdims=True)
            nonzero = norms.ravel() > 0
            U = U.copy()
            U[nonzero] = U[nonzero] / norms[nonzero]

        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=min(n_clusters, len(active)), n_init=10, random_state=random_state)
        sub_labels = km.fit_predict(U)
        for local_i, global_i in enumerate(active):
            labels[global_i] = sub_labels[local_i]
        next_label = int(sub_labels.max()) + 1 if len(sub_labels) else 0

    for i in np.nonzero(isolated)[0]:
        labels[i] = next_label
        next_label += 1

    # dense-label-encode by first appearance to keep ids contiguous and order-stable
    seen: dict[int, int] = {}
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        lbl = int(labels[i])
        if lbl not in seen:
            seen[lbl] = len(seen)
        out[i] = seen[lbl]
    return out
