"""Clustering and diversity-selection primitives shared by several splitter families.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import scipy.sparse as sp

from chemsplit.determinism import (
    argmax_tiebreak,
    argmin_tiebreak,
    first_argmax_2d,
    masked_argmax,
    row_argmin,
)

#: Tolerance for the float32 distance matrices used here (~1.2e-7 rounding noise).
EPS = 1e-6

__all__ = [
    "butina",
    "butina_from_neighbors",
    "duplex_order",
    "kennard_stone",
    "leader",
    "maxmin_pick",
    "maxmin_pick_columns",
    "optisim_pick",
    "optisim_pick_columns",
    "spectral_partition",
    "sphere_exclusion",
    "sphere_exclusion_rows",
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

    Deterministic: ties always resolve to the smallest record index.

    :param D: a dense symmetric distance matrix with a zero diagonal.
    :param cutoff: the cluster radius, as a distance.
    :param reorder: recompute neighbour counts after each cluster is taken, which is the
        original formulation.
    :return: clusters in creation order, each starting with its centroid.
    """
    return butina_from_neighbors(_neighbor_lists(D, cutoff), D.shape[0], reorder=reorder)


def butina_from_neighbors(
    neigh: Sequence[np.ndarray], n: int, reorder: bool = False
) -> list[list[int]]:
    """Taylor-Butina clustering from precomputed radius-neighbour lists.

    The neighbour lists are all this algorithm reads from the distance matrix, and they can be
    built blockwise -- so no ``n x n`` matrix, and bit-identical either way.

    :param neigh: per record, the ascending indices within the cutoff, excluding itself.
    :param n: the record count.
    :param reorder: recompute neighbour counts after each cluster is taken.
    :return: clusters in creation order, each starting with its centroid.
    """
    assigned = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []

    if not reorder:
        counts = np.array([len(neigh[i]) for i in range(n)])
        # sort by (-count, index): count descending, ties to the lower index
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
    """Greedy leader-follower clustering, in one deterministic pass.

    Each point joins the nearest existing leader within ``radius``, or becomes a leader itself.

    :param D: a dense symmetric distance matrix.
    :param radius: how far a point may sit from its leader.
    :param order: the scan order, which decides who becomes a leader. ``None`` uses ascending
        index.
    :return: clusters in creation order, each starting with its leader.
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

    Repeatedly takes the next unexcluded point as a representative, then excludes every
    remaining point within ``radius`` of it.

    :param D: a dense symmetric distance matrix.
    :param radius: the exclusion radius.
    :param order: the scan order. ``None`` uses ascending index.
    :return: the representatives, and per representative the points it claimed, itself
        included.
    """
    return sphere_exclusion_rows(D.shape[0], lambda i: D[i], radius, order=order)


def sphere_exclusion_rows(
    n: int,
    row: Callable[[int], np.ndarray],
    radius: float,
    order: Sequence[int] | None = None,
) -> tuple[list[int], list[list[int]]]:
    """Greedy sphere-exclusion from a row callable instead of a dense matrix.

    One row per representative is all it reads, so rows on demand remove the ``n x n`` matrix.
    Bit-identical: same rows, same radius, same scan order.

    :param n: the record count.
    :param row: returns row ``i`` of the distance matrix, length ``n``.
    :param radius: the exclusion radius.
    :param order: the scan order. ``None`` uses ascending index.
    :return: the representatives, and per representative the points it claimed, itself included.
    """
    scan = list(range(n)) if order is None else list(order)
    excluded = np.zeros(n, dtype=bool)
    reps: list[int] = []
    groups: list[list[int]] = []
    for i in scan:
        if excluded[i]:
            continue
        reps.append(i)
        within = np.nonzero(row(i) <= radius + EPS)[0]
        group = [int(j) for j in within if not excluded[j]]
        excluded[group] = True
        groups.append(group)
    return reps, groups


def maxmin_pick(
    D: np.ndarray,
    n_picks: int,
    init: Literal["random", "kennard_stone", "most_peripheral", "index_zero"] = "random",
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Greedy MaxMin diversity selection, from the Kennard-Stone family.

    Each round takes the unpicked point whose minimum distance to the picked set is largest,
    with ties going to the smallest index.

    :param D: a dense symmetric distance matrix.
    :param n_picks: how many points to select.
    :param init: how the first point is chosen.
    :param rng: generator for ``init="random"``.
    :return: the picked indices, in selection order.
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
    taken = np.zeros(n, dtype=bool)
    taken[picked] = True

    while len(picked) < n_picks:
        if bool(taken.all()):
            break
        next_i = masked_argmax(mind, taken)
        picked.append(next_i)
        taken[next_i] = True
        np.minimum(mind, D[:, next_i], out=mind)

    return picked


def maxmin_pick_columns(
    n: int,
    columns: Callable[[Sequence[int]], np.ndarray],
    n_picks: int,
    first: int,
    batch: int = 512,
) -> list[int]:
    """Greedy MaxMin from a column-block callable, without an ``n x n`` matrix.

    Exactly :func:`maxmin_pick`, reorganised so distances arrive in wide blocks. The running
    minimum stays **exact**, so the argmax is always the true next pick; batching only decides
    when distances are fetched. Single columns are bandwidth-bound, so fetching a block of
    candidates at once is nearly free.

    :param n: the record count.
    :param columns: given ascending indices ``js``, returns the ``(n, len(js))`` distance block.
    :param n_picks: how many points to select.
    :param first: the index of the first pick, chosen by the caller's ``init`` rule.
    :param batch: how many columns to fetch per block.
    :return: the picked indices, in selection order.
    """
    taken = np.zeros(n, dtype=bool)
    taken[first] = True
    picked = [first]
    mind = columns([first])[:, 0].astype(np.float64, copy=True)
    held: dict[int, np.ndarray] = {}

    while len(picked) < n_picks:
        if bool(taken.all()):
            break
        candidate = masked_argmax(mind, taken)
        if candidate not in held:
            # Refill with this candidate plus the next most promising ones, so the block
            # serves several consecutive picks. Which extras are prefetched cannot affect the
            # result -- `held` is only a cache, and every `mind` update uses the exact column of
            # the record actually picked -- so an ordinary argsort is fine here, with no
            # tie-breaking obligation.
            ranked = np.argsort(np.where(taken, -np.inf, mind), kind="stable")[::-1]
            wanted = [candidate]
            for i in ranked[:batch]:
                if len(wanted) >= batch:
                    break
                if int(i) != candidate and not taken[i]:
                    wanted.append(int(i))
            wanted = sorted(wanted)
            block = columns(wanted)
            held = {j: block[:, r] for r, j in enumerate(wanted)}
        np.minimum(mind, held.pop(candidate), out=mind)
        picked.append(candidate)
        taken[candidate] = True
    return picked


def kennard_stone(D: np.ndarray, n_picks: int) -> list[int]:
    """Kennard-Stone selection, seeded with the maximally distant pair.

    Ties go to the lexicographically smallest pair.

    :param D: a dense symmetric distance matrix.
    :param n_picks: how many points to select.
    :return: the picked indices, in selection order.
    """
    n = D.shape[0]
    if n < 2:
        return list(range(n))[:n_picks]
    # Upper triangle, scanned row-major-first: the same smallest-maximal-pair tie-break as the
    # Python scan, without two n(n-1)/2 index arrays.
    upper = np.triu(D, k=1)
    i0, j0 = first_argmax_2d(upper)
    picked = [i0, j0]
    mind = np.minimum(D[:, i0], D[:, j0])
    taken = np.zeros(n, dtype=bool)
    taken[[i0, j0]] = True
    while len(picked) < n_picks:
        if bool(taken.all()):
            break
        next_i = masked_argmax(mind, taken)
        picked.append(next_i)
        taken[next_i] = True
        np.minimum(mind, D[:, next_i], out=mind)
    return picked


def optisim_pick(
    D: np.ndarray,
    n_picks: int,
    subsample_size: int,
    radius: float,
    rng: np.random.Generator,
) -> list[int]:
    """OptiSim diversity selection (Clark 1997).

    Starts from one random record. Each round draws candidates without replacement until
    ``subsample_size`` of them lie further than ``radius`` from everything selected, then takes
    the one whose minimum distance to the selection is largest, ties to the smallest index. The
    rest of the subsample goes to a recycle bin that refills the pool when it empties. A
    candidate inside ``radius`` is dropped for good, since the selection only ever grows, which
    is what guarantees termination.

    ``subsample_size=1`` is random selection with sphere exclusion; a subsample covering every
    record is MaxMin with a random first pick.

    :param D: a dense symmetric distance matrix.
    :param n_picks: how many points to select.
    :param subsample_size: candidates that must clear ``radius`` before one is selected.
    :param radius: the exclusion radius.
    :param rng: generator for the candidate draws.
    :return: the picked indices, fewer than ``n_picks`` if no candidate outside ``radius``
        remains.
    """
    return optisim_pick_columns(D.shape[0], lambda j: D[:, j], n_picks, subsample_size, radius, rng)


def optisim_pick_columns(
    n: int,
    column: Callable[[int], np.ndarray],
    n_picks: int,
    subsample_size: int,
    radius: float,
    rng: np.random.Generator,
) -> list[int]:
    """Run :func:`optisim_pick` against a distance matrix supplied column by column.

    Only the selected records' columns are ever requested, so no full ``n x n`` matrix is
    needed.

    :param n: number of records.
    :param column: maps a record index to that record's distances to every record.
    :param n_picks: how many points to select.
    :param subsample_size: candidates that must clear ``radius`` before one is selected.
    :param radius: the exclusion radius.
    :param rng: generator for the candidate draws.
    :return: the picked indices, in selection order.
    """
    if n == 0 or n_picks <= 0:
        return []
    first = int(rng.integers(0, n))
    picked = [first]
    mind = np.asarray(column(first), dtype=np.float64)
    pool = rng.permutation(np.flatnonzero(mind > radius + EPS)).tolist()
    recycle: list[int] = []
    while len(picked) < n_picks:
        subsample: list[int] = []
        while len(subsample) < subsample_size:
            if not pool:
                if not recycle:
                    break
                pool = rng.permutation(np.asarray(recycle, dtype=np.int64)).tolist()
                recycle = []
            candidate = pool.pop()
            if mind[candidate] > radius + EPS:
                subsample.append(candidate)
        if not subsample:
            break
        best = argmax_tiebreak(lambda idx: mind[idx], sorted(subsample))
        picked.append(best)
        mind = np.minimum(mind, np.asarray(column(best), dtype=np.float64))
        recycle.extend(i for i in subsample if i != best)
    return picked


def _farthest_pair(D: np.ndarray, pool: np.ndarray) -> tuple[int, int]:
    """Farthest-apart pair among ``pool`` (ascending record indices); ties -> lexicographically
    smallest ``(i, j)``."""
    sub = D[np.ix_(pool, pool)]
    iu = np.triu_indices(pool.size, k=1)
    vals = sub[iu]
    top = vals.max()
    k = int(np.flatnonzero(vals >= top - EPS)[0])  # row-major order = lexicographic (i, j)
    return int(pool[iu[0][k]]), int(pool[iu[1][k]])


def duplex_order(D: np.ndarray, targets: Sequence[int]) -> list[list[int]]:
    """DUPLEX partitioning (Snee 1977), generalised to any number of partitions.

    Each partition with a non-zero target is seeded, in order, with the farthest-apart pair of
    still-unassigned records, or just the pair's first record when its target is 1. The
    partitions below target then take turns adding the unassigned record whose minimum distance
    to their own members is largest, ties to the smallest index, and drop out of the rotation
    once they are full.

    :param D: a dense symmetric distance matrix.
    :param targets: the record count per partition. Must sum to ``len(D)``.
    :return: each partition's records, in the order they were added.
    """
    n = D.shape[0]
    if sum(targets) != n:
        raise ValueError(f"targets sum to {sum(targets)}, expected n={n}")
    assigned = np.zeros(n, dtype=bool)
    members: list[list[int]] = [[] for _ in targets]
    mind = [np.full(n, np.inf) for _ in targets]

    def add(p: int, i: int) -> None:
        members[p].append(i)
        assigned[i] = True
        mind[p] = np.minimum(mind[p], D[:, i])

    for p, target in enumerate(targets):
        if target <= 0:
            continue
        pool = np.flatnonzero(~assigned)
        if pool.size == 1:
            add(p, int(pool[0]))
            continue
        i, j = _farthest_pair(D, pool)
        add(p, i)
        if target > 1:
            add(p, j)
    while not assigned.all():
        progressed = False
        for p, target in enumerate(targets):
            if len(members[p]) >= target or assigned.all():
                continue
            score = np.where(assigned, -np.inf, mind[p])
            add(p, int(row_argmin(-score[None, :])[0]))
            progressed = True
        if not progressed:
            break
    return members


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
    """Laplacian-eigenmap spectral partition.

    Isolated vertices, meaning a zero row-sum in ``W``, are kept out of the eigenproblem and
    reattached afterwards as singleton clusters.

    :param W: the affinity matrix, dense or sparse.
    :param n_clusters: how many clusters to cut the embedding into.
    :param laplacian: symmetric, random-walk, or unnormalized.
    :param drop_first: drop the trivial leading eigenvector.
    :param assign: cluster the embedding with k-means, or with the discretize rule.
    :param rng: unused; ``random_state`` seeds k-means.
    :param random_state: seed passed to k-means.
    :return: a dense-label-encoded int array of length ``n``.
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
        # Scaling by a diagonal matrix is a broadcast, not a matmul: the matmul form was O(n^3)
        # and allocated extra n x n matrices. Bit-identical -- the dropped terms are exact 0.0 * x.
        if laplacian == "unnormalized":
            L = -Wa.copy()
            L[np.diag_indices(len(active))] += dega
        elif laplacian == "rw":
            L = -(1.0 / dega)[:, None] * Wa
            L[np.diag_indices(len(active))] += 1.0
        else:  # "sym"
            dinv_sqrt = 1.0 / np.sqrt(dega)
            L = -(dinv_sqrt[:, None] * Wa * dinv_sqrt[None,:])
            L[np.diag_indices(len(active))] += 1.0

        k = min(n_clusters + (1 if drop_first else 0), len(active) - 1)
        k = max(k, 1)

        if rng is None:
            rng = np.random.default_rng(random_state)
        v0 = rng.standard_normal(L.shape[0])

        if len(active) <= k + 1 or len(active) < 50:
            # small enough for a dense eigensolve, which also avoids ARPACK trouble on tiny
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

    # encode by first appearance, so ids stay contiguous and order-stable
    seen: dict[int, int] = {}
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        lbl = int(labels[i])
        if lbl not in seen:
            seen[lbl] = len(seen)
        out[i] = seen[lbl]
    return out
