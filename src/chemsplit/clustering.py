"""Clustering and diversity-selection primitives shared by several splitter families.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import scipy.sparse as sp

from chemsplit.determinism import (
    argmax_tiebreak,
    argmin_tiebreak,
    first_ge_2d,
    masked_argmax,
    row_argmin,
)

#: Tolerance for the float32 distance matrices used here (~1.2e-7 rounding noise).
EPS = 1e-6

__all__ = [
    "butina",
    "butina_from_neighbors",
    "dbscan_from_neighbors",
    "duplex_order",
    "duplex_order_access",
    "kennard_stone",
    "leader",
    "PickDetail",
    "maxmin_pick",
    "maxmin_pick_detail",
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


@dataclasses.dataclass(frozen=True, slots=True)
class PickDetail:
    """A greedy selection plus the diagnostics that fall out of it for free.

    ``selection_mind`` is each pick's distance to the nearest already-picked point. That sequence
    covers every unordered pair in the selection exactly once, so its minimum *is* the minimum
    pairwise distance -- no ``k x k`` submatrix needed (51 GB at n=100000). ``final_mind`` is every
    record's distance to its nearest pick, so its maximum is the worst-case coverage.
    """

    picked: list[int]
    selection_mind: list[float]
    final_mind: np.ndarray

    @property
    def min_pairwise(self) -> float:
        """Minimum distance between two picked records, ``inf`` for fewer than two picks."""
        return min(self.selection_mind) if self.selection_mind else float("inf")

    @property
    def coverage(self) -> float:
        """Maximum over all records of the distance to the nearest picked record."""
        return float(self.final_mind.max()) if self.picked else float("nan")


def maxmin_pick(
    D: np.ndarray,
    n_picks: int,
    init: Literal["random", "kennard_stone", "most_peripheral", "index_zero"] = "random",
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Greedy MaxMin diversity selection: just the picks.

    :param D: a dense symmetric distance matrix.
    :param n_picks: how many points to select.
    :param init: how the first point is chosen.
    :param rng: generator for ``init="random"``.
    :return: the picked indices, in selection order.
    """
    return maxmin_pick_detail(D, n_picks, init=init, rng=rng).picked


def dbscan_from_neighbors(
    neigh: Sequence[np.ndarray], min_samples: int, n: int
) -> np.ndarray:
    """DBSCAN labels from precomputed eps-neighbour lists.

    Reproduces scikit-learn's ``dbscan_inner`` exactly -- core points seeded in index order, LIFO
    expansion -- so the labels match ``DBSCAN(metric="precomputed")`` on the dense matrix,
    including which cluster an ambiguous border point lands in. That matters because the
    neighbour lists can be built blockwise, while scikit-learn's own *sparse* precomputed path is
    not equivalent to its dense one (it reports fewer neighbours for the same stored entries).

    :param neigh: per record, the indices within eps, itself included.
    :param min_samples: neighbour count, self included, that makes a record a core point.
    :param n: the record count.
    :return: cluster labels, ``-1`` for noise.
    """
    is_core = np.array([len(a) >= min_samples for a in neigh], dtype=bool)
    labels = np.full(n, -1, dtype=np.int64)
    label_num = 0
    for seed in range(n):
        if labels[seed] != -1 or not is_core[seed]:
            continue
        stack = [seed]
        while stack:
            i = stack.pop()
            if labels[i] == -1:
                labels[i] = label_num
                if is_core[i]:
                    for j in neigh[i]:
                        if labels[j] == -1:
                            stack.append(int(j))
        label_num += 1
    return labels


def maxmin_pick_detail(
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
    :return: the picks, plus the diagnostics that fall out of the selection for free.
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
    at_pick = [float(D[a, b]) for i, a in enumerate(picked) for b in picked[i + 1:]]

    while len(picked) < n_picks:
        if bool(taken.all()):
            break
        next_i = masked_argmax(mind, taken)
        at_pick.append(float(mind[next_i]))
        picked.append(next_i)
        taken[next_i] = True
        np.minimum(mind, D[:, next_i], out=mind)

    return PickDetail(picked=picked, selection_mind=at_pick, final_mind=mind)


_MAX_HELD_COLUMNS_FACTOR = 16
"""Retained prefetched columns, as a multiple of ``batch``: ~3 GB at n=100000. Affects speed and
memory only -- the cache cannot change a pick."""


def maxmin_pick_columns(
    n: int,
    columns: Callable[[Sequence[int]], np.ndarray],
    n_picks: int,
    initial: Sequence[int],
    batch: int = 512,
) -> PickDetail:
    """Greedy MaxMin from a column-block callable, without an ``n x n`` matrix.

    Exactly :func:`maxmin_pick`, reorganised so distances arrive in wide blocks. The running
    minimum stays **exact**, so the argmax is always the true next pick; batching only decides
    when distances are fetched. Single columns are bandwidth-bound, so fetching a block of
    candidates at once is nearly free.

    :param n: the record count.
    :param columns: given ascending indices ``js``, returns the ``(n, len(js))`` distance block.
    :param n_picks: how many points to select.
    :param initial: the already-chosen starting picks -- one index for MaxMin, the farthest pair
        for Kennard-Stone.
    :param batch: how many columns to fetch per block.
    :return: the picks, plus the diagnostics that fall out of the selection for free.
    """
    initial = list(initial)
    taken = np.zeros(n, dtype=bool)
    taken[initial] = True
    picked = list(initial)
    block0 = columns(sorted(initial))
    mind = block0.min(axis=1).astype(np.float64, copy=True)
    order0 = {j: r for r, j in enumerate(sorted(initial))}
    # pairs inside the initial set, so min_pairwise covers them as the dense path does
    at_pick: list[float] = [
        float(block0[a, order0[b]])
        for x, a in enumerate(initial)
        for b in initial[x + 1:]
    ]
    held: dict[int, np.ndarray] = {}

    while len(picked) < n_picks:
        if bool(taken.all()):
            break
        candidate = masked_argmax(mind, taken)
        if candidate not in held:
            # Keep what is already held: tie plateaus push the next pick outside any fixed
            # prefetch, and discarding unused columns made the cost super-quadratic (only ~52 of
            # 512 consumed at n=20000). The prefetch set cannot change a pick, so a plain argsort
            # needs no tie-breaking.
            for j in [j for j in held if taken[j]]:
                del held[j]
            ranked = np.argsort(np.where(taken, -np.inf, mind), kind="stable")[::-1]
            wanted = [candidate]
            for i in ranked:
                if len(wanted) >= batch:
                    break
                j = int(i)
                if j != candidate and not taken[j] and j not in held:
                    wanted.append(j)
            wanted = sorted(wanted)
            block = columns(wanted)
            # .copy() matters: a column slice is a view pinning the whole block alive, so
            # retaining views pinned every block ever fetched -- 12.4 GB at n=100000.
            held.update({j: block[:, r].copy() for r, j in enumerate(wanted)})
            cap = _MAX_HELD_COLUMNS_FACTOR * batch
            if len(held) > cap:
                # Evict the least promising columns, trimming to the cap: halving the cache on
                # every overflow churned enough to cost 8x at n=100000.
                keep = set(wanted)
                for i in ranked:
                    if len(keep) >= cap:
                        break
                    keep.add(int(i))
                held = {j: col for j, col in held.items() if j in keep}
        at_pick.append(float(mind[candidate]))
        np.minimum(mind, held.pop(candidate), out=mind)
        picked.append(candidate)
        taken[candidate] = True
    return PickDetail(picked=picked, selection_mind=at_pick, final_mind=mind)


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
    # Lexicographically smallest pair within EPS of the maximum -- not the first pair attaining
    # the maximum exactly, which is a different record when an earlier pair sits just below it.
    # Lower triangle and diagonal masked to -inf so they can never qualify.
    upper = np.where(np.triu(np.ones_like(D, dtype=bool), k=1), D, -np.inf)
    i0, j0 = first_ge_2d(upper, float(upper.max()) - EPS)
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
    return duplex_order_access(
        D.shape[0], targets, lambda j: D[:, j], lambda pool: _farthest_pair(D, pool)
    )


def duplex_order_access(
    n: int,
    targets: Sequence[int],
    column: Callable[[int], np.ndarray],
    farthest_pair: Callable[[np.ndarray], tuple[int, int]],
    columns: Callable[[Sequence[int]], np.ndarray] | None = None,
    batch: int = 512,
) -> list[list[int]]:
    """DUPLEX partitioning from a column callable instead of a dense matrix.

    One column per assigned record and one farthest-pair search per partition seed is all the
    algorithm reads, so both can be served blockwise. Identical partitions either way.

    :param n: the record count.
    :param targets: the record count per partition. Must sum to ``n``.
    :param column: maps a record index to its distances to every record.
    :param farthest_pair: maps an ascending pool of indices to its farthest-apart pair.
    :param columns: optional batch form of ``column``. DUPLEX assigns *every* record, so it needs
        n columns; fetching them one at a time is memory-bandwidth-bound, and prefetching the
        most likely next picks amortises that. Which extras are prefetched cannot change a
        result -- the cache only decides when a column is fetched.
    :param batch: columns per prefetch when ``columns`` is given.
    :return: each partition's records, in the order they were added.
    """
    if sum(targets) != n:
        raise ValueError(f"targets sum to {sum(targets)}, expected n={n}")
    assigned = np.zeros(n, dtype=bool)
    members: list[list[int]] = [[] for _ in targets]
    mind = [np.full(n, np.inf) for _ in targets]

    # One cache per partition: they pick by their own min-distance vector, so a shared cache has
    # each partition evicting the others' prefetches and almost every pick misses.
    held: list[dict[int, np.ndarray]] = [{} for _ in targets]

    def col_of(p: int, i: int) -> np.ndarray:
        if columns is None:
            return column(i)
        cached = held[p].pop(i, None)
        return cached if cached is not None else column(i)

    def prefetch(p: int, scores: np.ndarray) -> None:
        """Cache columns for the records this partition is most likely to pick next."""
        if columns is None:
            return
        mine = held[p]
        for j in [j for j in mine if assigned[j]]:
            del mine[j]
        if len(mine) >= batch // 2:
            return
        # argpartition, not a full sort: only the top-`batch` set matters, not its order, and the
        # prefetch set cannot change a pick
        live = np.where(assigned, -np.inf, scores)
        take = min(batch, int((~assigned).sum()))
        if take <= 0:
            return
        cand = np.argpartition(-live, take - 1)[:take]
        wanted = sorted(int(j) for j in cand if not assigned[j] and int(j) not in mine)
        if not wanted:
            return
        block = columns(wanted)
        mine.update({j: block[:, r].copy() for r, j in enumerate(wanted)})

    def add(p: int, i: int) -> None:
        members[p].append(i)
        assigned[i] = True
        mind[p] = np.minimum(mind[p], col_of(p, i))

    for p, target in enumerate(targets):
        if target <= 0:
            continue
        pool = np.flatnonzero(~assigned)
        if pool.size == 1:
            add(p, int(pool[0]))
            continue
        i, j = farthest_pair(pool)
        add(p, i)
        if target > 1:
            add(p, j)
    while not assigned.all():
        progressed = False
        for p, target in enumerate(targets):
            if len(members[p]) >= target or assigned.all():
                continue
            score = np.where(assigned, -np.inf, mind[p])
            prefetch(p, mind[p])
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
