"""The ``task``-defined splitter family.

These splitters encode a specific downstream deployment question (hit identification, lead
optimisation, scaffold hopping, cold-start drug/target/pair prediction, virtual-screening bias,
decoy-benchmark construction) rather than a generic structural criterion.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import numpy as np
from rdkit import Chem

from chemsplit import scaffolds as _scaffolds
from chemsplit._fp_similarity import (
    blocked_max_similarity_to,
    blocked_threshold_counts,
    blocked_threshold_pairs,
    compute_distance_matrix,
    compute_neighbor_lists,
    compute_similarity_matrix,
    dense_matrix_fits,
    guard_memory,
)
from chemsplit._ga import mutate_bits
from chemsplit._optimize import BalanceProblem, solve_balance
from chemsplit._pair_assign import assign_pair_groups
from chemsplit._unionfind import UnionFind, dense_label_encode
from chemsplit.base import (
    BaseSplitter,
    GroupSplitter,
    SplitResult,
    Strictness,
    _Context,
    assign_groups,
)
from chemsplit.clustering import EPS as _clustering_EPS
from chemsplit.clustering import butina, butina_from_neighbors
from chemsplit.determinism import argmax_tiebreak, seed_for, seeded_python_random, stable_sort
from chemsplit.exceptions import (
    ConfigurationError,
    ConstraintUnsatisfiableError,
    DegenerateGroupingError,
    EmptyPartitionError,
    HomologyLeakWarning,
    InvariantError,
    LabelError,
    ParameterError,
    SizeToleranceWarning,
    SmallPartitionWarning,
    warn_with_details,
)
from chemsplit.featurizers import get_featurizer
from chemsplit.types import IndexArray

__all__ = [
    "HiSplitter",
    "LoSplitter",
    "ScaffoldHopSplitter",
    "ColdDrugSplitter",
    "ColdTargetSplitter",
    "ColdPairSplitter",
    "AVESplitter",
    "DecoyBenchmarkSplitter",
]

#: Tolerance for float32 similarity-matrix comparisons (~1.2e-7 rounding noise).
_EPS = 1e-6


class HiSplitter(GroupSplitter):
    """Hit-identification split: no test molecule may exceed ``threshold`` similarity to any
    training molecule; sizes are optimised under that hard constraint.

    :param threshold: the similarity ceiling between any test and any training molecule.
    :param coarse_cutoff: similarity at which records are pre-grouped into conflict components,
        which sets the optimisation's granularity without changing the guarantee.
    :param solver: greedy assignment, an exact ILP, or simulated annealing.
    :param max_discard_frac: largest fraction of records that may be discarded to meet the
        constraint.
    :param time_limit_s: wall-clock limit on the ILP solve.
    :param annealing_steps: steps taken by ``solver="annealing"``.
    :param annealing_t0: starting temperature for annealing.
    :param annealing_t1: final temperature for annealing.
    :param verify: recheck the realised maximum cross-similarity against ``threshold`` after
        solving, which is the only proof the split is correct.
    :param featurizer: featurizer alias or instance used to build the similarity matrix.
    :param metric: similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``threshold`` or ``coarse_cutoff`` is outside ``(0, 1)``,
        ``max_discard_frac`` is outside ``[0, 1)``, or ``solver`` is unknown.
    :raises ConstraintUnsatisfiableError: at split time, if the threshold cannot be met within
        ``max_discard_frac``.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Gives a **verified guarantee**: `metadata["max_cross_similarity"]` is checked against
      the threshold, so a hit-identification benchmark's central claim is evidence.
    - Formulating the assignment over conflict components, instead of pruning greedily, usually
      costs no data, unlike `similarity_threshold`'s `greedy_prune`.
    - Reproduces a widely reported result: models that look strong under scaffold splitting
      often fall toward random here.
    - Three solvers trade optimality for speed, and all three are deterministic.

    Pitfalls
    --------
    - On congeneric or focused datasets the conflict graph collapses into one component and the
      requested split is impossible. It raises rather than returning a leaky split.
    - The threshold, the fingerprint and its radius jointly define difficulty, so "Tanimoto
      0.4" alone is not a full design.
    - `coarse_cutoff` changes the optimisation granularity, and therefore the achievable
      balance, without changing the guarantee.
    - The guarantee is one-sided, test to train, so the *train* set may still hold
      near-duplicates of itself. This controls generalisation, not training redundancy.
    - Near-zero cross-similarity does not make the test molecules drug-like or interesting: an
      extreme split can be dominated by fragments and outliers.
    - `verify=False` removes the only proof the split is correct.

    Notes
    -----
    Independent, from-scratch optimisation engine (:mod:`chemsplit._optimize`). ``solver`` values
    ``"greedy"``/``"ilp"``/``"annealing"`` select among chemsplit's own benchmarked architectures
    (``greedy``, ``milp`` via ``scipy.optimize.milp``, ``anneal``).

    References
    ----------
    .. [1] Steshin, S. Lo-Hi: Practical ML Drug Discovery Benchmark. In *Advances in Neural
       Information Processing Systems 36* (Datasets and Benchmarks Track), **2023**;
       pp 64526-64554. https://doi.org/10.52202/075280-2816 (preprint:
       https://arxiv.org/abs/2310.06399)
    """

    splitter_id: ClassVar[str] = "hi"
    family: ClassVar[str] = "task"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        threshold: float = 0.4,
        coarse_cutoff: float = 0.6,
        solver: Literal["greedy", "ilp", "annealing"] = "greedy",
        max_discard_frac: float = 0.2,
        time_limit_s: float = 300.0,
        annealing_steps: int = 20_000,
        annealing_t0: float = 1.0,
        annealing_t1: float = 0.01,
        verify: bool = True,
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.coarse_cutoff = coarse_cutoff
        self.solver = solver
        self.max_discard_frac = max_discard_frac
        self.time_limit_s = time_limit_s
        self.annealing_steps = annealing_steps
        self.annealing_t0 = annealing_t0
        self.annealing_t1 = annealing_t1
        self.verify = verify
        self.featurizer = featurizer
        self.metric = metric
        self.max_memory_bytes = max_memory_bytes
        if not (0.0 < threshold < 1.0):
            raise ParameterError(f"threshold must satisfy 0 < t < 1, got {threshold!r}")
        if coarse_cutoff < threshold:
            raise ParameterError(
                f"coarse_cutoff ({coarse_cutoff}) must be >= threshold ({threshold}): coarse "
                "clusters finer than the constraint would make the conflict-component "
                "construction meaningless"
            )
        if solver not in ("greedy", "ilp", "annealing"):
            raise ParameterError(f"invalid solver: {solver!r}")
        if not (0.0 <= max_discard_frac < 1.0):
            raise ParameterError(
                f"max_discard_frac must satisfy 0 <= f < 1, got {max_discard_frac!r}"
            )

    def _conflict_components(
        self, ctx: _Context
    ) -> tuple[list[int], dict[int, list[int]], int, np.ndarray]:
        feat = get_featurizer(self.featurizer)
        dense = dense_matrix_fits(ctx.n, self.max_memory_bytes)
        coarse = 1.0 - self.coarse_cutoff
        if dense:
            D = compute_distance_matrix(
                ctx, feat, self.metric, self.max_memory_bytes, type(self).__name__, self.n_jobs
            )
            S: np.ndarray | None = 1.0 - D
            clusters = butina(D, cutoff=coarse, reorder=False)
        else:
            S = None
            neigh = compute_neighbor_lists(
                ctx, feat, self.metric, coarse, eps=_clustering_EPS, n_jobs=self.n_jobs
            )
            clusters = butina_from_neighbors(neigh, ctx.n, reorder=False)
        n_clusters = len(clusters)

        # Two clusters conflict exactly when some record pair across them exceeds the threshold,
        # so the record-level threshold graph settles every cluster pair in one pass. That
        # replaces an O(k^2) scan that extracted a submatrix per cluster pair -- with k near n on
        # a diverse library, that was the dominant cost even when the matrix fitted.
        cluster_of = np.empty(ctx.n, dtype=np.int64)
        for cid, members_ in enumerate(clusters):
            cluster_of[members_] = cid
        uf = UnionFind(n_clusters)
        if S is not None:
            pair_iter = (
                (i, np.nonzero(S[i][i + 1:] > self.threshold + _EPS)[0] + i + 1, None)
                for i in range(ctx.n)
            )
        else:
            pair_iter = blocked_threshold_pairs(
                ctx, feat, self.metric, self.threshold, eps=_EPS, n_jobs=self.n_jobs
            )
        for i, js, _vals in pair_iter:
            a = int(cluster_of[i])
            for j in js:
                b = int(cluster_of[j])
                if a != b:
                    uf.union(a, b)
        comps = uf.components()
        comp_records: dict[int, list[int]] = {}
        for rep, cluster_ids in comps.items():
            recs: list[int] = []
            for cid in cluster_ids:
                recs.extend(clusters[cid])
            comp_records[rep] = sorted(recs)
        comp_order = stable_sort(
            sorted(comp_records.keys()), key=lambda r: min(comp_records[r]), desc=False
        )
        return comp_order, comp_records, n_clusters, S

    def _group_labels(self, ctx: _Context) -> IndexArray:
        comp_order, comp_records, _n_clusters, _S = self._conflict_components(ctx)
        labels = np.empty(ctx.n, dtype=np.int64)
        for new_id, r in enumerate(comp_order):
            for rec in comp_records[r]:
                labels[rec] = new_id
        return labels

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        comp_order, comp_records, n_clusters, S = self._conflict_components(ctx)
        n_components = len(comp_order)
        component_sizes = [len(comp_records[r]) for r in comp_order]

        bucket_names: list[str] = []
        bucket_targets: list[float] = []
        for name, target in (
            ("train", ctx.sizes.n_train),
            ("valid", ctx.sizes.n_valid),
            ("test", ctx.sizes.n_test),
        ):
            if target > 0:
                bucket_names.append(name)
                bucket_targets.append(float(target))

        problem = BalanceProblem(
            n_items=n_components,
            n_buckets=len(bucket_names),
            item_size=np.asarray(component_sizes, dtype=np.int64),
            bucket_target_size=np.asarray(bucket_targets, dtype=np.float64),
            size_tolerance=self.size_tolerance,
        )
        architecture = {"greedy": "greedy", "ilp": "milp", "annealing": "anneal"}[self.solver]
        purpose = "hi.anneal" if self.solver == "annealing" else "hi.solver"
        rng = seed_for(ctx.rng_seeds, purpose, 0)
        solution = solve_balance(
            problem,
            rng=rng,
            time_limit_s=self.time_limit_s,
            architecture=architecture,
            anneal_steps=self.annealing_steps,
            anneal_t0=self.annealing_t0,
            anneal_t1=self.annealing_t1,
        )

        buckets: dict[str, list[int]] = {name: [] for name in bucket_names}
        for ci, r in enumerate(comp_order):
            buckets[bucket_names[int(solution.assignment[ci])]].extend(comp_records[r])

        train = sorted(buckets.get("train", []))
        valid = sorted(buckets.get("valid", []))
        test = sorted(buckets.get("test", []))
        discard: list[int] = []

        target_test = ctx.sizes.n_test
        deviation = abs(len(test) - target_test) / max(1, ctx.n)
        max_discard = int(self.max_discard_frac * ctx.n)
        cross_counts: dict[int, int] | None = None
        if deviation > self.size_tolerance and self.max_discard_frac > 0 and train and test:
            train_set = set(train)
            test_list = list(test)
            while (
                abs(len(test_list) - target_test) / max(1, ctx.n) > self.size_tolerance
                and len(discard) < max_discard
                and test_list
            ):
                # `train_set` is unchanged by this loop, so each count is constant: compute the
                # table once instead of rebuilding it every iteration.
                if cross_counts is None:
                    train_idx = sorted(train_set)
                    if S is not None:
                        cross_counts = {
                            t: int(np.sum(S[t, train_idx] > self.threshold + _EPS))
                            for t in test_list
                        }
                    else:
                        cross_counts = blocked_threshold_counts(
                            ctx, get_featurizer(self.featurizer), self.metric, self.threshold,
                            rows=test_list, cols=train_idx, eps=_EPS, n_jobs=self.n_jobs,
                        )
                if max(cross_counts.values(), default=0) == 0:
                    break
                worst = argmax_tiebreak(lambda t: cross_counts[t], sorted(test_list))
                test_list.remove(worst)
                discard.append(worst)
            test = sorted(test_list)

        if not test and target_test > 0:
            largest = max(component_sizes) if component_sizes else 0
            achievable = 1.0 - largest / max(1, ctx.n)
            raise ConstraintUnsatisfiableError(
                f"{type(self).__name__}: could not reach a non-empty test set at threshold="
                f"{self.threshold} within max_discard_frac={self.max_discard_frac} (largest "
                f"conflict component has {largest} of {ctx.n} records; best achievable non-"
                f"component test fraction ~{achievable:.3f}). Try a higher threshold."
            )

        train_arr = np.asarray(sorted(train), dtype=np.int64)
        valid_arr = np.asarray(sorted(valid), dtype=np.int64)
        test_arr = np.asarray(sorted(test), dtype=np.int64)
        discard_arr = np.asarray(sorted(discard), dtype=np.int64)

        max_cross_similarity = 0.0
        cross_violates = False
        if train_arr.size and test_arr.size:
            # same dtype and EPS as the stage-2 conflict check, so verification can't misfire
            if S is not None:
                cross_sub = S[np.ix_(test_arr, train_arr)]
                cross_violates = bool(np.any(cross_sub > self.threshold + _EPS))
                max_cross_similarity = float(cross_sub.max())
            else:
                max_sim = blocked_max_similarity_to(
                    ctx, get_featurizer(self.featurizer), self.metric, train_arr,
                    n_jobs=self.n_jobs,
                )[test_arr]
                cross_violates = bool(np.any(max_sim > self.threshold + _EPS))
                max_cross_similarity = float(max_sim.max())

        if self.verify and cross_violates:
            raise InvariantError(
                f"{type(self).__name__}: post-repair verification failed -- max cross-similarity "
                f"{max_cross_similarity} exceeds threshold {self.threshold}",
                splitter_id=self.splitter_id,
                params=self.get_params(),
                n_records=ctx.n,
            )

        groups_full = np.empty(ctx.n, dtype=np.int64)
        for new_id, r in enumerate(comp_order):
            for rec in comp_records[r]:
                groups_full[rec] = new_id

        extra_metadata = {
            "threshold": self.threshold,
            "coarse_cutoff": self.coarse_cutoff,
            "n_clusters": n_clusters,
            "n_components": n_components,
            "component_sizes": component_sizes,
            "max_cross_similarity": max_cross_similarity,
            "n_discarded": int(discard_arr.size),
            "solver": self.solver,
            "solver_status": solution.solver_status,
        }
        result = self._build_result(
            ctx,
            train=train_arr,
            valid=valid_arr,
            test=test_arr,
            discard=discard_arr,
            groups=groups_full,
            extra_metadata=extra_metadata,
        )
        self._check_size_tolerance(result, ctx)
        return [result]


class LoSplitter(GroupSplitter):
    """Lead-optimisation split: build clusters of mutually similar molecules that nevertheless
    span a range of activity, hold each cluster out whole, and evaluate within-cluster ranking.

    :param threshold: how similar molecules must be to join one cluster.
    :param min_cluster_size: smallest cluster that may be held out.
    :param max_clusters: cap on the number of held-out clusters.
    :param std_threshold: minimum label standard deviation within a cluster, in the label's own
        units, so that the cluster spans a real activity range.
    :param train_similarity_ceiling: prune training molecules above this similarity to any test
        molecule, or ``None`` to reuse ``threshold``.
    :param task_index: column of a multi-task ``y`` to use.
    :param evaluation: recorded in metadata to say whether scores are meant per cluster or
        pooled. It does not change the partition.
    :param featurizer: featurizer alias or instance used to build the similarity matrix.
    :param metric: similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix. Exceeding it raises rather than
        allocating.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if a similarity is outside ``(0, 1)``, a size parameter is below
        1, ``std_threshold`` is negative, or ``evaluation`` is unknown.
    :raises LabelError: at split time, if ``y`` is missing or ``task_index`` is out of range.
    :raises ConstraintUnsatisfiableError: at split time, if no cluster meets the size and
        spread criteria.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Answers the question a medicinal chemist actually asks -- which analogue should I make
      next? -- rather than the question most benchmarks answer.
    - Each held-out cluster is internally similar yet spans real activity, so a model that
      only separates coarse chemotypes scores at chance -- invisible to any cluster split.
    - `cluster_members` supports the metric this split calls for: Spearman correlation
      **within** each cluster, averaged across clusters, with pooled metrics secondary.
    - Train pruning removes the near-neighbour leak that would otherwise make within-cluster
      ranking trivial.

    Pitfalls
    --------
    - **The evaluation metric is part of the split.** Pooled R² or ROC-AUC discards the point
      of it; the per-cluster ranking over `metadata["cluster_members"]` is the result.
    - `std_threshold` is in the label's own units and assumes a log scale, so it is meaningless
      on a linear IC50 column.
    - Test-set size follows the cluster criteria rather than `test_size`, which acts as a cap.
    - Train pruning removes a large fraction of the data, and the most informative
      near-analogues. Intentional, and it makes absolute scores incomparable elsewhere.
    - Clusters are built greedily around high-degree centres, so early clusters absorb the
      densest regions and later ones are progressively weaker.
    - Activity spread within a cluster can be assay noise rather than SAR, so replicates are
      worth aggregating and single-assay data worth preferring.

    References
    ----------
    .. [1] Steshin, S. Lo-Hi: Practical ML Drug Discovery Benchmark. In *Advances in Neural
       Information Processing Systems 36* (Datasets and Benchmarks Track), **2023**;
       pp 64526-64554. https://doi.org/10.52202/075280-2816 (preprint:
       https://arxiv.org/abs/2310.06399)
    """

    splitter_id: ClassVar[str] = "lo"
    family: ClassVar[str] = "task"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        threshold: float = 0.4,
        min_cluster_size: int = 5,
        max_clusters: int = 50,
        std_threshold: float = 0.60,
        train_similarity_ceiling: float | None = None,
        task_index: int = 0,
        evaluation: Literal["per_cluster_rank", "pooled"] = "per_cluster_rank",
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.std_threshold = std_threshold
        self.train_similarity_ceiling = train_similarity_ceiling
        self.task_index = task_index
        self.evaluation = evaluation
        self.featurizer = featurizer
        self.metric = metric
        self.max_memory_bytes = max_memory_bytes
        if not (0.0 < threshold < 1.0):
            raise ParameterError(f"threshold must satisfy 0 < t < 1, got {threshold!r}")
        if min_cluster_size < 3:
            raise ParameterError(f"min_cluster_size must be >= 3, got {min_cluster_size!r}")
        if max_clusters < 1:
            raise ParameterError(f"max_clusters must be >= 1, got {max_clusters!r}")
        if std_threshold <= 0:
            raise ParameterError(f"std_threshold must be > 0, got {std_threshold!r}")
        if train_similarity_ceiling is not None and not (0.0 < train_similarity_ceiling <= 1.0):
            raise ParameterError(
                f"train_similarity_ceiling must be None or 0 < c <= 1, got "
                f"{train_similarity_ceiling!r}"
            )

    def _group_labels(self, ctx: _Context) -> IndexArray:
        # _partition is overridden below, so this exists only for compute_groups: one group
        # per threshold-neighbourhood cluster, singletons for the rest
        return self._build(ctx)[0]

    def _build(
        self, ctx: _Context
    ) -> tuple[IndexArray, list[list[int]], set[int], np.ndarray, float]:
        y = np.asarray(ctx.y, dtype=np.float64)
        y1 = y[:, self.task_index] if y.ndim == 2 else y
        feat = get_featurizer(self.featurizer)
        # Only the similarity-threshold neighbour sets are read from the matrix, so build them
        # blockwise when a dense one will not fit. Identical sets either way.
        if dense_matrix_fits(ctx.n, self.max_memory_bytes):
            S = compute_similarity_matrix(
                ctx, feat, self.metric, self.max_memory_bytes, type(self).__name__, self.n_jobs
            )
            neigh = [
                set(np.nonzero(S[i] >= self.threshold - _EPS)[0].tolist()) - {i}
                for i in range(ctx.n)
            ]
        else:
            S = None
            neigh = [
                set(a.tolist())
                for a in compute_neighbor_lists(
                    ctx,
                    feat,
                    self.metric,
                    self.threshold,
                    eps=_EPS,
                    mode="similarity_ge",
                    n_jobs=self.n_jobs,
                )
            ]

        available = set(range(ctx.n))
        clusters: list[list[int]] = []
        best_std = 0.0
        while len(clusters) < self.max_clusters:
            cand = [i for i in available if len(neigh[i] & available) + 1 >= self.min_cluster_size]
            if not cand:
                break
            placed = False
            for i in stable_sort(cand, key=lambda i: len(neigh[i] & available), desc=True):
                members = sorted([i] + list(neigh[i] & available))
                std = float(np.std(y1[members]))
                best_std = max(best_std, std)
                if std >= self.std_threshold:
                    clusters.append(members)
                    available -= set(members)
                    placed = True
                    break
            if not placed:
                break

        raw_keys = [f"__lo_singleton_{i}__" for i in range(ctx.n)]
        for cidx, members in enumerate(clusters):
            for m in members:
                raw_keys[m] = f"__lo_cluster_{cidx}__"
        labels = dense_label_encode(raw_keys)
        return labels, clusters, available, S, best_std

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        labels, clusters, available, S, best_std = self._build(ctx)
        if not clusters:
            raise ConstraintUnsatisfiableError(
                f"{type(self).__name__}: no cluster satisfied min_cluster_size="
                f"{self.min_cluster_size} and std_threshold={self.std_threshold}; best observed "
                f"std among size-qualifying clusters = {best_std:.4f}"
            )

        test_set = {i for members in clusters for i in members}
        if self.train_similarity_ceiling is not None:
            ceiling = self.train_similarity_ceiling
        else:
            ceiling = self.threshold
        test_list = sorted(test_set)
        leak: set[int] = set()
        if S is not None:
            for r in sorted(available):
                if S[r, test_list].max() > ceiling + _EPS:
                    leak.add(r)
        else:
            # the same reduction as S[r, test_list].max(), blockwise
            max_sim = blocked_max_similarity_to(
                ctx, get_featurizer(self.featurizer), self.metric, test_list, n_jobs=self.n_jobs
            )
            for r in sorted(available):
                if max_sim[r] > ceiling + _EPS:
                    leak.add(r)
        discard_set = leak
        train_set = available - leak

        if not train_set and ctx.sizes.n_train > 0:
            raise EmptyPartitionError(
                f"{type(self).__name__}: similarity pruning removed the entire training pool"
            )

        train_list = sorted(train_set)
        valid_list: list[int] = []
        if ctx.sizes.n_valid > 0 and train_list:
            rng = seed_for(ctx.rng_seeds, "lo.valid", 0)
            perm = [train_list[i] for i in rng.permutation(len(train_list))]
            n_valid = min(ctx.sizes.n_valid, len(perm))
            valid_list = sorted(perm[:n_valid])
            train_list = sorted(perm[n_valid:])

        train_arr = np.asarray(train_list, dtype=np.int64)
        valid_arr = np.asarray(valid_list, dtype=np.int64)
        test_arr = np.asarray(test_list, dtype=np.int64)
        discard_arr = np.asarray(sorted(discard_set), dtype=np.int64)

        extra_metadata = {
            "n_clusters": len(clusters),
            "cluster_members": clusters,
            "cluster_stds": [float(np.std(np.asarray(ctx.y)[m])) for m in clusters],
            "cluster_sizes": [len(m) for m in clusters],
            "n_pruned_from_train": int(len(discard_set)),
            "y_std_overall": float(np.std(np.asarray(ctx.y, dtype=np.float64))),
            "threshold": self.threshold,
            "std_threshold": self.std_threshold,
        }
        result = self._build_result(
            ctx,
            train=train_arr,
            valid=valid_arr,
            test=test_arr,
            discard=discard_arr,
            groups=labels,
            extra_metadata=extra_metadata,
        )
        if test_arr.size < ctx.sizes.n_test:
            warn_with_details(
                SizeToleranceWarning(
                    f"{type(self).__name__}: cluster criteria yielded {test_arr.size} test "
                    f"records, below the {ctx.sizes.n_test} target -- test_size is a ceiling here, "
                    "not a target",
                    details={"realised_test": int(test_arr.size), "target_test": ctx.sizes.n_test},
                )
            )
        return [result]


class ScaffoldHopSplitter(GroupSplitter):
    """Test set restricted to actives whose scaffolds are absent from train, while pharmacophoric
    features are retained.

    :param scaffold_kind: which scaffold must differ: the Murcko framework, its generic form,
        or the ring systems.
    :param active_definition: read ``y`` as a binary active flag, or threshold it.
    :param active_threshold: the potency cut for ``active_definition="threshold"``.
    :param pharmacophore_similarity: how pharmacophoric similarity is measured: not at all,
        with FCFP fingerprints, or with RDKit 2-D pharmacophore fingerprints.
    :param min_pharm_similarity: how similar a test active must stay to some training active.
    :param inactives_policy: put every inactive in train, split them along with the scaffold
        grouping, or discard them.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``min_pharm_similarity`` is outside ``[0, 1]``, or a mode
        parameter is unknown.
    :raises ConfigurationError: if ``active_definition="threshold"`` without
        ``active_threshold``.
    :raises LabelError: at split time, if ``y`` is missing.
    :raises ConstraintUnsatisfiableError: at split time, if too few actives or distinct
        scaffolds survive the criteria.

    Advantages
    ----------
    - The right evaluation for a virtual-screening novelty claim: a new framework with the
      recognition features still findable, which is what "scaffold hop" means.
    - The scaffold-disjointness constraint is asserted rather than assumed.
    - Keeping the scaffold criterion separate from the pharmacophore criterion makes both
      reportable and tunable.

    Pitfalls
    --------
    - Two coupled thresholds, `scaffold_kind` and `min_pharm_similarity`, define the
      difficulty, neither has a principled default, and the result moves a lot with both.
    - 2-D pharmacophore similarity is a weak proxy for 3-D recognition: a pair passing
      `min_pharm_similarity` may bind differently, and a real hop may fail it.
    - Needs enough actives spread over enough distinct scaffolds. Most single-target datasets
      lack them, so the splitter raises instead of producing a two-scaffold benchmark.
    - Inactives dominate screening data by record count, so `inactives_policy` controls the
      class balance of both partitions, and with it the headline metric.
    - Success here is evidence about ranking novel chemotypes, not about absolute potency
      prediction.

    References
    ----------
    .. [1] This composes a scaffold-disjointness constraint with a pharmacophore-similarity
       floor, rather than implementing a published splitting method. The concept and its
       topological-pharmacophore similarity: Schneider, G.; Neidhart, W.; Giller, T.;
       Schmid, G. "Scaffold-Hopping" by Topological Pharmacophore Search: A Contribution to
       Virtual Screening. *Angew. Chem. Int. Ed.* **1999**, 38 (19), 2894-2896.
       https://doi.org/10.1002/(SICI)1521-3773(19991004)38:19<2894::AID-ANIE2894>3.0.CO;2-F
    .. [2] Hu, Y.; Stumpfe, D.; Bajorath, J. Recent Advances in Scaffold Hopping.
       *J. Med. Chem.* **2017**, 60 (4), 1238-1246.
       https://doi.org/10.1021/acs.jmedchem.6b01437
    """

    splitter_id: ClassVar[str] = "scaffold_hop"
    family: ClassVar[str] = "task"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        scaffold_kind: Literal["murcko", "generic", "ring_system"] = "generic",
        active_definition: Literal["binary", "threshold"] = "binary",
        active_threshold: float | None = None,
        pharmacophore_similarity: Literal["none", "fcfp", "pharm2d"] = "fcfp",
        min_pharm_similarity: float = 0.5,
        inactives_policy: Literal[
            "train", "split_with_scaffolds", "discard"
        ] = "split_with_scaffolds",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.scaffold_kind = scaffold_kind
        self.active_definition = active_definition
        self.active_threshold = active_threshold
        self.pharmacophore_similarity = pharmacophore_similarity
        self.min_pharm_similarity = min_pharm_similarity
        self.inactives_policy = inactives_policy
        if scaffold_kind not in ("murcko", "generic", "ring_system"):
            raise ParameterError(f"invalid scaffold_kind: {scaffold_kind!r}")
        if active_definition == "threshold" and active_threshold is None:
            raise ParameterError("active_threshold is required when active_definition='threshold'")
        if pharmacophore_similarity not in ("none", "fcfp", "pharm2d"):
            raise ParameterError(f"invalid pharmacophore_similarity: {pharmacophore_similarity!r}")
        if inactives_policy not in ("train", "split_with_scaffolds", "discard"):
            raise ParameterError(f"invalid inactives_policy: {inactives_policy!r}")

    def _scaffold_key(self, mol: Any) -> str:
        if mol is None:
            return ""
        if self.scaffold_kind == "murcko":
            return _scaffolds.murcko_scaffold(mol)
        if self.scaffold_kind == "generic":
            return _scaffolds.generic_scaffold(mol)
        return "|".join(sorted(_scaffolds.ring_systems(mol)))

    def _group_labels(self, ctx: _Context) -> IndexArray:
        mols = ctx.mols
        if mols is None:
            raise ParameterError(f"{type(self).__name__} requires molecule input")
        return dense_label_encode([self._scaffold_key(m) for m in mols])

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        mols = ctx.mols
        if mols is None:
            raise ParameterError(f"{type(self).__name__} requires molecule input")
        y = np.asarray(ctx.y, dtype=np.float64)
        if self.active_definition == "binary":
            active_mask = y == 1
        else:
            active_mask = y >= self.active_threshold
        A = np.nonzero(active_mask)[0]
        inactive_idx = np.nonzero(~active_mask)[0]
        scaff = [self._scaffold_key(m) for m in mols]

        by_scaffold: dict[str, list[int]] = {}
        for i in A:
            by_scaffold.setdefault(scaff[i], []).append(int(i))
        if len(by_scaffold) < 2:
            raise DegenerateGroupingError(
                f"{type(self).__name__}: only {len(by_scaffold)} distinct active scaffold(s) -- "
                "need at least 2 to hold any out"
            )
        cand_scaffolds = sorted(
            by_scaffold.keys(),
            key=lambda s: (-len(by_scaffold[s]), min(by_scaffold[s])),
        )

        P = None
        if self.pharmacophore_similarity != "none":
            feat_name = "fcfp4"  # pharm2d not available without extra tooling; documented fallback
            feat = get_featurizer(feat_name)
            guard_memory(len(A), 2 * 1024**3, type(self).__name__)
            a_mols = [mols[i] for i in A]

            class _ActivesCtx:
                def __init__(self, mols_: list[Chem.rdchem.Mol]) -> None:
                    self.mols = mols_
                    self.n = len(mols_)
                    self._cache: dict[Any, Any] = {}

                def get_features(self, featurizer: Any = None) -> Any:
                    key = featurizer.name
                    if key not in self._cache:
                        self._cache[key] = featurizer.transform(self.mols)
                    return self._cache[key]

            P = compute_similarity_matrix(
                _ActivesCtx(a_mols), feat, "tanimoto", 2 * 1024**3, type(self).__name__, self.n_jobs
            )
        a_index = {int(a): pos for pos, a in enumerate(A.tolist())}

        test_scaffolds: list[str] = []
        best_pharm = 0.0
        n_test_target = ctx.sizes.n_test
        held_actives = 0
        for s in cand_scaffolds:
            if held_actives >= n_test_target > 0:
                break
            trial_test = by_scaffold[s]
            already_held = {i for ts in test_scaffolds for i in by_scaffold[ts]}
            trial_train_actives = [
                i for i in A.tolist() if i not in already_held and i not in trial_test
            ]
            if P is not None and trial_train_actives:
                train_pos = [a_index[i] for i in trial_train_actives]
                ok = True
                for t in trial_test:
                    sims = P[a_index[t], train_pos]
                    m = float(sims.max()) if sims.size else 0.0
                    best_pharm = max(best_pharm, m)
                    if m < self.min_pharm_similarity - _EPS:
                        ok = False
                if not ok:
                    continue
            elif not trial_train_actives:
                # keep at least one active scaffold in train
                continue
            test_scaffolds.append(s)
            held_actives += len(trial_test)

        if not test_scaffolds:
            raise ConstraintUnsatisfiableError(
                f"{type(self).__name__}: no scaffold could be held out at "
                f"min_pharm_similarity={self.min_pharm_similarity} (best achieved among "
                f"candidates: {best_pharm:.4f})"
            )

        test_set = {i for s in test_scaffolds for i in by_scaffold[s]}
        train_actives = set(A.tolist()) - test_set

        inactive_set = set(inactive_idx.tolist())
        discard_set: set[int] = set()
        if self.inactives_policy == "train":
            train_inactives = inactive_set
            test_inactives: set[int] = set()
        elif self.inactives_policy == "discard":
            train_inactives = set()
            test_inactives = set()
            discard_set |= inactive_set
        else:  # split_with_scaffolds
            test_scaffold_set = set(test_scaffolds)
            train_inactives = {i for i in inactive_set if scaff[i] not in test_scaffold_set}
            test_inactives = inactive_set - train_inactives

        train = sorted(train_actives | train_inactives)
        test = sorted(test_set | test_inactives)
        discard_arr_list = sorted(discard_set)

        overlap = {scaff[i] for i in train} & {scaff[i] for i in test}
        if overlap:
            raise InvariantError(
                f"{type(self).__name__}: {len(overlap)} scaffold(s) present on both sides after "
                "assignment",
                splitter_id=self.splitter_id,
                params=self.get_params(),
                n_records=ctx.n,
            )

        train_arr = np.asarray(train, dtype=np.int64)
        test_arr = np.asarray(test, dtype=np.int64)
        discard_arr = np.asarray(discard_arr_list, dtype=np.int64)
        valid_arr = np.array([], dtype=np.int64)

        groups = dense_label_encode(scaff)
        extra_metadata = {
            "n_test_scaffolds": len(test_scaffolds),
            "test_scaffolds": test_scaffolds,
            "n_active_test": int(len(test_set)),
            "n_active_train": int(len(train_actives)),
            "min_pharm_similarity_achieved": best_pharm,
            "scaffold_kind": self.scaffold_kind,
        }
        result = self._build_result(
            ctx,
            train=train_arr,
            valid=valid_arr,
            test=test_arr,
            discard=discard_arr,
            groups=groups,
            extra_metadata=extra_metadata,
        )
        return [result]




class _ColdStartBase(GroupSplitter):
    """Shared implementation for the three cold-start interaction splitters: holds out whole
    compound groups, whole target groups, or both (``axis``, fixed per concrete subclass).

    ``X`` must be a ``Sequence[tuple[compound_key, target_key]]``. A grouper may be a
    :class:`~chemsplit.base.GroupSplitter` instance or a registry id; ``None`` makes each
    compound or target its own group.
    """

    _axis: ClassVar[str] = "compound"
    accepts: ClassVar[tuple[str,...]] = ("interactions",)

    def __init__(
        self,
        *,
        compound_grouper: Any = None,
        target_grouper: Any = None,
        compound_structures: Any = None,
        target_sequences: Any = None,
        min_interactions_per_entity: int = 1,
        drop_unlabelled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.compound_grouper = compound_grouper
        self.target_grouper = target_grouper
        self.compound_structures = compound_structures
        self.target_sequences = target_sequences
        self.min_interactions_per_entity = min_interactions_per_entity
        self.drop_unlabelled = drop_unlabelled
        if compound_grouper is not None and compound_structures is None:
            raise ConfigurationError("compound_grouper requires compound_structures")
        if target_grouper is not None and target_sequences is None:
            raise ConfigurationError("target_grouper requires target_sequences")
        if min_interactions_per_entity < 1:
            raise ParameterError(
                f"min_interactions_per_entity must be >= 1, got {min_interactions_per_entity!r}"
            )

    def _entities(self, ctx: _Context) -> tuple[list, list, np.ndarray, np.ndarray]:
        raw = ctx.extra.get("raw_interactions")
        if raw is None:
            raise ParameterError(
                f"{type(self).__name__} requires interaction-pair input "
                "(Sequence[tuple[compound_key, target_key]])"
            )
        compounds: list = []
        targets: list = []
        c_seen: dict[Any, int] = {}
        t_seen: dict[Any, int] = {}
        compound_idx = np.empty(len(raw), dtype=np.int64)
        target_idx = np.empty(len(raw), dtype=np.int64)
        for i, (c, t) in enumerate(raw):
            if c not in c_seen:
                c_seen[c] = len(compounds)
                compounds.append(c)
            if t not in t_seen:
                t_seen[t] = len(targets)
                targets.append(t)
            compound_idx[i] = c_seen[c]
            target_idx[i] = t_seen[t]
        return compounds, targets, compound_idx, target_idx

    def _entity_groups(self, entities: list, structures: Any, grouper: Any) -> np.ndarray:
        if grouper is None or structures is None:
            return np.arange(len(entities), dtype=np.int64)
        seqs = [structures[e] for e in entities]
        return np.asarray(grouper.compute_groups(seqs), dtype=np.int64)

    def _group_labels(self, ctx: _Context) -> IndexArray:
        compounds, targets, compound_idx, target_idx = self._entities(ctx)
        cg = self._entity_groups(compounds, self.compound_structures, self.compound_grouper)
        tg = self._entity_groups(targets, self.target_sequences, self.target_grouper)
        if self._axis == "compound":
            return dense_label_encode(cg[compound_idx].tolist())
        if self._axis == "target":
            return dense_label_encode(tg[target_idx].tolist())
        # pair: no single label array captures a two-axis intersection, so report the
        # compound axis for introspection and let _partition do the real work
        return dense_label_encode(cg[compound_idx].tolist())

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        compounds, targets, compound_idx, target_idx = self._entities(ctx)
        cg = self._entity_groups(compounds, self.compound_structures, self.compound_grouper)
        tg = self._entity_groups(targets, self.target_sequences, self.target_grouper)

        if self._axis == "target" and self.target_grouper is None:
            warn_with_details(
                HomologyLeakWarning(
                    f"{type(self).__name__}: target_grouper=None -- homologous targets will leak "
                    "across the train/test boundary unless a sequence-identity grouper (e.g. "
                    "sequence_identity) is supplied",
                    details={"splitter_id": self.splitter_id},
                )
            )

        keep_mask = np.ones(ctx.n, dtype=bool)
        if self.drop_unlabelled and ctx.y is not None:
            y_arr = np.asarray(ctx.y, dtype=object)
            for i in range(ctx.n):
                v = y_arr[i]
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    keep_mask[i] = False

        c_counts = np.bincount(compound_idx, minlength=len(compounds))
        t_counts = np.bincount(target_idx, minlength=len(targets))
        for i in range(ctx.n):
            if not keep_mask[i]:
                continue
            if c_counts[compound_idx[i]] < self.min_interactions_per_entity:
                keep_mask[i] = False
            elif t_counts[target_idx[i]] < self.min_interactions_per_entity:
                keep_mask[i] = False

        forced_discard = set(int(i) for i in np.nonzero(~keep_mask)[0])
        keep_idx = np.nonzero(keep_mask)[0]

        rng = self._rng_for_group_assignment(ctx)
        if self._axis == "compound":
            labels = cg[compound_idx[keep_idx]]
            labels_dense = dense_label_encode(labels.tolist())
            if len(set(labels_dense.tolist())) < 2:
                raise DegenerateGroupingError(
                    f"{type(self).__name__}: fewer than 2 compound groups after filtering"
                )
            buckets_local = assign_groups(labels_dense, ctx.sizes, self.group_assignment, rng)
            train = keep_idx[buckets_local["train"]]
            valid = keep_idx[buckets_local.get("valid", np.array([], dtype=np.int64))]
            test = keep_idx[buckets_local["test"]]
        elif self._axis == "target":
            labels = tg[target_idx[keep_idx]]
            labels_dense = dense_label_encode(labels.tolist())
            if len(set(labels_dense.tolist())) < 2:
                raise DegenerateGroupingError(
                    f"{type(self).__name__}: fewer than 2 target groups after filtering"
                )
            buckets_local = assign_groups(labels_dense, ctx.sizes, self.group_assignment, rng)
            train = keep_idx[buckets_local["train"]]
            valid = keep_idx[buckets_local.get("valid", np.array([], dtype=np.int64))]
            test = keep_idx[buckets_local["test"]]
        else:  # pair
            labels_a = dense_label_encode(cg[compound_idx[keep_idx]].tolist())
            labels_b = dense_label_encode(tg[target_idx[keep_idx]].tolist())
            if len(set(labels_a.tolist())) < 2 or len(set(labels_b.tolist())) < 2:
                raise DegenerateGroupingError(
                    f"{type(self).__name__}: fewer than 2 groups on the compound or target axis "
                    "after filtering"
                )
            pair_result = assign_pair_groups(
                labels_a, labels_b, ctx.sizes, rng, mode="both_novel",
                group_assignment=self.group_assignment,
            )
            train = keep_idx[pair_result["train"]]
            valid = keep_idx[pair_result["valid"]]
            test = keep_idx[pair_result["test"]]
            forced_discard |= set(int(keep_idx[i]) for i in pair_result["discard"])

        train_set = set(train.tolist())
        train_compounds = set(compound_idx[list(train_set)].tolist()) if train_set else set()
        train_targets = set(target_idx[list(train_set)].tolist()) if train_set else set()
        all_compounds = set(range(len(compounds)))
        all_targets = set(range(len(targets)))
        targets_lost = sorted(all_targets - train_targets)
        compounds_lost = sorted(all_compounds - train_compounds)

        discard_arr = np.asarray(sorted(forced_discard), dtype=np.int64)
        train_arr = np.asarray(sorted(train.tolist()), dtype=np.int64)
        valid_arr = np.asarray(sorted(valid.tolist()), dtype=np.int64)
        test_arr = np.asarray(sorted(test.tolist()), dtype=np.int64)

        discard_frac = discard_arr.size / max(1, ctx.n)
        if self._axis == "pair" and discard_frac > 0.8:
            warn_with_details(
                SmallPartitionWarning(
                    f"{type(self).__name__}: {discard_frac:.1%} of interactions discarded by the "
                    "pair cold-start construction",
                    details={"discard_frac": discard_frac},
                )
            )

        groups_full = self._group_labels(ctx)
        extra_metadata: dict[str, Any] = {
            "axis": self._axis,
            "n_compounds": len(compounds),
            "n_targets": len(targets),
            "n_compound_groups": int(len(set(cg.tolist()))),
            "n_target_groups": int(len(set(tg.tolist()))),
            "targets_lost_from_train": [targets[i] for i in targets_lost],
            "compounds_lost_from_train": [compounds[i] for i in compounds_lost],
            "records_discarded": int(discard_arr.size),
        }
        if self._axis == "pair":
            f_test = ctx.sizes.n_test / max(1, ctx.n)
            extra_metadata["sqrt_fraction_used"] = math.sqrt(max(0.0, f_test))
            extra_metadata["discard_frac"] = discard_frac

        result = self._build_result(
            ctx,
            train=train_arr,
            valid=valid_arr,
            test=test_arr,
            discard=discard_arr,
            groups=groups_full,
            extra_metadata=extra_metadata,
        )
        self._check_size_tolerance(result, ctx)
        return [result]


class ColdDrugSplitter(_ColdStartBase):
    """Cold-start on the compound axis: held-out compounds are unseen (and, with
    ``compound_grouper``, structurally novel) against a fixed, familiar panel of targets.

    :param compound_grouper: a group-forming :class:`~chemsplit.base.GroupSplitter`, or a
        registry id, grouping the compound axis so held-out compounds are structurally novel
        and not merely unseen keys. ``None`` makes each compound its own group.
    :param target_grouper: the same for the target axis, e.g. a sequence-identity grouping.
    :param compound_structures: SMILES per compound key, needed when ``compound_grouper``
        works on structures.
    :param target_sequences: sequence per target key, needed when ``target_grouper`` works on
        sequences.
    :param min_interactions_per_entity: drop compounds or targets with fewer interactions than
        this before splitting.
    :param drop_unlabelled: discard interaction records whose label is missing.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if a grouper is neither ``None``, a string nor a
        :class:`GroupSplitter`, or ``min_interactions_per_entity`` is below 1.
    :raises InputKindError: at split time, if ``X`` is not a sequence of
        ``(compound_key, target_key)`` tuples.
    :raises InputError: at split time, if ``compound_structures`` or ``target_sequences`` is
        needed but missing or incomplete.

    Advantages
    ----------
    - Matches the most common deployment question for a DTI model: here is a new compound,
      which of my known targets does it hit?
    - `compound_grouper="butina"` upgrades it from unseen compound to unseen chemotype, a much
      stronger claim for no extra effort.
    - Usually keeps every target represented in training, so per-target metrics stay
      computable.

    Pitfalls
    --------
    - The weakest of the three cold-start settings, routinely reported as the strongest: a
      model can score well on target-level marginals alone -- "kinase X is promiscuous".
    - Without `compound_grouper`, held-out compounds are merely *unseen*, not *novel*: an
      analogue of a training compound is a different key and the same chemistry.
    - Interaction matrices are very unbalanced, so record-count and entity-count sizing
      disagree sharply. The splitter sizes by records and reports both.
    - Targets can vanish from training entirely, hiding a cold-target evaluation inside a
      cold-drug split. `targets_lost_from_train` counts them.

    References
    ----------
    .. [1] Park, Y.; Marcotte, E. M. Flaws in Evaluation Schemes for Pair-Input Computational
       Predictions. *Nat. Methods* **2012**, 9 (12), 1134-1136.
       https://doi.org/10.1038/nmeth.2259
    .. [2] Pahikkala, T.; Airola, A.; Pietilä, S. et al. Toward More Realistic Drug-Target
       Interaction Predictions. *Brief. Bioinform.* **2015**, 16 (2), 325-337.
       https://doi.org/10.1093/bib/bbu010
    """

    splitter_id: ClassVar[str] = "cold_drug"
    family: ClassVar[str] = "task"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = False
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False
    _axis: ClassVar[str] = "compound"


class ColdTargetSplitter(_ColdStartBase):
    """Cold-start on the target axis: held-out targets are unseen against a fixed, familiar panel
    of compounds.

    :param compound_grouper: a group-forming :class:`~chemsplit.base.GroupSplitter`, or a
        registry id, grouping the compound axis so held-out compounds are structurally novel
        and not merely unseen keys. ``None`` makes each compound its own group.
    :param target_grouper: the same for the target axis, e.g. a sequence-identity grouping.
    :param compound_structures: SMILES per compound key, needed when ``compound_grouper``
        works on structures.
    :param target_sequences: sequence per target key, needed when ``target_grouper`` works on
        sequences.
    :param min_interactions_per_entity: drop compounds or targets with fewer interactions than
        this before splitting.
    :param drop_unlabelled: discard interaction records whose label is missing.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if a grouper is neither ``None``, a string nor a
        :class:`GroupSplitter`, or ``min_interactions_per_entity`` is below 1.
    :raises InputKindError: at split time, if ``X`` is not a sequence of
        ``(compound_key, target_key)`` tuples.
    :raises InputError: at split time, if ``compound_structures`` or ``target_sequences`` is
        needed but missing or incomplete.

    Advantages
    ----------
    - The zero-shot-target setting, and the only evaluation supporting a claim of predicting
      activity at a target with no training data.
    - With `target_grouper="sequence_identity"`, the identity ceiling between train and test
      targets is measured and reported.

    Pitfalls
    --------
    - **Homologues leak by default.** Holding out a kinase while training on its 95%-identical
      paralogue is no cold-target experiment, hence the warning when no sequence-identity
      grouping is given.
    - Compounds are seen, so compound-level marginals -- "this compound is promiscuous" -- are
      exploitable in the same way `cold_drug` exploits target marginals.
    - Target counts run in the tens to hundreds, so holding out 20% leaves few test targets
      and enormous variance; `leave_one_cluster_out` gives leave-one-target-out.
    - Assay protocols differ per target, so a held-out target brings a protocol shift along
      with the biological one.

    References
    ----------
    .. [1] Park, Y.; Marcotte, E. M. Flaws in Evaluation Schemes for Pair-Input Computational
       Predictions. *Nat. Methods* **2012**, 9 (12), 1134-1136.
       https://doi.org/10.1038/nmeth.2259
    .. [2] Pahikkala, T.; Airola, A.; Pietilä, S. et al. Toward More Realistic Drug-Target
       Interaction Predictions. *Brief. Bioinform.* **2015**, 16 (2), 325-337.
       https://doi.org/10.1093/bib/bbu010
    """

    splitter_id: ClassVar[str] = "cold_target"
    family: ClassVar[str] = "task"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = False
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False
    _axis: ClassVar[str] = "target"


class ColdPairSplitter(_ColdStartBase):
    """Cold-start on both axes at once: neither the compound nor the target of a test interaction
    was seen in training. Test size is a product of both axes' held-
    out fractions, sized via :func:`chemsplit._pair_assign.assign_pair_groups`'s ``sqrt(f)``
    rule.

    :param compound_grouper: a group-forming :class:`~chemsplit.base.GroupSplitter`, or a
        registry id, grouping the compound axis so held-out compounds are structurally novel
        and not merely unseen keys. ``None`` makes each compound its own group.
    :param target_grouper: the same for the target axis, e.g. a sequence-identity grouping.
    :param compound_structures: SMILES per compound key, needed when ``compound_grouper``
        works on structures.
    :param target_sequences: sequence per target key, needed when ``target_grouper`` works on
        sequences.
    :param min_interactions_per_entity: drop compounds or targets with fewer interactions than
        this before splitting.
    :param drop_unlabelled: discard interaction records whose label is missing.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if a grouper is neither ``None``, a string nor a
        :class:`GroupSplitter`, or ``min_interactions_per_entity`` is below 1.
    :raises InputKindError: at split time, if ``X`` is not a sequence of
        ``(compound_key, target_key)`` tuples.
    :raises InputError: at split time, if ``compound_structures`` or ``target_sequences`` is
        needed but missing or incomplete.

    Advantages
    ----------
    - The only setting that measures genuine interaction learning rather than memorised row and
      column marginals, since neither the compound nor the target has been seen.
    - `records_discarded` states how much of an interaction matrix is usable for a double-blind
      evaluation, which is usually a surprising number.

    Pitfalls
    --------
    - Discards the two off-diagonal blocks, typically 50-90% of the data, so the surviving test
      block is small and noisy.
    - Both axes are split at once, so the realised test fraction is a product that rarely
      matches the request. The `sqrt(f)` rule approximates it and reports the deviation.
    - Performance here runs much lower than in `cold_drug` or `cold_target` and is not
      comparable to either, which is a common and serious error in DTI papers.
    - Small target counts can leave a handful of interactions in the test block, for which no
      metric is stable.

    References
    ----------
    .. [1] Park, Y.; Marcotte, E. M. Flaws in Evaluation Schemes for Pair-Input Computational
       Predictions. *Nat. Methods* **2012**, 9 (12), 1134-1136.
       https://doi.org/10.1038/nmeth.2259
       (regime C4: neither member of the test pair was seen in training)
    .. [2] Pahikkala, T.; Airola, A.; Pietilä, S. et al. Toward More Realistic Drug-Target
       Interaction Predictions. *Brief. Bioinform.* **2015**, 16 (2), 325-337.
       https://doi.org/10.1093/bib/bbu010
       (setting S4; also the source of the steep performance drop this splitter produces)
    """

    splitter_id: ClassVar[str] = "cold_pair"
    family: ClassVar[str] = "task"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = False
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False
    _axis: ClassVar[str] = "pair"




def _trapezoid_auc(values: list[float]) -> float:
    n_bins = len(values) - 1
    if n_bins <= 0:
        return 0.0
    return sum((values[k] + values[k + 1]) / 2.0 for k in range(n_bins)) / n_bins


class AVESplitter(BaseSplitter):
    """Asymmetric Validation Embedding: minimise the analogue bias that lets a nearest-neighbour
    baseline "win" a virtual screen.

    :param target_bias: the AVE bias value to optimise towards.
    :param tolerance: how close to ``target_bias`` counts as done. A large value effectively
        disables the GA and reports the initial bias only.
    :param n_bins: bins used in the nearest-neighbour distance histograms behind the bias.
    :param population_size: GA population size.
    :param n_generations: how many generations to run.
    :param crossover_prob: probability of crossing a selected pair.
    :param mutation_prob: probability of mutating an individual.
    :param mutation_indpb: per-gene flip probability within a mutated individual.
    :param init_splitter: splitter, by registry id or instance, producing the starting
        population.
    :param featurizer: featurizer alias or instance used to build the similarity matrix.
    :param metric: similarity metric; see :mod:`chemsplit.metrics`.
    :param max_memory_bytes: ceiling on the pairwise matrix.
    :param kwargs: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``target_bias``, ``tolerance`` or a GA parameter is out of
        range.
    :raises LabelError: at split time, if ``y`` is missing or not binary.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Removes the specific artefact -- actives clustered near actives -- that made many
      published virtual-screening benchmarks solvable by a 1-nearest-neighbour baseline.
    - The bias is one reportable number, computed identically before and after, which makes the
      debiasing auditable.
    - `ave_initial` on its own, with the GA disabled by a large `tolerance`, is a cheap audit of
      any existing benchmark.

    Pitfalls
    --------
    - **Aggressive debiasing over-corrects.** Driving AVE to 0 strips genuine signal with the
      artefact; `ave_initial` is always in `metadata` so debiased and raw sit side by side.
    - AVE depends on the fingerprint and metric, so a debiased split is debiased only with
      respect to that representation, and a model on different features may still see the bias.
    - By far the most expensive splitter here: hundreds of generations, each needing
      nearest-neighbour statistics.
    - Defined only for binary actives and inactives. No accepted continuous analogue exists.
    - Small active counts make AVE unstable, and the GA will optimise noise just as happily.
    - A near-zero AVE removes one known bias, not all of them.

    Notes
    -----
    The GA operators are hand-rolled on a seeded ``random.Random`` rather than taken from DEAP,
    which reads Python's global ``random`` module. Mutation is shared with ``simpd`` through
    :mod:`chemsplit._ga`; crossover, repair and selection differ and stay separate.

    References
    ----------
    .. [1] Wallach, I.; Heifets, A. Most Ligand-Based Classification Benchmarks Reward
       Memorization Rather than Generalization. *J. Chem. Inf. Model.* **2018**, 58 (5),
       916-932. https://doi.org/10.1021/acs.jcim.7b00403
    .. [2] Sieg, J.; Flachsenberg, F.; Rarey, M. In Need of Bias Control: Evaluating Chemical
       Data for Machine Learning in Structure-Based Virtual Screening. *J. Chem. Inf. Model.*
       **2019**, 59 (3), 947-961. https://doi.org/10.1021/acs.jcim.8b00712
    .. [3] Rohrer, S. G.; Baumann, K. Maximum Unbiased Validation (MUV) Data Sets for Virtual
       Screening Based on PubChem Bioactivity Data. *J. Chem. Inf. Model.* **2009**, 49 (2),
       169-184. https://doi.org/10.1021/ci8002649
    """

    splitter_id: ClassVar[str] = "ave"
    family: ClassVar[str] = "task"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    extras: ClassVar[tuple[str,...]] = ("ga",)
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        target_bias: float = 0.0,
        tolerance: float = 0.02,
        n_bins: int = 100,
        population_size: int = 200,
        n_generations: int = 150,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        mutation_indpb: float = 0.02,
        init_splitter: str | Any = "stratified_random",
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        max_memory_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target_bias = target_bias
        self.tolerance = tolerance
        self.n_bins = n_bins
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_indpb = mutation_indpb
        self.init_splitter = init_splitter
        self.featurizer = featurizer
        self.metric = metric
        self.max_memory_bytes = max_memory_bytes
        if n_bins < 1:
            raise ParameterError(f"n_bins must be >= 1, got {n_bins!r}")
        if population_size < 2:
            raise ParameterError(f"population_size must be >= 2, got {population_size!r}")
        if isinstance(init_splitter, str) and init_splitter != "stratified_random":
            raise ParameterError(
                f"init_splitter string values are only recognised for the default "
                f"('stratified_random'); pass a constructed BaseSplitter instance for "
                "anything else "
                f"(chemsplit.registry-based string lookup is not available yet), got "
                f"{init_splitter!r}"
            )

    def _ave(
        self,
        S: np.ndarray | None,
        y: np.ndarray,
        test_mask: np.ndarray,
        ctx: _Context | None = None,
    ) -> tuple[float, dict[str, float]]:
        train_mask = ~test_mask
        active = y == 1
        test_active = np.nonzero(test_mask & active)[0]
        test_inactive = np.nonzero(test_mask & ~active)[0]
        train_active = np.nonzero(train_mask & active)[0]
        train_inactive = np.nonzero(train_mask & ~active)[0]

        def nn_sim(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
            if query.size == 0 or ref.size == 0:
                return np.zeros(query.size)
            if S is not None:
                return S[np.ix_(query, ref)].max(axis=1)
            # a per-row maximum, so blocking the rows is exact
            return blocked_max_similarity_to(
                ctx, get_featurizer(self.featurizer), self.metric, ref,
                rows=query, n_jobs=self.n_jobs,
            )

        aa_nn = nn_sim(test_active, train_active)
        ai_nn = nn_sim(test_active, train_inactive)
        ii_nn = nn_sim(test_inactive, train_inactive)
        ia_nn = nn_sim(test_inactive, train_active)

        thresholds = [k / self.n_bins for k in range(self.n_bins + 1)]

        def curve(nn: np.ndarray) -> list[float]:
            if nn.size == 0:
                return [0.0 for _ in thresholds]
            return [float(np.mean(nn >= t - _EPS)) for t in thresholds]

        auc_aa = _trapezoid_auc(curve(aa_nn))
        auc_ai = _trapezoid_auc(curve(ai_nn))
        auc_ii = _trapezoid_auc(curve(ii_nn))
        auc_ia = _trapezoid_auc(curve(ia_nn))
        ave = (auc_aa - auc_ai) + (auc_ii - auc_ia)
        return ave, {"auc_AA": auc_aa, "auc_AI": auc_ai, "auc_II": auc_ii, "auc_IA": auc_ia}

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        y = np.asarray(ctx.y)
        uniq = set(np.unique(y).tolist())
        if not uniq <= {0, 1}:
            raise LabelError(
                f"{type(self).__name__} requires binary {{0,1}} labels, got {sorted(uniq)}"
            )
        y = y.astype(np.int64)

        # Each fitness evaluation reads four per-row maxima over a column subset, all of which
        # block exactly, so past the dense ceiling they are recomputed instead. Memory stops
        # being the limit here; the GA's evaluation count is.
        feat = get_featurizer(self.featurizer)
        S = None
        if dense_matrix_fits(ctx.n, self.max_memory_bytes):
            S = compute_similarity_matrix(
                ctx, feat, self.metric, self.max_memory_bytes, type(self).__name__,
                self.n_jobs,
            )

        n_test_target = ctx.sizes.n_test
        n_active = int(np.sum(y == 1))
        if n_active < 20 or (ctx.n - n_active) < 20:
            warn_with_details(
                SmallPartitionWarning(
                    f"{type(self).__name__}: only {n_active} active(s) -- AVE will be dominated by "
                    "a handful of nearest-neighbour relations and is noisy",
                    details={"n_active": n_active, "n_records": ctx.n},
                )
            )

        # A seeded permutation stratified by y, rather than calling init_splitter directly,
        # which would need raw X/y instead of ctx. The repair step preserves the active
        # fraction anyway, so any stratified start behaves the same.
        rng0 = seed_for(ctx.rng_seeds, "ave.init", 0)
        active_idx = np.nonzero(y == 1)[0]
        inactive_idx = np.nonzero(y == 0)[0]
        active_perm = active_idx[rng0.permutation(active_idx.size)]
        inactive_perm = inactive_idx[rng0.permutation(inactive_idx.size)]
        active_frac = active_idx.size / ctx.n
        n_test_active = int(round(n_test_target * active_frac))
        n_test_inactive = max(0, n_test_target - n_test_active)
        n_test_active = min(n_test_active, active_idx.size)
        n_test_inactive = min(n_test_inactive, inactive_idx.size)
        init_test = np.zeros(ctx.n, dtype=bool)
        init_test[active_perm[:n_test_active]] = True
        init_test[inactive_perm[:n_test_inactive]] = True

        ave_initial, _ = self._ave(S, y, init_test, ctx)

        best_mask = init_test
        best_ave = ave_initial
        converged = abs(ave_initial - self.target_bias) <= self.tolerance
        generations_run = 0

        if not converged:
            pyrng = seeded_python_random(ctx.rng_seeds, "ave.ga", 0)

            def fitness(mask: np.ndarray) -> float:
                ave, _ = self._ave(S, y, mask, ctx)
                return -((ave - self.target_bias) ** 2)

            def repair(mask: np.ndarray) -> np.ndarray:
                mask = mask.copy()
                targets = ((active_perm, n_test_active), (inactive_perm, n_test_inactive))
                for idx, target_n in targets:
                    current = mask[idx]
                    n_on = int(current.sum())
                    if n_on > target_n:
                        on_positions = idx[current]
                        turn_off = pyrng.sample(list(on_positions), n_on - target_n)
                        mask[turn_off] = False
                    elif n_on < target_n:
                        off_positions = idx[~current]
                        turn_on = pyrng.sample(list(off_positions), target_n - n_on)
                        mask[turn_on] = True
                return mask

            population = [best_mask.copy()]
            for _ in range(self.population_size - 1):
                m = best_mask.copy()
                flips = pyrng.sample(range(ctx.n), max(1, int(self.mutation_indpb * ctx.n)))
                for f in flips:
                    m[f] = not m[f]
                population.append(repair(m))

            fitnesses = [fitness(m) for m in population]
            for gen in range(self.n_generations):
                generations_run = gen + 1
                new_pop = []
                while len(new_pop) < self.population_size:
                    i1, i2 = pyrng.sample(range(len(population)), 2)
                    parent = population[i1] if fitnesses[i1] >= fitnesses[i2] else population[i2]
                    child = parent.copy()
                    if pyrng.random() < self.crossover_prob:
                        i3 = pyrng.randrange(len(population))
                        cut = pyrng.randrange(1, ctx.n) if ctx.n > 1 else 0
                        child[cut:] = population[i3][cut:]
                    if pyrng.random() < self.mutation_prob:
                        child = mutate_bits(child, self.mutation_indpb, pyrng)
                    new_pop.append(repair(child))
                population = new_pop
                fitnesses = [fitness(m) for m in population]
                gen_best = max(range(len(population)), key=lambda i: fitnesses[i])
                if fitnesses[gen_best] > -((best_ave - self.target_bias) ** 2):
                    best_mask = population[gen_best]
                    best_ave, _ = self._ave(S, y, best_mask, ctx)
                if abs(best_ave - self.target_bias) <= self.tolerance:
                    converged = True
                    break

        ave_final, aucs = self._ave(S, y, best_mask, ctx)
        train_idx = np.nonzero(~best_mask)[0].astype(np.int64)
        test_idx = np.nonzero(best_mask)[0].astype(np.int64)

        if not converged:
            warn_with_details(
                SizeToleranceWarning(
                    f"{type(self).__name__}: AVE target_bias={self.target_bias} not reached "
                    f"within tolerance after {generations_run} generation(s); achieved "
                    f"{ave_final:.4f}",
                    details={"target_bias": self.target_bias, "achieved": ave_final},
                )
            )

        metadata = {
            "ave_initial": ave_initial,
            "ave_final": ave_final,
            "target_bias": self.target_bias,
            "converged": bool(converged),
            "generations_run": int(generations_run),
            **aucs,
            "n_active_train": int(np.sum(y[train_idx] == 1)) if train_idx.size else 0,
            "n_active_test": int(np.sum(y[test_idx] == 1)) if test_idx.size else 0,
            "realised_sizes": {
                "train": int(train_idx.size),
                "valid": 0,
                "test": int(test_idx.size),
            },
        }
        return [
            SplitResult(
                train=np.sort(train_idx),
                valid=np.array([], dtype=np.int64),
                test=np.sort(test_idx),
                discard=np.array([], dtype=np.int64),
                groups=None,
                splitter_id=self.splitter_id,
                params=self.get_params(),
                n_records=ctx.n,
                metadata=metadata,
            )
        ]


class DecoyBenchmarkSplitter(GroupSplitter):
    """Builds property-matched decoy sets (or consumes a curated benchmark's own partition), and
    groups each active with its decoys so they never straddle the train/test boundary.

    :param scheme: match decoys on 2-D properties, draw them by spatial binning, or consume a
        curated benchmark's own assignment.
    :param decoy_ratio: how many decoys to pair with each active.
    :param match_properties: which descriptors ``scheme="property_matched"`` matches on.
    :param match_tolerance: per-property tolerance for that matching.
    :param topology_dissimilarity: minimum topological distance a decoy must keep from its
        active, so a matched decoy is not an analogue.
    :param decoy_pool: candidate decoys as SMILES, or ``None`` to draw them from ``X``.
    :param predefined_assignment: the active/decoy assignment for
        ``scheme="predefined"``.
    :param spatial_bins: bins per axis for ``scheme="spatial_random"``.
    :param kwargs: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``decoy_ratio`` or ``spatial_bins`` is below 1,
        ``topology_dissimilarity`` is outside ``[0, 1]``, the property and tolerance lists
        differ in length, or ``scheme`` is unknown.
    :raises ConfigurationError: if ``scheme="predefined"`` without ``predefined_assignment``.
    :raises LabelError: at split time, if ``y`` is missing or not binary.

    Advantages
    ----------
    - Property matching removes the trivial signal -- actives being heavier, greasier or more
      charged than library molecules -- that lets a model screen on weight alone.
    - Grouping each active with its own decoys prevents the subtler leak of an active in train
      and its property twin in test.
    - `mean_active_decoy_similarity` and the shortfall table make the construction auditable,
      which is what was missing from benchmarks later found to be biased.

    Pitfalls
    --------
    - **Property-matched decoys carry a bias of their own.** Matching 2-D properties while
      enforcing topological dissimilarity leaves a latent signature a deep model can learn,
      which is how several decoy benchmarks turned out solvable without binding signal.
      `AVESplitter` measures it.
    - Decoys are *presumed* inactive rather than measured inactive, and a few percent are
      usually real binders, which caps achievable precision.
    - The choice of decoy pool defines the benchmark, and there is no default to fall back on.
    - `decoy_ratio` fixes the class imbalance and with it the headline metric, so enrichment
      factors at 1% do not compare across ratios.
    - Shortfalls concentrate on the most unusual actives, so the effective ratio varies
      systematically across the active set.

    References
    ----------
    .. [1] Mysinger, M. M.; Carchia, M.; Irwin, J. J.; Shoichet, B. K. Directory of Useful
       Decoys, Enhanced (DUD-E): Better Ligands and Decoys for Better Benchmarking.
       *J. Med. Chem.* **2012**, 55 (14), 6582-6594. https://doi.org/10.1021/jm300687e
    .. [2] Huang, N.; Shoichet, B. K.; Irwin, J. J. Benchmarking Sets for Molecular Docking.
       *J. Med. Chem.* **2006**, 49 (23), 6789-6801. https://doi.org/10.1021/jm0608356
    .. [3] Chen, L.; Cruz, A.; Ramsey, S. et al. Hidden Bias in the DUD-E Dataset Leads to
       Misleading Performance of Deep Learning in Structure-Based Virtual Screening.
       *PLoS ONE* **2019**, 14 (8), e0220113. https://doi.org/10.1371/journal.pone.0220113
    """

    splitter_id: ClassVar[str] = "decoy_benchmark"
    family: ClassVar[str] = "task"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    _DEFAULT_PROPS = (
        "MolWt",
        "MolLogP",
        "NumRotatableBonds",
        "NumHDonors",
        "NumHAcceptors",
        "FormalCharge",
    )
    _DEFAULT_TOL = (25.0, 1.0, 1, 1, 1, 0)

    def __init__(
        self,
        *,
        scheme: Literal["property_matched", "spatial_random", "predefined"] = "property_matched",
        decoy_ratio: int = 50,
        match_properties: Any = _DEFAULT_PROPS,
        match_tolerance: Any = _DEFAULT_TOL,
        topology_dissimilarity: float = 0.25,
        decoy_pool: Any = None,
        predefined_assignment: Any = None,
        spatial_bins: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.scheme = scheme
        self.decoy_ratio = decoy_ratio
        self.match_properties = match_properties
        self.match_tolerance = match_tolerance
        self.topology_dissimilarity = topology_dissimilarity
        self.decoy_pool = decoy_pool
        self.predefined_assignment = predefined_assignment
        self.spatial_bins = spatial_bins
        if scheme not in ("property_matched", "spatial_random", "predefined"):
            raise ParameterError(f"invalid scheme: {scheme!r}")
        if scheme == "property_matched" and decoy_pool is None:
            raise ConfigurationError(
                "decoy_pool is required for scheme='property_matched' -- there is no default "
                "decoy library"
            )
        if scheme == "predefined" and predefined_assignment is None:
            raise ConfigurationError("predefined_assignment is required for scheme='predefined'")

    def _props(self, mol: Any) -> np.ndarray:
        from rdkit.Chem import Descriptors

        table = {
            "MolWt": Descriptors.MolWt,
            "MolLogP": Descriptors.MolLogP,
            "TPSA": Descriptors.TPSA,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
            "NumHDonors": Descriptors.NumHDonors,
            "NumHAcceptors": Descriptors.NumHAcceptors,
            "FormalCharge": lambda m: float(Chem.GetFormalCharge(m)),
            "RingCount": Descriptors.RingCount,
        }
        return np.asarray([table[p](mol) for p in self.match_properties], dtype=np.float64)

    def _group_labels(self, ctx: _Context) -> IndexArray:
        # _partition is overridden for property_matched and spatial_random; predefined is a
        # plain assignment, so every record is its own singleton
        return np.arange(ctx.n, dtype=np.int64)

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        if self.scheme == "predefined":
            return self._partition_predefined(ctx)
        mols = ctx.mols
        if mols is None:
            raise ParameterError(f"{type(self).__name__} requires molecule input")
        y = np.asarray(ctx.y)
        if set(np.unique(y).tolist()) <= {0, 1}:
            active_idx = np.nonzero(y == 1)[0]
        else:
            active_idx = np.arange(ctx.n)

        if self.scheme == "spatial_random":
            return self._partition_spatial(ctx, mols, active_idx)

        decoy_mols = []
        for s in self.decoy_pool:
            m = Chem.MolFromSmiles(s)
            decoy_mols.append(m)
        # the matrix below is the rectangular actives-by-decoys block, not a square over both


        active_props = np.asarray([self._props(mols[i]) for i in active_idx])
        missing = np.full(len(self.match_properties), np.inf)
        decoy_props = np.asarray(
            [self._props(m) if m is not None else missing for m in decoy_mols]
        )
        mu = active_props.mean(axis=0)
        sigma = active_props.std(axis=0)
        sigma[sigma == 0] = 1.0

        feat = get_featurizer("ecfp4")
        active_fp = feat.transform([mols[i] for i in active_idx])
        decoy_fp = feat.transform(
            [m if m is not None else Chem.MolFromSmiles("C") for m in decoy_mols]
        )
        from chemsplit.metrics import pairwise_distances

        # The loop below reads one active's row at a time, so rows are produced on demand when
        # the full actives-by-decoys block will not fit. Each row is a wide fetch, so this is no
        # slower per row than slicing a materialised block.
        dense_ad = dense_matrix_fits(len(active_idx), 2 * 1024**3, cols=len(decoy_mols))
        S_ad = (
            1.0 - pairwise_distances(active_fp, decoy_fp, metric="tanimoto")
            if dense_ad
            else None
        )

        def ad_row(pos: int) -> np.ndarray:
            if S_ad is not None:
                return S_ad[pos]
            return (
                1.0
                - pairwise_distances(
                    active_fp[pos : pos + 1], decoy_fp, metric="tanimoto", n_jobs=self.n_jobs
                )[0]
            )

        tol = np.asarray(self.match_tolerance, dtype=np.float64)
        used = np.zeros(len(decoy_mols), dtype=bool)
        active_decoys: dict[int, list[int]] = {}
        shortfalls: dict[int, int] = {}
        sims_used: list[float] = []
        for pos, a in enumerate(active_idx.tolist()):
            avail = np.nonzero(~used)[0]
            if avail.size == 0:
                active_decoys[a] = []
                shortfalls[a] = self.decoy_ratio
                continue
            diffs = np.abs(decoy_props[avail] - active_props[pos])
            prop_ok = np.all(diffs <= tol[None,:] + _EPS, axis=1)
            row = ad_row(pos)
            topo_ok = row[avail] < self.topology_dissimilarity - _EPS
            cand = avail[prop_ok & topo_ok]
            if cand.size < self.decoy_ratio:
                shortfalls[a] = self.decoy_ratio - int(cand.size)
                chosen = cand
            else:
                z = (decoy_props[cand] - mu) / sigma
                za = (active_props[pos] - mu) / sigma
                dist = np.linalg.norm(z - za[None,:], axis=1)
                order = np.lexsort((cand, dist))
                chosen = cand[order[: self.decoy_ratio]]
            used[chosen] = True
            active_decoys[a] = chosen.tolist()
            sims_used.extend(float(row[c]) for c in chosen)

        n_pool_used = int(used.sum())
        n_actives_kept = int(len(active_idx))
        raw_keys = ["__decoy_unassigned__"] * ctx.n
        # SplitResult indices must stay inside the original X, which the decoys never were,
        # so only the actives are split here; the decoy bookkeeping goes into metadata for the
        # caller to join on.
        for a in active_idx.tolist():
            raw_keys[a] = f"__decoy_group_{a}__"
        for i in range(ctx.n):
            if raw_keys[i] == "__decoy_unassigned__":
                raw_keys[i] = f"__decoy_singleton_{i}__"
        labels = dense_label_encode(raw_keys)

        rng = self._rng_for_group_assignment(ctx)
        buckets_local = assign_groups(labels, ctx.sizes, self.group_assignment, rng)
        extra_metadata = {
            "scheme": self.scheme,
            "n_actives": n_actives_kept,
            "n_decoys": n_pool_used,
            "realised_decoy_ratio": (n_pool_used / n_actives_kept) if n_actives_kept else 0.0,
            "decoy_shortfalls": {int(k): int(v) for k, v in shortfalls.items()},
            "property_match_stats": {"match_properties": list(self.match_properties)},
            "mean_active_decoy_similarity": float(np.mean(sims_used)) if sims_used else 0.0,
        }
        mean_ratio = extra_metadata["realised_decoy_ratio"]
        if mean_ratio < self.decoy_ratio / 2:
            warn_with_details(
                SizeToleranceWarning(
                    f"{type(self).__name__}: mean realised decoy ratio {mean_ratio:.1f} is below "
                    f"half the requested {self.decoy_ratio}",
                    details={"realised_decoy_ratio": mean_ratio, "requested": self.decoy_ratio},
                )
            )
        result = self._build_result(
            ctx,
            train=buckets_local["train"],
            valid=buckets_local.get("valid", np.array([], dtype=np.int64)),
            test=buckets_local["test"],
            discard=np.asarray(sorted(ctx.extra.get("forced_discard", [])), dtype=np.int64),
            groups=labels,
            extra_metadata=extra_metadata,
        )
        return [result]

    def _partition_spatial(
        self, ctx: _Context, mols: list[Chem.rdchem.Mol], active_idx: np.ndarray
    ) -> list[SplitResult]:
        # spatial_random bins actives in normalised (MolWt, MolLogP) space and hands whole
        # bins to buckets, so partitions cover comparable cells. Without a decoy_pool there is
        # nothing external to match against, so every record takes part as its own unit.
        from rdkit.Chem import Descriptors

        props = np.asarray([[Descriptors.MolWt(m), Descriptors.MolLogP(m)] for m in mols])
        lo, hi = props.min(axis=0), props.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        norm = (props - lo) / span
        bins = np.clip((norm * self.spatial_bins).astype(np.int64), 0, self.spatial_bins - 1)
        raw_keys = [f"__spatial_{b[0]}_{b[1]}__" for b in bins]
        labels = dense_label_encode(raw_keys)
        rng = seed_for(ctx.rng_seeds, "decoy.spatial", 0)
        buckets_local = assign_groups(labels, ctx.sizes, self.group_assignment, rng)
        result = self._build_result(
            ctx,
            train=buckets_local["train"],
            valid=buckets_local.get("valid", np.array([], dtype=np.int64)),
            test=buckets_local["test"],
            discard=np.asarray(sorted(ctx.extra.get("forced_discard", [])), dtype=np.int64),
            groups=labels,
            extra_metadata={
                "scheme": "spatial_random",
                "n_actives": int(len(active_idx)),
                "n_decoys": 0,
                "realised_decoy_ratio": 0.0,
                "decoy_shortfalls": {},
                "property_match_stats": {"spatial_bins": self.spatial_bins},
                "mean_active_decoy_similarity": 0.0,
            },
        )
        return [result]

    def _partition_predefined(self, ctx: _Context) -> list[SplitResult]:
        assignment = self.predefined_assignment
        if len(assignment) != ctx.n:
            raise ParameterError(
                f"predefined_assignment has length {len(assignment)}, expected {ctx.n}"
            )
        buckets: dict[str, list[int]] = {"train": [], "valid": [], "test": [], "discard": []}
        for i, label in enumerate(assignment):
            if label not in buckets:
                raise ParameterError(f"invalid predefined_assignment value: {label!r}")
            buckets[label].append(i)
        return [
            SplitResult(
                train=np.asarray(sorted(buckets["train"]), dtype=np.int64),
                valid=np.asarray(sorted(buckets["valid"]), dtype=np.int64),
                test=np.asarray(sorted(buckets["test"]), dtype=np.int64),
                discard=np.asarray(sorted(buckets["discard"]), dtype=np.int64),
                groups=None,
                splitter_id=self.splitter_id,
                params=self.get_params(),
                n_records=ctx.n,
                metadata={
                    "scheme": "predefined",
                    "n_actives": 0,
                    "n_decoys": 0,
                    "realised_decoy_ratio": 0.0,
                    "decoy_shortfalls": {},
                    "property_match_stats": {},
                    "mean_active_decoy_similarity": 0.0,
                    "realised_sizes": {
                        "train": len(buckets["train"]),
                        "valid": len(buckets["valid"]),
                        "test": len(buckets["test"]),
                    },
                },
            )
        ]
