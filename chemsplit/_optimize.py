"""Independent balanced multi-way partitioning engine.

Determinism contract: every backend is either structurally deterministic (no RNG) or takes an
explicit ``numpy.random.Generator`` seeded via :func:`chemsplit.determinism.seed_for` — never bare
``random``/``np.random`` module state. ``solver_status`` is one of ``"optimal"``,
``"time_limit_feasible"``, ``"infeasible"``; only branch-and-bound and MILP may report
``"optimal"`` (they carry an optimality certificate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "BalanceProblem",
    "BalanceSolution",
    "solve_balance",
]

ArchitectureName = Literal["bnb", "milp", "greedy", "local_search", "anneal"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BalanceProblem:
    """Assign ``n_items`` atomic units to ``n_buckets``, minimising weighted deviation from
    per-bucket (and, optionally, per-task-per-bucket) targets under a shared objective.

    :param n_items: Problem size.
    :param n_buckets: Problem size.
    :param item_size: Number of underlying records each item (cluster/component) represents,
        shape ``(n_items,)``.
    :param bucket_target_size: Target aggregate ``item_size`` per bucket (e.g. ``resolve_sizes``'
        train/valid/test targets, or per-fold targets for k-fold), shape ``(n_buckets,)``.
    :param size_tolerance: Recorded for callers' feasibility interpretation; not itself enforced
        as a hard constraint by the heuristic backends (they minimise the weighted deviation,
        they do not reject solutions outside tolerance — callers decide feasibility from the
        returned objective/realised sizes).
    :param item_task_counts: Per-item, per-task label counts (e.g. non-NaN label count within the
        item), shape ``(n_items, n_tasks)``. ``None`` when there is no task dimension (e.g. the
        hit-identification splitter's component-balance problem).
    :param item_task_actives: Per-item, per-task active/positive counts, shape
        ``(n_items, n_tasks)``. Only meaningful when ``item_task_counts`` is also given; ignored
        otherwise.
    :param task_weight: Per-task objective weight, shape ``(n_tasks,)``. Defaults to all-ones
        when omitted but ``item_task_counts`` is given.
    :param task_tolerance: Recorded for callers' feasibility interpretation, mirroring
        ``size_tolerance``.
    :param fixed_bucket: Items pre-pinned to a specific bucket (symmetry breaking / caller-imposed
        constraints). Every backend must honour this exactly.
    """

    n_items: int
    n_buckets: int
    item_size: np.ndarray
    bucket_target_size: np.ndarray
    size_tolerance: float = 0.05
    item_task_counts: np.ndarray | None = None
    item_task_actives: np.ndarray | None = None
    task_weight: np.ndarray | None = None
    task_tolerance: float | None = None
    fixed_bucket: dict[int, int] = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        if self.item_task_counts is None:
            return 0
        return int(self.item_task_counts.shape[1])

    def __post_init__(self) -> None:
        if self.item_size.shape != (self.n_items,):
            raise ValueError("item_size must have shape (n_items,)")
        if self.bucket_target_size.shape != (self.n_buckets,):
            raise ValueError("bucket_target_size must have shape (n_buckets,)")
        if self.item_task_counts is not None and self.item_task_counts.shape[0] != self.n_items:
            raise ValueError("item_task_counts must have n_items rows")
        if self.item_task_actives is not None and self.item_task_counts is None:
            raise ValueError("item_task_actives requires item_task_counts to also be given")


@dataclass(frozen=True, slots=True)
class BalanceSolution:
    """The result of solving a :class:`BalanceProblem`."""

    assignment: np.ndarray  # (n_items,) int64, bucket index per item
    objective: float
    solver_status: Literal["optimal", "time_limit_feasible", "infeasible"]
    architecture: str
    wall_time_s: float
    n_iterations: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared objective
# ---------------------------------------------------------------------------


def _bucket_sums(item_values: np.ndarray, assignment: np.ndarray, n_buckets: int) -> np.ndarray:
    """Sum ``item_values`` (shape (n_items,) or (n_items, k)) grouped by ``assignment`` bucket."""
    if item_values.ndim == 1:
        return np.bincount(assignment, weights=item_values, minlength=n_buckets)
    out = np.zeros((n_buckets, item_values.shape[1]), dtype=np.float64)
    for b in range(n_buckets):
        mask = assignment == b
        if np.any(mask):
            out[b] = item_values[mask].sum(axis=0)
    return out


def _objective(problem: BalanceProblem, assignment: np.ndarray) -> float:
    """The single objective every backend minimises: an independently designed, internally
    consistent weighted-L1-deviation objective.

    ``objective = size_term + count_term + actives_term`` where:

    - ``size_term = sum_b |realised_size[b] - bucket_target_size[b]| / max(1, mean(bucket_target_size))``
    - ``count_term`` (only when ``item_task_counts`` is given) =
      ``sum_{t,b} task_weight[t] * |realised_count[t,b] - target_share[t,b]| / max(1, task_total[t])``,
      where ``target_share[t,b] = bucket_target_size[b] / sum(bucket_target_size) * task_total[t]``
      (each task's total mass distributed proportionally to bucket size targets).
    - ``actives_term`` — the same formula applied to ``item_task_actives`` in place of
      ``item_task_counts``, only when both are given.
    """
    n_buckets = problem.n_buckets
    realised_size = _bucket_sums(problem.item_size.astype(np.float64), assignment, n_buckets)
    mean_target = problem.bucket_target_size.mean() if n_buckets else 1.0
    size_term = float(np.abs(realised_size - problem.bucket_target_size).sum()) / max(
        1.0, mean_target
    )

    total = size_term
    if problem.item_task_counts is not None:
        n_tasks = problem.n_tasks
        weight = (
            problem.task_weight if problem.task_weight is not None else np.ones(n_tasks)
        )
        target_total = problem.bucket_target_size.sum()
        bucket_share = (
            problem.bucket_target_size / target_total
            if target_total > 0
            else np.full(n_buckets, 1.0 / max(1, n_buckets))
        )
        task_total = problem.item_task_counts.sum(axis=0)
        realised_counts = _bucket_sums(problem.item_task_counts, assignment, n_buckets)
        target_share = np.outer(bucket_share, task_total)  # (n_buckets, n_tasks)
        denom = np.maximum(1.0, task_total)
        count_term = float(
            (weight * np.abs(realised_counts - target_share) / denom).sum()
        )
        total += count_term

        if problem.item_task_actives is not None:
            actives_total = problem.item_task_actives.sum(axis=0)
            realised_actives = _bucket_sums(problem.item_task_actives, assignment, n_buckets)
            actives_share = np.outer(bucket_share, actives_total)
            actives_denom = np.maximum(1.0, actives_total)
            actives_term = float(
                (weight * np.abs(realised_actives - actives_share) / actives_denom).sum()
            )
            total += actives_term

    return total


# ---------------------------------------------------------------------------
# Incremental bucket-state bookkeeping (used by local_search and anneal)
# ---------------------------------------------------------------------------


class _BucketState:
    """Running per-bucket totals, enabling O(n_tasks) delta-objective evaluation of a single
    item move instead of an O(n_items) full-objective recomputation."""

    __slots__ = (
        "problem",
        "bucket_size",
        "bucket_counts",
        "bucket_actives",
        "mean_target",
        "bucket_share",
        "task_total",
        "count_denom",
        "weight",
        "actives_total",
        "actives_denom",
    )

    def __init__(self, problem: BalanceProblem, assignment: np.ndarray) -> None:
        self.problem = problem
        n_buckets = problem.n_buckets
        self.bucket_size = _bucket_sums(
            problem.item_size.astype(np.float64), assignment, n_buckets
        )
        self.mean_target = problem.bucket_target_size.mean() if n_buckets else 1.0

        self.bucket_counts = None
        self.bucket_actives = None
        self.bucket_share = None
        self.task_total = None
        self.count_denom = None
        self.weight = None
        self.actives_total = None
        self.actives_denom = None
        if problem.item_task_counts is not None:
            self.bucket_counts = _bucket_sums(problem.item_task_counts, assignment, n_buckets)
            target_total = problem.bucket_target_size.sum()
            self.bucket_share = (
                problem.bucket_target_size / target_total
                if target_total > 0
                else np.full(n_buckets, 1.0 / max(1, n_buckets))
            )
            self.task_total = problem.item_task_counts.sum(axis=0)
            self.count_denom = np.maximum(1.0, self.task_total)
            self.weight = (
                problem.task_weight
                if problem.task_weight is not None
                else np.ones(problem.n_tasks)
            )
            if problem.item_task_actives is not None:
                self.bucket_actives = _bucket_sums(
                    problem.item_task_actives, assignment, n_buckets
                )
                self.actives_total = problem.item_task_actives.sum(axis=0)
                self.actives_denom = np.maximum(1.0, self.actives_total)

    def _size_term(self, bucket_size: np.ndarray) -> float:
        return float(
            np.abs(bucket_size - self.problem.bucket_target_size).sum()
        ) / max(1.0, self.mean_target)

    def _count_term(self, bucket_counts: np.ndarray) -> float:
        if self.bucket_counts is None:
            return 0.0
        target_share = np.outer(self.bucket_share, self.task_total)
        return float((self.weight * np.abs(bucket_counts - target_share) / self.count_denom).sum())

    def _actives_term(self, bucket_actives: np.ndarray) -> float:
        if self.bucket_actives is None:
            return 0.0
        target_share = np.outer(self.bucket_share, self.actives_total)
        return float(
            (self.weight * np.abs(bucket_actives - target_share) / self.actives_denom).sum()
        )

    def objective(self) -> float:
        return (
            self._size_term(self.bucket_size)
            + self._count_term(self.bucket_counts if self.bucket_counts is not None else 0.0)
            + self._actives_term(
                self.bucket_actives if self.bucket_actives is not None else 0.0
            )
        )

    def move_delta(self, item: int, from_bucket: int, to_bucket: int) -> float:
        """Objective change from moving ``item`` out of ``from_bucket`` into ``to_bucket``,
        without mutating state."""
        before = self._size_term(self.bucket_size)
        size = self.bucket_size.copy()
        size[from_bucket] -= self.problem.item_size[item]
        size[to_bucket] += self.problem.item_size[item]
        after = self._size_term(size)
        delta = after - before

        if self.bucket_counts is not None:
            before_c = self._count_term(self.bucket_counts)
            counts = self.bucket_counts.copy()
            counts[from_bucket] -= self.problem.item_task_counts[item]
            counts[to_bucket] += self.problem.item_task_counts[item]
            delta += self._count_term(counts) - before_c

        if self.bucket_actives is not None:
            before_a = self._actives_term(self.bucket_actives)
            actives = self.bucket_actives.copy()
            actives[from_bucket] -= self.problem.item_task_actives[item]
            actives[to_bucket] += self.problem.item_task_actives[item]
            delta += self._actives_term(actives) - before_a

        return delta

    def apply_move(self, item: int, from_bucket: int, to_bucket: int) -> None:
        self.bucket_size[from_bucket] -= self.problem.item_size[item]
        self.bucket_size[to_bucket] += self.problem.item_size[item]
        if self.bucket_counts is not None:
            self.bucket_counts[from_bucket] -= self.problem.item_task_counts[item]
            self.bucket_counts[to_bucket] += self.problem.item_task_counts[item]
        if self.bucket_actives is not None:
            self.bucket_actives[from_bucket] -= self.problem.item_task_actives[item]
            self.bucket_actives[to_bucket] += self.problem.item_task_actives[item]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _solve_greedy(
    problem: BalanceProblem, *, rng: np.random.Generator | None = None, time_limit_s: float = 60.0
) -> BalanceSolution:
    """Deficit-first construction: largest items first (ties -> smallest index), each assigned to
    the bucket currently furthest below its target (ties -> smallest bucket index). No RNG,
    structurally deterministic."""
    from chemsplit.determinism import argmax_tiebreak, stable_sort

    t0 = time.monotonic()
    n_items, n_buckets = problem.n_items, problem.n_buckets
    order = stable_sort(range(n_items), key=lambda i: problem.item_size[i], desc=True)

    assignment = np.full(n_items, -1, dtype=np.int64)
    bucket_size = np.zeros(n_buckets)

    for i in order:
        if i in problem.fixed_bucket:
            b = problem.fixed_bucket[i]
        else:
            b = argmax_tiebreak(
                lambda bb: problem.bucket_target_size[bb] - bucket_size[bb], range(n_buckets)
            )
        assignment[i] = b
        bucket_size[b] += problem.item_size[i]

    obj = _objective(problem, assignment)
    return BalanceSolution(
        assignment=assignment,
        objective=obj,
        solver_status="time_limit_feasible",
        architecture="greedy",
        wall_time_s=time.monotonic() - t0,
        n_iterations=1,
        diagnostics={},
    )


def _solve_local_search(
    problem: BalanceProblem,
    *,
    rng: np.random.Generator | None = None,
    time_limit_s: float = 60.0,
    max_passes: int = 200,
) -> BalanceSolution:
    """Greedy construction followed by fixed-order, first-improvement swap/move hill-climbing with
    incremental delta-objective evaluation (module docstring)."""
    t0 = time.monotonic()
    greedy = _solve_greedy(problem, time_limit_s=time_limit_s)
    assignment = greedy.assignment.copy()
    state = _BucketState(problem, assignment)
    n_items, n_buckets = problem.n_items, problem.n_buckets

    n_passes = 0
    for n_passes in range(1, max_passes + 1):  # noqa: B007 -- read after the loop as n_iterations
        improved = False
        for i in range(n_items):
            if i in problem.fixed_bucket:
                continue
            cur = int(assignment[i])
            for b in range(n_buckets):
                if b == cur:
                    continue
                delta = state.move_delta(i, cur, b)
                if delta < -1e-12:
                    state.apply_move(i, cur, b)
                    assignment[i] = b
                    cur = b
                    improved = True
                    break
            if (i & 0x3FF) == 0 and time.monotonic() - t0 > time_limit_s:
                improved = False
                break
        if not improved:
            break
        if time.monotonic() - t0 > time_limit_s:
            break

    obj = state.objective()
    return BalanceSolution(
        assignment=assignment,
        objective=obj,
        solver_status="time_limit_feasible",
        architecture="local_search",
        wall_time_s=time.monotonic() - t0,
        n_iterations=n_passes,
        diagnostics={},
    )


def _solve_anneal(
    problem: BalanceProblem,
    *,
    rng: np.random.Generator,
    time_limit_s: float = 60.0,
    anneal_steps: int = 20_000,
    anneal_t0: float = 1.0,
    anneal_t1: float = 0.01,
) -> BalanceSolution:
    """Seeded Metropolis simulated annealing, starting from the greedy solution. Requires an
    explicit ``rng``."""
    if rng is None:
        raise ValueError("_solve_anneal requires an explicit rng")
    t0 = time.monotonic()
    greedy = _solve_greedy(problem, time_limit_s=time_limit_s)
    assignment = greedy.assignment.copy()
    state = _BucketState(problem, assignment)
    n_items, n_buckets = problem.n_items, problem.n_buckets

    free_items = np.array([i for i in range(n_items) if i not in problem.fixed_bucket])
    best_assignment = assignment.copy()
    best_obj = state.objective()
    cur_obj = best_obj

    steps_done = 0
    if len(free_items) >= 1 and n_buckets >= 2:
        for step in range(anneal_steps):
            if (step & 0x3FF) == 0 and time.monotonic() - t0 > time_limit_s:
                break
            steps_done = step + 1
            frac = step / max(1, anneal_steps - 1)
            T = anneal_t0 * (anneal_t1 / anneal_t0) ** frac

            i = int(free_items[int(rng.integers(0, len(free_items)))])
            cur = int(assignment[i])
            others = [b for b in range(n_buckets) if b != cur]
            b = others[int(rng.integers(0, len(others)))]

            delta = state.move_delta(i, cur, b)
            accept = delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12))
            if accept:
                state.apply_move(i, cur, b)
                assignment[i] = b
                cur_obj += delta
                if cur_obj < best_obj - 1e-12:
                    best_obj = cur_obj
                    best_assignment = assignment.copy()

    return BalanceSolution(
        assignment=best_assignment,
        objective=best_obj,
        solver_status="time_limit_feasible",
        architecture="anneal",
        wall_time_s=time.monotonic() - t0,
        n_iterations=steps_done,
        diagnostics={},
    )


_BNB_MAX_ITEMS = 16


def _solve_bnb(
    problem: BalanceProblem, *, rng: np.random.Generator | None = None, time_limit_s: float = 60.0
) -> BalanceSolution:
    """Exact branch-and-bound. Small instances only (``n_items <= 16``) -- also used as an
    exactness oracle in tests. Deterministic (no RNG): explores buckets in ascending order
    at each branch."""
    if problem.n_items > _BNB_MAX_ITEMS:
        raise ValueError(
            f"_solve_bnb is only valid for n_items <= {_BNB_MAX_ITEMS} "
            f"(got {problem.n_items}); use a different architecture"
        )
    t0 = time.monotonic()
    n_items, n_buckets = problem.n_items, problem.n_buckets

    # Process largest items first: tightens bounds fastest.
    order = sorted(range(n_items), key=lambda i: -problem.item_size[i])
    suffix_size = np.zeros(n_items + 1)
    for k in range(n_items - 1, -1, -1):
        suffix_size[k] = suffix_size[k + 1] + problem.item_size[order[k]]

    best_assignment = np.zeros(n_items, dtype=np.int64)
    best_obj = [np.inf]
    timed_out = [False]

    def lower_bound(bucket_size: np.ndarray) -> float:
        # Admissible: current overshoot can only grow, never shrink, as more items are added.
        return float(np.maximum(0.0, bucket_size - problem.bucket_target_size).sum()) / max(
            1.0, problem.bucket_target_size.mean() if n_buckets else 1.0
        )

    assignment = np.full(n_items, -1, dtype=np.int64)

    def dfs(k: int, bucket_size: np.ndarray) -> None:
        if timed_out[0] or time.monotonic() - t0 > time_limit_s:
            timed_out[0] = True
            return
        if k == n_items:
            obj = _objective(problem, assignment)
            if obj < best_obj[0]:
                best_obj[0] = obj
                best_assignment[:] = assignment
            return
        if lower_bound(bucket_size) >= best_obj[0]:
            return
        i = order[k]
        candidate_buckets = (
            [problem.fixed_bucket[i]] if i in problem.fixed_bucket else range(n_buckets)
        )
        for b in candidate_buckets:
            assignment[i] = b
            bucket_size[b] += problem.item_size[i]
            dfs(k + 1, bucket_size)
            bucket_size[b] -= problem.item_size[i]
            if timed_out[0]:
                return
        assignment[i] = -1

    dfs(0, np.zeros(n_buckets))

    status = "time_limit_feasible" if timed_out[0] else "optimal"
    if not np.isfinite(best_obj[0]):
        return BalanceSolution(
            assignment=np.zeros(n_items, dtype=np.int64),
            objective=float("inf"),
            solver_status="infeasible",
            architecture="bnb",
            wall_time_s=time.monotonic() - t0,
            diagnostics={},
        )
    return BalanceSolution(
        assignment=best_assignment,
        objective=float(best_obj[0]),
        solver_status=status,
        architecture="bnb",
        wall_time_s=time.monotonic() - t0,
        diagnostics={},
    )


_MILP_MAX_BINARIES = 5000


def _solve_milp(
    problem: BalanceProblem,
    *,
    rng: np.random.Generator | None = None,
    time_limit_s: float = 60.0,
    mip_gap: float = 1e-4,
) -> BalanceSolution:
    """Exact/near-exact via ``scipy.optimize.milp`` (bundled HiGHS backend). ``scipy``'s own docs
    state the HiGHS wrapper "is deterministic"; no explicit thread/seed knob is exposed by
    ``scipy.optimize.milp``'s ``options`` dict (verified against the installed scipy: only
    ``disp``, ``node_limit``, ``presolve``, ``time_limit``, ``mip_rel_gap`` are recognised) -- the
    determinism contract is instead verified empirically by a repeat-solve test in
    ``tests/test_optimize.py``.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    n_items, n_buckets = problem.n_items, problem.n_buckets
    if n_items * n_buckets > _MILP_MAX_BINARIES:
        raise ValueError(
            f"_solve_milp is only valid for n_items*n_buckets <= {_MILP_MAX_BINARIES} "
            f"(got {n_items * n_buckets}); use a different architecture"
        )
    t0 = time.monotonic()

    n_x = n_items * n_buckets

    def xidx(i: int, b: int) -> int:
        return i * n_buckets + b

    n_tasks = problem.n_tasks
    has_actives = problem.item_task_actives is not None
    n_size_slack = n_buckets
    n_count_slack = n_tasks * n_buckets
    n_actives_slack = n_tasks * n_buckets if has_actives else 0
    n_vars = n_x + n_size_slack + n_count_slack + n_actives_slack

    def size_slack_idx(b: int) -> int:
        return n_x + b

    def count_slack_idx(t: int, b: int) -> int:
        return n_x + n_size_slack + t * n_buckets + b

    def actives_slack_idx(t: int, b: int) -> int:
        return n_x + n_size_slack + n_count_slack + t * n_buckets + b

    mean_target = problem.bucket_target_size.mean() if n_buckets else 1.0
    c = np.zeros(n_vars)
    for b in range(n_buckets):
        c[size_slack_idx(b)] = 1.0 / max(1.0, mean_target)

    weight = None
    task_total = actives_total = None
    bucket_share = None
    if n_tasks:
        weight = problem.task_weight if problem.task_weight is not None else np.ones(n_tasks)
        target_total = problem.bucket_target_size.sum()
        bucket_share = (
            problem.bucket_target_size / target_total
            if target_total > 0
            else np.full(n_buckets, 1.0 / max(1, n_buckets))
        )
        task_total = problem.item_task_counts.sum(axis=0)
        count_denom = np.maximum(1.0, task_total)
        for t in range(n_tasks):
            for b in range(n_buckets):
                c[count_slack_idx(t, b)] = weight[t] / count_denom[t]
        if has_actives:
            actives_total = problem.item_task_actives.sum(axis=0)
            actives_denom = np.maximum(1.0, actives_total)
            for t in range(n_tasks):
                for b in range(n_buckets):
                    c[actives_slack_idx(t, b)] = weight[t] / actives_denom[t]

    rows_i: list[int] = []
    rows_j: list[int] = []
    rows_v: list[float] = []
    b_l: list[float] = []
    b_u: list[float] = []
    row = 0

    def add_row(pairs: list[tuple[int, float]], lo: float, hi: float) -> None:
        nonlocal row
        for j, v in pairs:
            rows_i.append(row)
            rows_j.append(j)
            rows_v.append(v)
        b_l.append(lo)
        b_u.append(hi)
        row += 1

    for i in range(n_items):
        add_row([(xidx(i, b), 1.0) for b in range(n_buckets)], 1.0, 1.0)

    for b in range(n_buckets):
        pairs = [(xidx(i, b), float(problem.item_size[i])) for i in range(n_items)]
        target = float(problem.bucket_target_size[b])
        add_row(pairs + [(size_slack_idx(b), -1.0)], -np.inf, target)
        add_row([(j, -v) for j, v in pairs] + [(size_slack_idx(b), -1.0)], -np.inf, -target)

    for t in range(n_tasks):
        for b in range(n_buckets):
            pairs = [
                (xidx(i, b), float(problem.item_task_counts[i, t])) for i in range(n_items)
            ]
            target = float(bucket_share[b] * task_total[t])
            add_row(pairs + [(count_slack_idx(t, b), -1.0)], -np.inf, target)
            add_row([(j, -v) for j, v in pairs] + [(count_slack_idx(t, b), -1.0)], -np.inf, -target)
            if has_actives:
                apairs = [
                    (xidx(i, b), float(problem.item_task_actives[i, t])) for i in range(n_items)
                ]
                atarget = float(bucket_share[b] * actives_total[t])
                add_row(apairs + [(actives_slack_idx(t, b), -1.0)], -np.inf, atarget)
                add_row(
                    [(j, -v) for j, v in apairs] + [(actives_slack_idx(t, b), -1.0)],
                    -np.inf,
                    -atarget,
                )

    from scipy.sparse import csr_matrix

    A = csr_matrix((rows_v, (rows_i, rows_j)), shape=(row, n_vars))
    constraint = LinearConstraint(A, np.array(b_l), np.array(b_u))

    lower = np.zeros(n_vars)
    upper = np.full(n_vars, np.inf)
    upper[:n_x] = 1.0
    integrality = np.zeros(n_vars)
    integrality[:n_x] = 1

    for i, b in problem.fixed_bucket.items():
        for bb in range(n_buckets):
            v = 1.0 if bb == b else 0.0
            lower[xidx(i, bb)] = v
            upper[xidx(i, bb)] = v

    bounds = Bounds(lower, upper)

    res = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=[constraint],
        options={"time_limit": time_limit_s, "mip_rel_gap": mip_gap, "presolve": True},
    )

    wall = time.monotonic() - t0
    if res.x is None:
        status = "infeasible" if res.status == 2 else "time_limit_feasible"
        return BalanceSolution(
            assignment=np.zeros(n_items, dtype=np.int64),
            objective=float("inf"),
            solver_status="infeasible" if status == "infeasible" else "time_limit_feasible",
            architecture="milp",
            wall_time_s=wall,
            diagnostics={"scipy_status": int(res.status), "message": res.message},
        )

    from chemsplit.determinism import argmax_tiebreak

    x = res.x[:n_x].reshape(n_items, n_buckets)
    # Rounding a near-binary MILP solution to a hard assignment; ties (essentially never seen from
    # a real solve, but routed through argmax_tiebreak for the same smallest-index determinism
    # guarantee as everywhere else --).
    assignment = np.array(
        [argmax_tiebreak(lambda b, row=x[i]: row[b], range(n_buckets)) for i in range(n_items)],
        dtype=np.int64,
    )
    status_map = {0: "optimal", 1: "time_limit_feasible"}
    solver_status = status_map.get(int(res.status), "time_limit_feasible")
    obj = _objective(problem, assignment)
    return BalanceSolution(
        assignment=assignment,
        objective=obj,
        solver_status=solver_status,
        architecture="milp",
        wall_time_s=wall,
        diagnostics={
            "scipy_status": int(res.status),
            "mip_gap": getattr(res, "mip_gap", None),
            "mip_node_count": getattr(res, "mip_node_count", None),
        },
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Thresholds on n_items*n_buckets, frozen empirically. See
# tests/test_optimize.py::test_dispatch_thresholds_stable, which pins these values so an
# accidental edit is caught. "auto" never routes to milp above the tiny bnb-eligible regime;
# milp stays available as an explicit opt-in (used as an exactness oracle in tests).
_DISPATCH_BNB_MAX_BINARIES = 48  # n_items <= 16 guard (below) dominates in practice


def _select_architecture(problem: BalanceProblem) -> ArchitectureName:
    """Pure function of problem size -- a frozen lookup table, never runtime auto-tuning, so
    ``architecture="auto"`` stays perfectly reproducible. ``local_search`` completes in well
    under 5s across tested problem sizes and is preferred above the tiny bnb-eligible regime."""
    n_binaries = problem.n_items * problem.n_buckets
    if problem.n_items <= _BNB_MAX_ITEMS and n_binaries <= _DISPATCH_BNB_MAX_BINARIES:
        return "bnb"
    return "local_search"


_BACKENDS = {
    "greedy": _solve_greedy,
    "local_search": _solve_local_search,
    "anneal": _solve_anneal,
    "bnb": _solve_bnb,
    "milp": _solve_milp,
}


def solve_balance(
    problem: BalanceProblem,
    *,
    rng: np.random.Generator | None = None,
    time_limit_s: float = 60.0,
    architecture: ArchitectureName | Literal["auto"] = "auto",
    mip_gap: float = 1e-4,
    anneal_steps: int = 20_000,
    anneal_t0: float = 1.0,
    anneal_t1: float = 0.01,
) -> BalanceSolution:
    """Solve ``problem``, dispatching to the architecture named (or, for ``"auto"``, selected by
    the frozen, size-keyed :func:`_select_architecture` lookup table)."""
    name = _select_architecture(problem) if architecture == "auto" else architecture
    if name not in _BACKENDS:
        raise ValueError(f"unknown architecture {name!r}; expected one of {sorted(_BACKENDS)} or 'auto'")

    if name == "anneal":
        return _solve_anneal(
            problem,
            rng=rng,
            time_limit_s=time_limit_s,
            anneal_steps=anneal_steps,
            anneal_t0=anneal_t0,
            anneal_t1=anneal_t1,
        )
    if name == "milp":
        return _solve_milp(problem, rng=rng, time_limit_s=time_limit_s, mip_gap=mip_gap)
    return _BACKENDS[name](problem, rng=rng, time_limit_s=time_limit_s)
