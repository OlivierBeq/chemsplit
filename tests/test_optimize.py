"""Tests for chemsplit._optimize: the independent balanced-partitioning engine."""

from __future__ import annotations

import numpy as np
import pytest

from chemsplit._optimize import (
    BalanceProblem,
    _objective,
    _select_architecture,
    _solve_anneal,
    _solve_bnb,
    _solve_greedy,
    _solve_local_search,
    _solve_milp,
    solve_balance,
)


def _tiny_problem(seed=0, n=10, n_buckets=3, n_tasks=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 20, size=n).astype(np.int64)
    total = sizes.sum()
    raw = np.array([total * 0.6, total * 0.25, total * 0.15])
    if n_buckets <= 3:
        targets = raw[:n_buckets]
    else:
        targets = np.concatenate([raw, np.full(n_buckets - 3, total * 0.05)])
    targets = targets / targets.sum() * total
    kwargs = {}
    if n_tasks:
        kwargs["item_task_counts"] = rng.integers(0, 5, size=(n, n_tasks)).astype(np.float64)
    return BalanceProblem(
        n_items=n, n_buckets=n_buckets, item_size=sizes, bucket_target_size=targets, **kwargs
    )


# ---------------------------------------------------------------------------
# Objective correctness
# ---------------------------------------------------------------------------


def test_objective_permutation_invariant():
    problem = _tiny_problem(n=8, n_tasks=2)
    assignment = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    obj = _objective(problem, assignment)

    perm = np.array([3, 0, 5, 1, 7, 2, 4, 6])
    permuted = BalanceProblem(
        n_items=problem.n_items,
        n_buckets=problem.n_buckets,
        item_size=problem.item_size[perm],
        bucket_target_size=problem.bucket_target_size,
        item_task_counts=problem.item_task_counts[perm],
    )
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    permuted_assignment = assignment[perm]
    obj2 = _objective(permuted, permuted_assignment)
    assert obj == pytest.approx(obj2)


def test_objective_zero_when_evenly_split():
    sizes = np.array([10, 10, 10, 10], dtype=np.int64)
    problem = BalanceProblem(
        n_items=4, n_buckets=2, item_size=sizes, bucket_target_size=np.array([20.0, 20.0])
    )
    assignment = np.array([0, 0, 1, 1])
    assert _objective(problem, assignment) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["greedy", "local_search", "bnb"])
def test_unseeded_backends_deterministic(arch):
    problem = _tiny_problem(n=10, n_tasks=2)
    sol1 = solve_balance(problem, architecture=arch, time_limit_s=10)
    sol2 = solve_balance(problem, architecture=arch, time_limit_s=10)
    assert np.array_equal(sol1.assignment, sol2.assignment)


def test_anneal_deterministic_given_seed():
    problem = _tiny_problem(n=15, n_tasks=1)
    sol1 = _solve_anneal(problem, rng=np.random.default_rng(42), time_limit_s=5, anneal_steps=2000)
    sol2 = _solve_anneal(problem, rng=np.random.default_rng(42), time_limit_s=5, anneal_steps=2000)
    assert np.array_equal(sol1.assignment, sol2.assignment)


def test_anneal_requires_rng():
    problem = _tiny_problem(n=5)
    with pytest.raises(ValueError):
        _solve_anneal(problem, rng=None, time_limit_s=1)


def test_milp_deterministic_repeat_solve():
    problem = _tiny_problem(n=10, n_tasks=2)
    sol1 = _solve_milp(problem, time_limit_s=15)
    sol2 = _solve_milp(problem, time_limit_s=15)
    assert np.array_equal(sol1.assignment, sol2.assignment)
    assert sol1.solver_status == "optimal"


def test_auto_never_raises_across_sizes():
    for n in (5, 50, 500):
        problem = _tiny_problem(seed=n, n=n, n_buckets=3, n_tasks=(2 if n < 500 else 0))
        sol = solve_balance(problem, architecture="auto", rng=np.random.default_rng(0), time_limit_s=10)
        assert sol.assignment.shape == (n,)
        assert sol.solver_status in ("optimal", "time_limit_feasible")


# ---------------------------------------------------------------------------
# Optimality / quality
# ---------------------------------------------------------------------------


def test_bnb_matches_or_beats_heuristics():
    problem = _tiny_problem(n=12, n_buckets=3, n_tasks=2)
    bnb = _solve_bnb(problem, time_limit_s=20)
    greedy = _solve_greedy(problem)
    ls = _solve_local_search(problem, time_limit_s=10)
    assert bnb.solver_status == "optimal"
    assert bnb.objective <= greedy.objective + 1e-9
    assert bnb.objective <= ls.objective + 1e-9


def test_bnb_rejects_large_instances():
    problem = _tiny_problem(n=64, n_buckets=3)
    with pytest.raises(ValueError):
        _solve_bnb(problem, time_limit_s=1)


def test_milp_agrees_with_bnb_on_small_instance():
    problem = _tiny_problem(n=10, n_buckets=3, n_tasks=2)
    bnb = _solve_bnb(problem, time_limit_s=20)
    milp = _solve_milp(problem, time_limit_s=20)
    assert milp.objective == pytest.approx(bnb.objective, abs=1e-6)


def test_local_search_never_worse_than_greedy():
    for seed in range(5):
        problem = _tiny_problem(seed=seed, n=40, n_buckets=3, n_tasks=3)
        greedy = _solve_greedy(problem)
        ls = _solve_local_search(problem, time_limit_s=10)
        assert ls.objective <= greedy.objective + 1e-9


def test_fixed_bucket_is_honoured():
    problem = _tiny_problem(n=10, n_buckets=3)
    problem = BalanceProblem(
        n_items=problem.n_items,
        n_buckets=problem.n_buckets,
        item_size=problem.item_size,
        bucket_target_size=problem.bucket_target_size,
        fixed_bucket={0: 2, 1: 2},
    )
    for arch in ("greedy", "local_search", "bnb"):
        sol = solve_balance(problem, architecture=arch, time_limit_s=10)
        assert sol.assignment[0] == 2
        assert sol.assignment[1] == 2


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatch_thresholds_stable():
    """Pins the frozen dispatch constants against accidental edits."""
    import chemsplit._optimize as opt

    assert opt._DISPATCH_BNB_MAX_BINARIES == 48
    assert opt._BNB_MAX_ITEMS == 16


def test_select_architecture_is_pure_size_function():
    p_small = _tiny_problem(n=10, n_buckets=3)
    p_large = _tiny_problem(n=5000, n_buckets=5)
    assert _select_architecture(p_small) == "bnb"
    assert _select_architecture(p_large) == "local_search"


def test_unknown_architecture_raises():
    problem = _tiny_problem(n=5)
    with pytest.raises(ValueError):
        solve_balance(problem, architecture="not_a_real_architecture", time_limit_s=1)
