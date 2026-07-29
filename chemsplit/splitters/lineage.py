"""``lineage`` splitter family: date-based and provenance-based splitters.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

import numpy as np

from chemsplit._unionfind import dense_label_encode
from chemsplit.base import BaseSplitter, GroupSplitter, SplitResult, Strictness, _Context
from chemsplit.determinism import (
    seed_for,
    seeded_python_random,
    stable_sort,
)
from chemsplit.exceptions import (
    ConstraintUnsatisfiableError,
    DegenerateClusterWarning,
    DegenerateGroupingError,
    EmptyPartitionError,
    InputError,
    LabelError,
    MissingDependencyError,
    ParameterError,
    ParseWarning,
    warn_with_details,
)
from chemsplit.types import IndexArray

__all__ = ["PartySplitter", "SIMPDSplitter", "SourceSplitter", "TemporalSplitter"]

#: Folded into chemsplit.registry / chemsplit.__init__'s __all__ at integration time.
_EXPORTED = __all__

_EPS = 1e-9


# ---------------------------------------------------------------------------
# TemporalSplitter
# ---------------------------------------------------------------------------


def _coerce_dates(dates: Any, n: int) -> np.ndarray:
    if dates is None:
        raise LabelError(
            "TemporalSplitter requires dates: pass dates=... (or date_col=... for a DataFrame) "
            "to split()/split_result()."
        )
    arr = np.asarray(dates)
    try:
        coerced = arr.astype("datetime64[D]")
    except (ValueError, TypeError) as exc:
        raise InputError(f"could not coerce `dates` to datetime64[D]: {exc}") from exc
    if len(coerced) != n:
        raise InputError(f"len(dates)={len(coerced)} does not match n={n}")
    if np.isnat(coerced).any():
        offenders = np.nonzero(np.isnat(coerced))[0][:20].tolist()
        raise InputError(
            f"dates contains NaT at indices (first 20 shown): {offenders}. A record without a "
            "date cannot be time-split; drop or impute explicitly."
        )
    return coerced


def _parse_offset_days(design: "str | int") -> int:
    """Best-effort parse of a pandas-offset-like string (``"90D"``, ``"6M"``, ``"1Y"``) or a
    plain integer day count, into an integer number of days.

    Simplification, documented: month/year units are approximated as 30/365 days respectively
    rather than resolved against a calendar (pandas' ``DateOffset`` calendar arithmetic is not
    used here to keep window arithmetic index-free and trivially vectorisable). Good enough for
    the windowing granularity this splitter targets; exact calendar arithmetic can be substituted
    later without changing the public API.
    """
    if isinstance(design, (int, np.integer)) and not isinstance(design, bool):
        return int(design)
    if not isinstance(design, str):
        raise ParameterError(f"expected a pandas-offset-like string or int, got {design!r}")
    m = re.fullmatch(r"\s*(\d+)\s*([DWMY])\s*", design.upper())
    if not m:
        raise ParameterError(f"could not parse offset string {design!r} (expected e.g. '90D', '6M', '1Y')")
    count, unit = int(m.group(1)), m.group(2)
    days_per_unit = {"D": 1, "W": 7, "M": 30, "Y": 365}[unit]
    return count * days_per_unit


class TemporalSplitter(BaseSplitter):
    """Date-cut split: train on the past, test on the future."""

    splitter_id: ClassVar[str] = "temporal"
    family: ClassVar[str] = "lineage"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = False
    requires_dates: ClassVar[bool] = True
    requires_targets: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features", "interactions", "sequences")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = True
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = True

    def __init__(
        self,
        *,
        cut_date: "str | np.datetime64 | None" = None,
        valid_cut_date: "str | np.datetime64 | None" = None,
        embargo: "str | int" = 0,
        mode: Literal["single", "rolling", "expanding"] = "single",
        n_windows: int = 5,
        window: "str | int" = "365D",
        tie_policy: Literal["train", "test", "discard"] = "train",
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.cut_date = cut_date
        self.valid_cut_date = valid_cut_date
        self.embargo = embargo
        self.mode = mode
        self.n_windows = n_windows
        self.window = window
        self.tie_policy = tie_policy
        if mode not in ("single", "rolling", "expanding"):
            raise ParameterError(f"invalid mode: {mode!r}")
        if tie_policy not in ("train", "test", "discard"):
            raise ParameterError(f"invalid tie_policy: {tie_policy!r}")
        if isinstance(n_windows, bool) or not isinstance(n_windows, (int, np.integer)) or n_windows < 1:
            raise ParameterError(f"n_windows must be a positive int, got {n_windows!r}")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return int(self.n_windows) if self.mode in ("rolling", "expanding") else 1

    def _check_preconditions(self, ctx: _Context) -> None:
        _coerce_dates(ctx.dates, ctx.n)

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        dates = _coerce_dates(ctx.dates, ctx.n)
        if len(set(dates.tolist())) == 1:
            raise ConstraintUnsatisfiableError(
                f"all {ctx.n} records share one date ({dates[0]}); a temporal cut is degenerate"
            )
        if self.mode == "single":
            return [self._single_cut(ctx, dates)]
        return self._windowed(ctx, dates)

    def _single_cut(self, ctx: _Context, dates: np.ndarray) -> SplitResult:
        n = ctx.n
        order = stable_sort(list(range(n)), key=lambda i: (dates[i], i))
        if self.cut_date is not None:
            cut = np.datetime64(self.cut_date, "D")
        else:
            q = ctx.sizes.n_train + ctx.sizes.n_valid - 1
            q = min(max(q, 0), n - 1)
            cut = dates[order[q]]
        if cut < dates.min() or cut > dates.max():
            raise EmptyPartitionError(
                f"cut_date {cut} is outside the data range "
                f"[{dates.min()}, {dates.max()}]"
            )

        embargo_days = _parse_offset_days(self.embargo)
        if embargo_days > 0:
            embargo_end = cut + np.timedelta64(embargo_days, "D")
        else:
            embargo_end = cut

        n_ties = int(np.sum(dates == cut))
        train_mask = dates < cut
        tie_mask = dates == cut
        if self.tie_policy == "train":
            train_mask = train_mask | tie_mask
            post_tie_mask = np.zeros(n, dtype=bool)
        elif self.tie_policy == "discard":
            post_tie_mask = tie_mask
        else:  # "test" -- still subject to the embargo, since ties sit exactly at its start
            post_tie_mask = tie_mask

        after_cut = dates > cut
        discard_mask = post_tie_mask | (after_cut & (dates < embargo_end))
        test_mask = (~train_mask) & (~discard_mask) & (after_cut | (tie_mask & (self.tie_policy == "test") & (embargo_days == 0)))
        # Records strictly after the embargo window are always test, regardless of tie_policy.
        test_mask = test_mask | (dates >= embargo_end) & (~train_mask) & (~discard_mask)

        train = np.nonzero(train_mask)[0]
        test = np.nonzero(test_mask)[0]
        discard = np.nonzero(discard_mask)[0]

        valid = np.array([], dtype=np.int64)
        if self.valid_cut_date is not None:
            vcut = np.datetime64(self.valid_cut_date, "D")
            valid_mask_local = dates[train] >= vcut
            valid = train[valid_mask_local]
            train = train[~valid_mask_local]
        elif ctx.sizes.n_valid > 0:
            train_sorted = stable_sort(train.tolist(), key=lambda i: (dates[i], i), desc=True)
            n_valid = min(ctx.sizes.n_valid, len(train_sorted))
            valid = np.array(sorted(train_sorted[:n_valid]), dtype=np.int64)
            train = np.array(sorted(train_sorted[n_valid:]), dtype=np.int64)

        if len(train) == 0 or len(test) == 0:
            raise EmptyPartitionError(
                f"temporal cut at {cut} (embargo {self.embargo}) leaves train={len(train)}, "
                f"test={len(test)}"
            )

        metadata: dict[str, Any] = {
            "cut_date": str(cut),
            "valid_cut_date": str(self.valid_cut_date) if self.valid_cut_date is not None else None,
            "embargo": str(self.embargo),
            "train_date_range": [str(dates[train].min()), str(dates[train].max())] if len(train) else None,
            "test_date_range": [str(dates[test].min()), str(dates[test].max())] if len(test) else None,
            "n_ties_at_cut": n_ties,
            "n_embargoed": int(np.sum(after_cut & (dates < embargo_end))),
            "window_index": None,
            "realised_sizes": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
        }
        result = SplitResult(
            train=np.sort(train.astype(np.int64)),
            valid=np.sort(valid.astype(np.int64)),
            test=np.sort(test.astype(np.int64)),
            discard=np.sort(discard.astype(np.int64)),
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=ctx.n,
            metadata=metadata,
        )
        if n_ties > 0 and abs(len(train) - (ctx.sizes.n_train + ctx.sizes.n_valid)) > max(10, 0.01 * ctx.n):
            from chemsplit.exceptions import SizeToleranceWarning

            warn_with_details(
                SizeToleranceWarning(
                    f"TemporalSplitter: {n_ties} ties at the cut date shifted realised sizes "
                    f"beyond tolerance"
                )
            )
        return result

    def _windowed(self, ctx: _Context, dates: np.ndarray) -> list[SplitResult]:
        window_days = _parse_offset_days(self.window)
        date_max = dates.max()
        results: list[SplitResult] = []
        for i in range(int(self.n_windows)):
            offset_end = (int(self.n_windows) - i - 1) * window_days
            offset_start = (int(self.n_windows) - i) * window_days
            test_start = date_max - np.timedelta64(offset_start, "D")
            test_end = date_max - np.timedelta64(offset_end, "D")
            is_last = i == int(self.n_windows) - 1
            test_mask = (dates >= test_start) & ((dates < test_end) if not is_last else (dates <= test_end))
            if self.mode == "rolling":
                train_start = test_start - np.timedelta64(window_days, "D")
                train_mask = (dates >= train_start) & (dates < test_start)
            else:  # "expanding"
                train_mask = dates < test_start
            train = np.nonzero(train_mask)[0]
            test = np.nonzero(test_mask)[0]
            if len(train) == 0 or len(test) == 0:
                continue
            # Records outside this fold's (train_start, test_end] window are not part of this
            # temporal fold at all -- I2 requires every record land in train/valid/test/discard,
            # so the rest of the timeline is `discard` *for this fold specifically* (# discard = "records deliberately dropped by the splitter").
            used_mask = train_mask | test_mask
            discard = np.nonzero(~used_mask)[0]
            metadata = {
                "cut_date": str(test_start),
                "valid_cut_date": None,
                "embargo": str(self.embargo),
                "train_date_range": [str(dates[train].min()), str(dates[train].max())],
                "test_date_range": [str(dates[test].min()), str(dates[test].max())],
                "n_ties_at_cut": 0,
                "n_embargoed": 0,
                "window_index": i,
                "realised_sizes": {"train": int(len(train)), "valid": 0, "test": int(len(test))},
            }
            results.append(
                SplitResult(
                    train=np.sort(train.astype(np.int64)),
                    valid=np.array([], dtype=np.int64),
                    test=np.sort(test.astype(np.int64)),
                    discard=np.sort(discard.astype(np.int64)),
                    groups=None,
                    splitter_id=self.splitter_id,
                    params=self.get_params(),
                    n_records=ctx.n,
                    metadata=metadata,
                )
            )
        if not results:
            raise EmptyPartitionError(
                f"mode={self.mode!r} with window={self.window!r}, n_windows={self.n_windows} "
                "produced no non-empty (train, test) window over the data's date range"
            )
        return results


# ---------------------------------------------------------------------------
# SIMPDSplitter
# ---------------------------------------------------------------------------


class SIMPDSplitter(BaseSplitter):
    """Simulated time split: a multi-objective GA rearranges an undated dataset until the
    train/test pair reproduces the descriptor/property shifts measured in real time splits.

    Requires the ``ga`` extra (``deap``).

    **Implementation note:** DEAP's built-in variation operators (``tools.cxTwoPoint``,
    ``tools.mutFlipBit``, ``tools.selTournament``) read Python's *global* ``random`` module
    internally, which conflicts with chemsplit's rule that the global ``random`` module must
    never be touched (see ``chemsplit/determinism.py``). This implementation uses DEAP only for
    the RNG-free parts (``creator``/``base.Fitness`` bookkeeping and ``tools.selNSGA2``, which is
    a deterministic rank/crowding-distance sort) and hand-rolls crossover/mutation/tournament
    selection using a dedicated ``random.Random`` instance from
    :func:`chemsplit.determinism.seeded_python_random`.
    """

    splitter_id: ClassVar[str] = "simpd"
    family: ClassVar[str] = "lineage"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = True
    requires_dates: ClassVar[bool] = False
    requires_targets: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol")
    extras: ClassVar[tuple[str,...]] = ("ga",)
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    _DEFAULT_TARGETS: ClassVar[dict[str, float]] = {
        "frac_test_in_train_cluster": 0.60,
        "delta_active_frac": -0.03,
        "delta_MolWt": 0.25,
        "delta_MolLogP": 0.15,
        "delta_TPSA": 0.10,
        "g_sim": 0.35,
    }
    _DEFAULT_DESCRIPTORS: ClassVar[tuple[str,...]] = (
        "MolWt", "MolLogP", "TPSA", "NumRotatableBonds", "NumHAcceptors", "NumHDonors",
        "FractionCSP3", "NumAromaticRings", "RingCount", "HeavyAtomCount",
    )

    def __init__(
        self,
        *,
        targets: "dict[str, float] | None" = None,
        descriptors: "tuple[str,...]" = _DEFAULT_DESCRIPTORS,
        population_size: int = 500,
        n_generations: int = 200,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        mutation_indpb: float = 0.02,
        tournament_size: int = 3,
        cluster_for_g_sim: "str | BaseSplitter" = "butina",
        early_stop_patience: int = 40,
        max_memory_bytes: int = 2 * 1024**3,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        resolved = dict(self._DEFAULT_TARGETS)
        if targets:
            unknown = set(targets) - set(self._DEFAULT_TARGETS)
            if unknown:
                raise ParameterError(
                    f"unknown target key(s) {sorted(unknown)}; expected a subset of "
                    f"{sorted(self._DEFAULT_TARGETS)}"
                )
            resolved.update(targets)
        self.targets = resolved
        self.descriptors = tuple(descriptors)
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_indpb = mutation_indpb
        self.tournament_size = tournament_size
        self.cluster_for_g_sim = cluster_for_g_sim
        self.early_stop_patience = early_stop_patience
        self.max_memory_bytes = max_memory_bytes

    def _check_preconditions(self, ctx: _Context) -> None:
        if ctx.n < 200:
            raise ParameterError(
                f"SIMPDSplitter requires n >= 200 (got {ctx.n}); the GA cannot meaningfully "
                "optimise on smaller datasets. Use temporal (TemporalSplitter, if dates are "
                "available) or butina (ButinaSplitter) instead."
            )

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        try:
            import deap.base as deap_base
            import deap.creator as deap_creator
            import deap.tools as deap_tools
        except ImportError as exc:
            raise MissingDependencyError("SIMPDSplitter", "ga") from exc

        from rdkit.Chem import Descriptors

        from chemsplit._fp_similarity import guard_memory
        from chemsplit.clustering import butina
        from chemsplit.featurizers import get_featurizer
        from chemsplit.metrics import pairwise_distances

        n = ctx.n
        n_test = ctx.sizes.n_test
        mols = ctx.mols
        y = np.asarray(ctx.y, dtype=float)

        guard_memory(n, self.max_memory_bytes, "SIMPDSplitter")
        featurizer = get_featurizer("ecfp4")
        F = ctx.get_features(featurizer)
        D = pairwise_distances(F, metric="tanimoto")
        S = 1.0 - D

        desc_values = {}
        for name in self.descriptors:
            fn = getattr(Descriptors, name, None)
            if fn is None:
                raise ParameterError(f"unknown RDKit descriptor {name!r}")
            desc_values[name] = np.array(
                [fn(m) if m is not None else 0.0 for m in mols], dtype=float
            )

        # Butina clusters for `frac_test_in_train_cluster`, per `cluster_for_g_sim`. The default
        # "butina" (renamed ButinaSplitter) is not resolved via the registry here (the
        # `similarity` family may not exist yet at any given point in this package's build) --
        # calling the underlying `chemsplit.clustering.butina` primitive directly with its default
        # cutoff (0.35 distance) reproduces the same clustering ButinaSplitter would use.
        # Documented simplification.
        if isinstance(self.cluster_for_g_sim, str):
            clusters = butina(D, cutoff=0.35)
            cluster_of = np.empty(n, dtype=np.int64)
            for cid, members in enumerate(clusters):
                for m in members:
                    cluster_of[m] = cid
        else:
            groups = self.cluster_for_g_sim.compute_groups(mols)
            cluster_of = np.asarray(groups, dtype=np.int64)

        bundle = ctx.rng_seeds
        py_rng = seeded_python_random(bundle, "simpd.ga", 0)

        target_keys = list(self.targets.keys())
        target_vals = np.array([self.targets[k] for k in target_keys], dtype=float)

        def observe(test_mask: np.ndarray) -> np.ndarray:
            test_idx = np.nonzero(test_mask)[0]
            train_idx = np.nonzero(~test_mask)[0]
            obs = {}
            if len(test_idx) and len(train_idx):
                same_cluster = np.isin(cluster_of[test_idx], cluster_of[train_idx])
                obs["frac_test_in_train_cluster"] = float(np.mean(same_cluster))
                test_active = float(np.mean(y[test_idx] >= np.median(y)))
                train_active = float(np.mean(y[train_idx] >= np.median(y)))
                obs["delta_active_frac"] = test_active - train_active
                for name in ("MolWt", "MolLogP", "TPSA"):
                    key = f"delta_{name}"
                    if key in self.targets:
                        a, b = desc_values[name][test_idx], desc_values[name][train_idx]
                        pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0) if len(a) > 1 and len(b) > 1 else 1.0
                        obs[key] = float((a.mean() - b.mean()) / max(pooled_std, 1e-9))
                sub = S[np.ix_(test_idx, train_idx)]
                obs["g_sim"] = float(np.mean(sub.max(axis=1))) if sub.size else 0.0
            else:
                obs = dict.fromkeys(target_keys, 0.0)
            return np.array([obs.get(k, 0.0) for k in target_keys], dtype=float)

        def fitness_of(mask: np.ndarray) -> tuple[float,...]:
            observed = observe(mask)
            return tuple((-((observed - target_vals) ** 2)).tolist())

        if not hasattr(deap_creator, "FitnessSIMPD"):
            deap_creator.create("FitnessSIMPD", deap_base.Fitness, weights=(1.0,) * len(target_keys))
        if not hasattr(deap_creator, "IndividualSIMPD"):
            deap_creator.create("IndividualSIMPD", np.ndarray, fitness=deap_creator.FitnessSIMPD)

        def repair(mask: np.ndarray) -> np.ndarray:
            mask = mask.copy()
            true_idx = np.nonzero(mask)[0].tolist()
            false_idx = np.nonzero(~mask)[0].tolist()
            while len(true_idx) > n_test:
                pick = py_rng.randrange(len(true_idx))
                i = true_idx.pop(pick)
                mask[i] = False
                false_idx.append(i)
            while len(true_idx) < n_test:
                pick = py_rng.randrange(len(false_idx))
                i = false_idx.pop(pick)
                mask[i] = True
                true_idx.append(i)
            return mask

        def new_individual() -> Any:
            idx = py_rng.sample(range(n), n_test)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            ind = mask.view(deap_creator.IndividualSIMPD)
            ind.fitness.values = fitness_of(mask)
            return ind

        def crossover(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if n < 2:
                return a, b
            p1, p2 = sorted(py_rng.sample(range(n), 2))
            a2, b2 = a.copy(), b.copy()
            a2[p1:p2], b2[p1:p2] = b[p1:p2].copy(), a[p1:p2].copy()
            return repair(a2), repair(b2)

        def mutate(a: np.ndarray) -> np.ndarray:
            flip = np.array([py_rng.random() < self.mutation_indpb for _ in range(n)])
            a2 = a.copy()
            a2[flip] = ~a2[flip]
            return repair(a2)

        def tournament(pop: list) -> Any:
            contestants = [pop[py_rng.randrange(len(pop))] for _ in range(self.tournament_size)]
            return max(contestants, key=lambda ind: ind.fitness.values)

        population = [new_individual() for _ in range(self.population_size)]
        best_sq_err = None
        stall = 0
        hypervolume_trace: list[float] = []

        def sq_err(ind: Any) -> float:
            return float(sum(v * v for v in ind.fitness.values))

        for gen in range(int(self.n_generations)):
            offspring = []
            while len(offspring) < self.population_size:
                p1, p2 = tournament(population), tournament(population)
                c1, c2 = np.array(p1), np.array(p2)
                if py_rng.random() < self.crossover_prob:
                    c1, c2 = crossover(c1, c2)
                if py_rng.random() < self.mutation_prob:
                    c1 = mutate(c1)
                if py_rng.random() < self.mutation_prob:
                    c2 = mutate(c2)
                for c in (c1, c2):
                    ind = c.view(deap_creator.IndividualSIMPD)
                    ind.fitness.values = fitness_of(c)
                    offspring.append(ind)
            combined = population + offspring[: self.population_size]
            population = deap_tools.selNSGA2(combined, self.population_size)

            gen_best = min(sq_err(ind) for ind in population)
            hypervolume_trace.append(-gen_best)
            if best_sq_err is None or gen_best < best_sq_err - 1e-12:
                best_sq_err = gen_best
                stall = 0
            else:
                stall += 1
            if stall >= self.early_stop_patience:
                break

        best = min(
            population,
            key=lambda ind: (sq_err(ind), tuple(int(b) for b in np.asarray(ind))),
        )
        best_mask = np.asarray(best, dtype=bool)
        observed = observe(best_mask)
        achieved = dict(zip(target_keys, observed.tolist()))
        errors = {k: abs(achieved[k] - self.targets[k]) for k in target_keys}
        for k, err in errors.items():
            if abs(self.targets[k]) > 1e-9 and err > 0.25 * abs(self.targets[k]):
                from chemsplit.exceptions import SizeToleranceWarning

                warn_with_details(
                    SizeToleranceWarning(
                        f"SIMPDSplitter: objective {k!r} residual {err:.4f} exceeds 25% of its "
                        f"target {self.targets[k]!r}"
                    )
                )

        test = np.nonzero(best_mask)[0]
        train = np.nonzero(~best_mask)[0]
        valid = np.array([], dtype=np.int64)
        if ctx.sizes.n_valid > 0:
            rng = seed_for(ctx.rng_seeds, "simpd.valid_carve", 0)
            perm = rng.permutation(train)
            valid = np.sort(perm[: ctx.sizes.n_valid])
            train = np.sort(perm[ctx.sizes.n_valid:])

        metadata = {
            "targets": self.targets,
            "achieved": achieved,
            "objective_errors": errors,
            "generations_run": gen + 1,
            "converged": stall >= self.early_stop_patience,
            "pareto_front_size": len(population),
            "hypervolume_trace": hypervolume_trace,
            "realised_sizes": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
        }
        return [
            SplitResult(
                train=np.sort(train.astype(np.int64)),
                valid=np.sort(valid.astype(np.int64)),
                test=np.sort(test.astype(np.int64)),
                discard=np.array([], dtype=np.int64),
                groups=None,
                splitter_id=self.splitter_id,
                params=self.get_params(),
                n_records=ctx.n,
                metadata=metadata,
            )
        ]


# ---------------------------------------------------------------------------
# SourceSplitter
# ---------------------------------------------------------------------------


class SourceSplitter(GroupSplitter):
    """Groups by provenance: document, assay, lab, vendor, plate, or any caller-supplied key."""

    splitter_id: ClassVar[str] = "source"
    family: ClassVar[str] = "lineage"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    requires_labels: ClassVar[bool] = False
    requires_dates: ClassVar[bool] = False
    requires_targets: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features", "interactions", "sequences")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = True
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = True

    def __init__(
        self,
        *,
        source: "Any | None" = None,
        source_col: "str | None" = None,
        hierarchy: "list[str] | None" = None,
        min_source_size: int = 1,
        small_source_policy: Literal["own_group", "pool", "discard"] = "own_group",
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.source = source
        self.source_col = source_col
        self.hierarchy = hierarchy
        self.min_source_size = min_source_size
        self.small_source_policy = small_source_policy
        if small_source_policy not in ("own_group", "pool", "discard"):
            raise ParameterError(f"invalid small_source_policy: {small_source_policy!r}")
        if isinstance(min_source_size, bool) or not isinstance(min_source_size, (int, np.integer)) or min_source_size < 1:
            raise ParameterError(f"min_source_size must be >= 1, got {min_source_size!r}")

    def _resolve_source(self, ctx: _Context) -> list[Any]:
        if self.source is not None:
            source = list(self.source)
        elif self.source_col is not None:
            # DataFrame column resolution is not yet wired through `_Context` (core infra only
            # threads smiles/mol/y/dates/targets/sequences today) -- documented gap, not silently
            # ignored.
            raise ParameterError(
                "SourceSplitter(source_col=...) requires DataFrame column resolution, which is "
                "not yet implemented in this build; pass `source=<sequence>` directly instead."
            )
        else:
            raise InputError("SourceSplitter requires `source` (a per-record provenance sequence)")
        if len(source) != ctx.n:
            raise InputError(f"len(source)={len(source)} does not match n={ctx.n}")
        return source

    def _group_labels(self, ctx: _Context) -> IndexArray:
        source = self._resolve_source(ctx)
        n_missing = 0
        keys: list[Any] = []
        for s in source:
            if s is None or (isinstance(s, float) and np.isnan(s)):
                keys.append("__MISSING__")
                n_missing += 1
            elif self.hierarchy is not None and isinstance(s, (list, tuple)):
                keys.append(tuple(s))
            else:
                keys.append(s)
        if n_missing:
            warn_with_details(ParseWarning(f"{n_missing} record(s) have a missing `source` value"))

        labels = dense_label_encode(keys)
        ctx.extra["_lineage_source_n_missing"] = n_missing
        ctx.extra["_lineage_source_keys"] = keys

        sizes = np.bincount(labels)
        small = np.nonzero(sizes < self.min_source_size)[0]
        if len(small) and self.small_source_policy != "own_group":
            small_mask = np.isin(labels, small)
            if self.small_source_policy == "pool":
                pooled_id = int(labels.max()) + 1
                labels = labels.copy()
                labels[small_mask] = pooled_id
                labels = dense_label_encode(labels.tolist())
            else:  # "discard"
                existing = set(ctx.extra.get("forced_discard", []))
                existing |= set(np.nonzero(small_mask)[0].tolist())
                ctx.extra["forced_discard"] = sorted(existing)
        return labels

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        sizes_arr = np.bincount(labels)
        n_sources = len(sizes_arr)
        largest_frac = float(sizes_arr.max()) / ctx.n if ctx.n else 0.0
        if largest_frac > 0.95:
            raise DegenerateGroupingError(
                f"SourceSplitter: one source holds {largest_frac:.1%} of records (> 95%)"
            )
        if largest_frac > 0.60:
            warn_with_details(
                DegenerateClusterWarning(f"SourceSplitter: largest source holds {largest_frac:.1%} of records")
            )
        return {
            "n_sources": int(n_sources),
            "source_sizes": sizes_arr.tolist(),
            "largest_source_frac": largest_frac,
            "n_missing_source": int(ctx.extra.get("_lineage_source_n_missing", 0)),
        }


# ---------------------------------------------------------------------------
# PartySplitter
# ---------------------------------------------------------------------------


class PartySplitter(GroupSplitter):
    """Partitions across data owners for federated evaluation, with deliberately non-IID parties.

    Overrides :meth:`_partition` entirely (leave-one-party-out is
    not the standard ``assign_groups`` train/valid/test bucketing every other ``GroupSplitter``
    uses).
    """

    splitter_id: ClassVar[str] = "party"
    family: ClassVar[str] = "lineage"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    requires_labels: ClassVar[bool] = False
    requires_dates: ClassVar[bool] = False
    requires_targets: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features", "interactions", "sequences")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        party: "Any | None" = None,
        n_parties: int = 3,
        synthesis: Literal["given", "dirichlet", "cluster", "label_skew"] = "given",
        dirichlet_alpha: float = 0.5,
        clusterer: "str | BaseSplitter" = "butina",
        held_out_party: "int | Literal['each']" = "each",
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.party = party
        self.n_parties = n_parties
        self.synthesis = synthesis
        self.dirichlet_alpha = dirichlet_alpha
        self.clusterer = clusterer
        self.held_out_party = held_out_party
        if synthesis not in ("given", "dirichlet", "cluster", "label_skew"):
            raise ParameterError(f"invalid synthesis: {synthesis!r}")
        if isinstance(n_parties, bool) or not isinstance(n_parties, (int, np.integer)) or n_parties < 1:
            raise ParameterError(f"n_parties must be a positive int, got {n_parties!r}")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return int(self.n_parties) if self.held_out_party == "each" else 1

    def _group_labels(self, ctx: _Context) -> IndexArray:
        if ctx.n < self.n_parties:
            raise ParameterError(f"n_parties={self.n_parties} exceeds n={ctx.n}")

        if self.synthesis == "given":
            if self.party is None:
                raise InputError("PartySplitter(synthesis='given') requires `party`")
            if len(self.party) != ctx.n:
                raise InputError(f"len(party)={len(self.party)} does not match n={ctx.n}")
            return dense_label_encode(list(self.party))

        from chemsplit.clustering import butina
        from chemsplit.featurizers import get_featurizer
        from chemsplit.metrics import pairwise_distances

        featurizer = get_featurizer("ecfp4")
        F = ctx.get_features(featurizer)
        D = pairwise_distances(F, metric="tanimoto")
        clusters = butina(D, cutoff=0.35)

        for attempt in range(10):
            if self.synthesis == "cluster":
                labels = np.empty(ctx.n, dtype=np.int64)
                for cid, members in enumerate(clusters):
                    party = cid % self.n_parties
                    for m in members:
                        labels[m] = party
            elif self.synthesis == "dirichlet":
                rng = seed_for(ctx.rng_seeds, "party.dirichlet", attempt)
                labels = np.empty(ctx.n, dtype=np.int64)
                for members in clusters:
                    props = rng.dirichlet(np.full(self.n_parties, self.dirichlet_alpha))
                    assigned = rng.choice(self.n_parties, size=len(members), p=props)
                    for m, party in zip(members, assigned, strict=True):
                        labels[m] = party
            else:  # "label_skew"
                if ctx.y is None:
                    raise InputError("PartySplitter(synthesis='label_skew') requires labels (y)")
                rng = seed_for(ctx.rng_seeds, "party.roundrobin", attempt)
                y = np.asarray(ctx.y, dtype=float)
                order = np.argsort(y, kind="stable")
                n_bins = min(self.n_parties * 4, ctx.n)
                bins = np.array_split(order, n_bins)
                labels = np.empty(ctx.n, dtype=np.int64)
                for b in bins:
                    props = rng.dirichlet(np.full(self.n_parties, self.dirichlet_alpha))
                    assigned = rng.choice(self.n_parties, size=len(b), p=props)
                    for m, party in zip(b, assigned, strict=True):
                        labels[m] = party

            labels = dense_label_encode(labels.tolist())
            if len(set(labels.tolist())) == self.n_parties:
                return labels
        raise ConstraintUnsatisfiableError(
            f"PartySplitter: could not synthesise {self.n_parties} non-empty parties in 10 attempts"
        )

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        labels = self._group_labels(ctx)
        n_parties_actual = int(labels.max()) + 1
        party_sizes = np.bincount(labels, minlength=n_parties_actual)

        if self.held_out_party == "each":
            held_list = list(range(n_parties_actual))
        else:
            if not (0 <= self.held_out_party < n_parties_actual):
                raise ParameterError(f"held_out_party={self.held_out_party!r} out of range")
            held_list = [self.held_out_party]

        overlap = self._chemical_overlap_matrix(ctx, labels, n_parties_actual)

        results = []
        for p in held_list:
            test = np.nonzero(labels == p)[0]
            train_parties = [q for q in range(n_parties_actual) if q != p]
            train = np.array(
                sorted(i for q in train_parties for i in np.nonzero(labels == q)[0].tolist()),
                dtype=np.int64,
            )
            valid = np.array([], dtype=np.int64)
            if ctx.sizes.n_valid > 0 and train_parties:
                valid_party = min(train_parties, key=lambda q: (int(party_sizes[q]), q))
                valid = np.sort(np.nonzero(labels == valid_party)[0].astype(np.int64))
                train = np.array(sorted(set(train.tolist()) - set(valid.tolist())), dtype=np.int64)

            metadata = {
                "n_parties": n_parties_actual,
                "party_sizes": party_sizes.tolist(),
                "held_out_party": int(p),
                "synthesis": self.synthesis,
                "dirichlet_alpha": self.dirichlet_alpha if self.synthesis == "dirichlet" else None,
                "per_party_label_mean": (
                    [float(np.mean(np.asarray(ctx.y, dtype=float)[labels == q])) for q in range(n_parties_actual)]
                    if ctx.y is not None
                    else None
                ),
                "chemical_overlap_matrix": overlap,
                "realised_sizes": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
            }
            results.append(
                SplitResult(
                    train=np.sort(train.astype(np.int64)),
                    valid=np.sort(valid.astype(np.int64)),
                    test=np.sort(test.astype(np.int64)),
                    discard=np.array([], dtype=np.int64),
                    groups=labels,
                    splitter_id=self.splitter_id,
                    params=self.get_params(),
                    n_records=ctx.n,
                    metadata=metadata,
                )
            )
        return results

    def _chemical_overlap_matrix(
        self, ctx: _Context, labels: IndexArray, n_parties_actual: int
    ) -> "list[list[float]] | None":
        if ctx.mols is None:
            return None
        from chemsplit.featurizers import get_featurizer
        from chemsplit.metrics import pairwise_distances

        featurizer = get_featurizer("ecfp4")
        F = ctx.get_features(featurizer)
        S = 1.0 - pairwise_distances(F, metric="tanimoto")
        mat = [[0.0] * n_parties_actual for _ in range(n_parties_actual)]
        for a in range(n_parties_actual):
            idx_a = np.nonzero(labels == a)[0]
            for b in range(n_parties_actual):
                idx_b = np.nonzero(labels == b)[0]
                if len(idx_a) == 0 or len(idx_b) == 0:
                    continue
                sub = S[np.ix_(idx_a, idx_b)]
                mat[a][b] = float(np.mean(sub.max(axis=1)))
        return mat
