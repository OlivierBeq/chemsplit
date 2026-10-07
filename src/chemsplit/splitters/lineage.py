"""``lineage`` splitter family: date-based and provenance-based splitters.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

import numpy as np

from chemsplit._fp_similarity import (
    blocked_max_similarity_to,
    compute_neighbor_lists,
    dense_matrix_fits,
)
from chemsplit._ga import mutate_bits
from chemsplit._unionfind import dense_label_encode
from chemsplit.base import BaseSplitter, GroupSplitter, SplitResult, Strictness, _Context
from chemsplit.clustering import EPS as _CLUSTER_EPS
from chemsplit.clustering import butina_from_neighbors
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

__all__ = [
    "FidelitySplitter",
    "PartySplitter",
    "SIMPDSplitter",
    "SourceSplitter",
    "TemporalSplitter",
]

_EXPORTED = __all__

_EPS = 1e-9




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


def _parse_offset_days(spec: str | int) -> int:
    """Best-effort parse of a pandas-offset-like string (``"90D"``, ``"6M"``, ``"1Y"``) or a
    plain integer day count, into an integer number of days.

    Simplification, documented: month/year units are approximated as 30/365 days respectively
    rather than resolved against a calendar (pandas' ``DateOffset`` calendar arithmetic is not
    used here to keep window arithmetic index-free and trivially vectorisable). Good enough for
    the windowing granularity this splitter targets; exact calendar arithmetic can be substituted
    later without changing the public API.
    """
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        return int(spec)
    if not isinstance(spec, str):
        raise ParameterError(f"expected a pandas-offset-like string or int, got {spec!r}")
    m = re.fullmatch(r"\s*(\d+)\s*([DWMY])\s*", spec.upper())
    if not m:
        raise ParameterError(
            f"could not parse offset string {spec!r} (expected e.g. '90D', '6M', '1Y')"
        )
    count, unit = int(m.group(1)), m.group(2)
    days_per_unit = {"D": 1, "W": 7, "M": 30, "Y": 365}[unit]
    return count * days_per_unit


class TemporalSplitter(BaseSplitter):
    """Date-cut split: train on the past, test on the future.

    :param cut_date: the train/test boundary, as an ISO string or ``numpy.datetime64``.
        ``None`` places the cut at the quantile implied by the resolved sizes.
    :param valid_cut_date: an earlier boundary carving a validation band out of train.
    :param embargo: a gap after the cut, as a pandas-like offset string or days. Records
        inside it are discarded, closing the leak from assays that report months late.
    :param mode: one cut, a rolling fixed-width window, or an expanding window.
    :param n_windows: how many windows the rolling and expanding modes produce.
    :param window: window width for those modes, as an offset string or days.
    :param tie_policy: where records dated exactly on the cut go.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if an offset string cannot be parsed, ``n_windows`` is below 1, or
        ``mode`` or ``tie_policy`` is unknown.
    :raises ConfigurationError: if ``cut_date`` is combined with a windowed mode, or
        ``valid_cut_date`` is not earlier than ``cut_date``.
    :raises InputError: at split time, if ``dates`` is missing or the wrong length.

    Advantages
    ----------
    - The closest available proxy for prospective performance: it reproduces the entangled
      correlations between chemistry, assay protocol, project goals and era.
    - Needs no featurization and no parameters beyond the cut date.
    - `embargo` closes the look-ahead leak from assays reporting months after registration,
      which a naive date cut cannot see.
    - `mode="rolling"` and `"expanding"` produce a performance-over-time curve, which is what a
      maintenance decision needs.

    Pitfalls
    --------
    - Confounds chemistry, assay protocol, target selection and data volume, so a drop shows
      that the model degrades, not *why*. Good for realism, bad for attribution.
    - Dates are frequently wrong: registration, first-test, publication and deposition dates
      differ, and datasets mix them. The splitter cannot detect that.
    - Public datasets rarely carry usable timestamps, and ChEMBL document years are coarse and
      often unrepresentative of when the work happened. `simpd` covers the no-dates case.
    - One contiguous era makes the test set chemically homogeneous with correlated errors, so
      a single cut carries wide implicit uncertainty. `mode="rolling"` averages over several.
    - Without `embargo`, slow-reporting assays leak the future into training.
    - In `"expanding"` mode the training set grows with time, so later windows are not
      comparable to earlier ones without normalising for `n_train`.

    References
    ----------
    .. [1] Sheridan, R. P. Time-Split Cross-Validation as a Method for Estimating the Goodness of
       Prospective Prediction. *J. Chem. Inf. Model.* **2013**, 53 (4), 783-790.
       https://doi.org/10.1021/ci400084k
    .. [2] Mayr, A.; Klambauer, G.; Unterthiner, T. et al. Large-Scale Comparison of Machine
       Learning Methods for Drug Target Prediction on ChEMBL. *Chem. Sci.* **2018**, 9 (24),
       5441-5451. https://doi.org/10.1039/C8SC00148K
    .. [3] ``embargo`` and the rolling and expanding window modes are standard time-series
       cross-validation practice rather than a cheminformatics method.
    """

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
        cut_date: str | np.datetime64 | None = None,
        valid_cut_date: str | np.datetime64 | None = None,
        embargo: str | int = 0,
        mode: Literal["single", "rolling", "expanding"] = "single",
        n_windows: int = 5,
        window: str | int = "365D",
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
        if (
            isinstance(n_windows, bool)
            or not isinstance(n_windows, (int, np.integer))
            or n_windows < 1
        ):
            raise ParameterError(f"n_windows must be a positive int, got {n_windows!r}")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Report how many splits will be yielded.

        :param X: ignored, as are ``y`` and ``groups``; the signature is sklearn\'s.
        :return: ``n_windows`` in a windowed mode, else ``1``.
        """
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
        after_cut = dates > cut
        if self.tie_policy == "train":
            train_mask = train_mask | tie_mask
            always_discard = np.zeros(n, dtype=bool)
            discard_eligible = after_cut
        elif self.tie_policy == "discard":
            always_discard = tie_mask
            discard_eligible = after_cut
        else:  # "test": ties are eligible for test, subject to the embargo like after-cut records
            always_discard = np.zeros(n, dtype=bool)
            discard_eligible = after_cut | tie_mask

        discard_mask = always_discard | (discard_eligible & (dates < embargo_end))
        test_mask = (~train_mask) & (~discard_mask) & discard_eligible

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
            "train_date_range": (
                [str(dates[train].min()), str(dates[train].max())] if len(train) else None
            ),
            "test_date_range": (
                [str(dates[test].min()), str(dates[test].max())] if len(test) else None
            ),
            "n_ties_at_cut": n_ties,
            "n_embargoed": int(np.sum(after_cut & (dates < embargo_end))),
            "window_index": None,
            "realised_sizes": {
                "train": int(len(train)),
                "valid": int(len(valid)),
                "test": int(len(test)),
            },
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
        target_train = ctx.sizes.n_train + ctx.sizes.n_valid
        if n_ties > 0 and abs(len(train) - target_train) > max(10, 0.01 * ctx.n):
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
            upper = (dates <= test_end) if is_last else (dates < test_end)
            test_mask = (dates >= test_start) & upper
            if self.mode == "rolling":
                train_start = test_start - np.timedelta64(window_days, "D")
                train_mask = (dates >= train_start) & (dates < test_start)
            else:  # "expanding"
                train_mask = dates < test_start
            train = np.nonzero(train_mask)[0]
            test = np.nonzero(test_mask)[0]
            if len(train) == 0 or len(test) == 0:
                continue
            # Records outside this fold's (train_start, test_end] window play no part in it,
            # and every record has to land somewhere, so the rest of the timeline is discarded
            # for this fold only.
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


class SIMPDSplitter(BaseSplitter):
    """Simulated time split: a multi-objective GA rearranges an undated dataset until the
    train/test pair reproduces the descriptor/property shifts measured in real time splits.

    DEAP's variation operators read Python's global ``random`` module, which this library
    never touches, so only its RNG-free parts are used: ``creator``/``base.Fitness``
    bookkeeping and ``tools.selNSGA2``, a deterministic rank and crowding-distance sort.
    Crossover, mutation and tournament selection run off a dedicated ``random.Random`` from
    :func:`chemsplit.determinism.seeded_python_random`. Mutation is shared with
    :class:`~chemsplit.splitters.task.AVESplitter` through :mod:`chemsplit._ga`; repair,
    crossover and selection stay separate.

    :param targets: the descriptor and property shifts to aim for, or ``None`` for the
        published medians.
    :param descriptors: which RDKit descriptors the objectives are computed over.
    :param population_size: GA population size.
    :param n_generations: how many generations to run.
    :param crossover_prob: probability of crossing a selected pair.
    :param mutation_prob: probability of mutating an individual.
    :param mutation_indpb: per-gene flip probability within a mutated individual.
    :param tournament_size: tournament size for parent selection.
    :param cluster_for_g_sim: clusterer, by registry id or instance, behind the group-similarity
        objective.
    :param early_stop_patience: generations without improvement before stopping early.
    :param max_memory_bytes: ceiling on the pairwise matrix.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises MissingDependencyError: if the ``ga`` extra is not installed.
    :raises ParameterError: if a GA parameter is out of range, ``targets`` names an unknown
        objective, or ``descriptors`` names an unknown descriptor.
    :raises ScalabilityError: at split time, if the pairwise matrix would exceed
        ``max_memory_bytes``.

    Advantages
    ----------
    - Makes a time-like evaluation possible on the many public datasets that lack usable
      dates.
    - The objectives are explicit and measurable, so the resemblance to a real time split can
      be checked by comparing `metadata["achieved"]` against `targets`.
    - Multi-objective optimisation surfaces trade-offs instead of collapsing them into one
      hand-weighted score.

    Pitfalls
    --------
    - **Simulates the statistics of a time split, not time itself.** Protocol drift, changing
      project goals and real unforeseeability are not reproduced, so a model can score well
      here and still fail prospectively.
    - The default targets are medians from one study of specific internal datasets, not
      universal constants, so carrying them to another therapeutic area is an assumption.
    - Expensive: hundreds of generations over hundreds of individuals, each needing
      nearest-neighbour statistics.
    - Conflicting objectives make the GA stochastic: different seeds give different splits at
      similar objective values, pinned down by the seed and `metadata["achieved"]`.
    - Optimising against label-derived objectives such as `delta_active_frac` uses the labels
      to design the experiment.

    References
    ----------
    .. [1] Landrum, G. A.; Beckers, M.; Lanini, J.; Schneider, N.; Stiefl, N.; Riniker, S.
       SIMPD: An Algorithm for Generating Simulated Time Splits for Validating Machine Learning
       Approaches. *J. Cheminform.* **2023**, 15, 119.
       https://doi.org/10.1186/s13321-023-00787-9
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
        targets: dict[str, float] | None = None,
        descriptors: tuple[str,...] = _DEFAULT_DESCRIPTORS,
        population_size: int = 500,
        n_generations: int = 200,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        mutation_indpb: float = 0.02,
        tournament_size: int = 3,
        cluster_for_g_sim: str | BaseSplitter = "butina",
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

        # Clusters for frac_test_in_train_cluster. The default "butina" calls the
        # chemsplit.clustering.butina primitive directly, at ButinaSplitter's own default
        # cutoff, rather than going through the registry.
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
                        if len(a) > 1 and len(b) > 1:
                            pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
                        else:
                            pooled_std = 1.0
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
            deap_creator.create(
                "FitnessSIMPD", deap_base.Fitness, weights=(1.0,) * len(target_keys)
            )
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
            return repair(mutate_bits(a, self.mutation_indpb, py_rng))

        def tournament(pop: list) -> Any:
            contestants = [pop[py_rng.randrange(len(pop))] for _ in range(self.tournament_size)]
            return max(contestants, key=lambda ind: ind.fitness.values)

        population = [new_individual() for _ in range(self.population_size)]
        best_sq_err = None
        stall = 0
        hypervolume_trace: list[float] = []

        def sq_err(ind: Any) -> float:
            return float(sum(v * v for v in ind.fitness.values))

        # gen is read after the loop, as generations_run
        for gen in range(int(self.n_generations)):  # noqa: B007
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
        achieved = dict(zip(target_keys, observed.tolist(), strict=True))
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
            "realised_sizes": {
                "train": int(len(train)),
                "valid": int(len(valid)),
                "test": int(len(test)),
            },
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


class SourceSplitter(GroupSplitter):
    """Groups by provenance: document, assay, lab, vendor, plate, or any caller-supplied key.

    :param source: per-record source labels, or ``None`` to read them from ``source_col``.
    :param source_col: column name to take the labels from when ``X`` is a frame.
    :param hierarchy: column names from coarsest to finest, so that the grouping key is the
        tuple of levels present.
    :param min_source_size: sources smaller than this are handled by
        ``small_source_policy``.
    :param small_source_policy: keep each tiny source as its own group, pool them into one, or
        discard them.
    :param base: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``min_source_size`` is below 1, or ``small_source_policy`` is
        unknown.
    :raises ConfigurationError: if neither or both of ``source`` and ``source_col`` are given.
    :raises InputError: at split time, if the labels are the wrong length or the column is
        absent.

    Advantages
    ----------
    - Catches a leak scaffold and cluster splits both miss: one publication contributing a
      congeneric series under one protocol with one systematic offset, each learnable.
    - Needs no chemistry, featurization or seed. It is a metadata join.
    - Composes with `leave_one_cluster_out` for a per-laboratory error profile, often the most
      actionable diagnostic available.

    Pitfalls
    --------
    - Source metadata is often wrong or missing, and a `NaN`-heavy column degenerates toward
      a random split; `metadata["n_missing_source"]` counts the gaps.
    - Source sizes are very skewed -- a few large campaigns, a long tail of two-compound
      papers -- so the ratio drifts and `SizeToleranceWarning` is expected.
    - Grouping by source does **not** guarantee chemical separation, since two labs can
      publish the same series. `group_k_fold` over a merged grouping covers both axes.
    - The chosen hierarchy level changes the experiment: assay-level grouping is much weaker
      than document-level, which is weaker than lab-level.

    References
    ----------
    .. [1] Grouping by provenance is generic. The inter-source noise and leakage it targets
       are measured in [2]-[4].
    .. [2] Landrum, G. A.; Riniker, S. Combining IC50 or Ki Values from Different Sources Is a
       Source of Significant Noise. *J. Chem. Inf. Model.* **2024**, 64 (5), 1560-1567.
       https://doi.org/10.1021/acs.jcim.4c00049
    .. [3] Kramer, C.; Kalliokoski, T.; Gedeck, P.; Vulpetti, A. The Experimental Uncertainty
       of Heterogeneous Public Ki Data. *J. Med. Chem.* **2012**, 55 (11), 5165-5173.
       https://doi.org/10.1021/jm300131x
    .. [4] Kalliokoski, T.; Kramer, C.; Vulpetti, A.; Gedeck, P. Comparability of Mixed IC50
       Data -- A Statistical Analysis. *PLoS ONE* **2013**, 8 (4), e61007.
       https://doi.org/10.1371/journal.pone.0061007
    """

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
        source: Any | None = None,
        source_col: str | None = None,
        hierarchy: list[str] | None = None,
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
        if (
            isinstance(min_source_size, bool)
            or not isinstance(min_source_size, (int, np.integer))
            or min_source_size < 1
        ):
            raise ParameterError(f"min_source_size must be >= 1, got {min_source_size!r}")

    def _resolve_source(self, ctx: _Context) -> list[Any]:
        if self.source is not None:
            source = list(self.source)
        elif self.source_col is not None:
            # _Context doesn't thread DataFrame columns, only
            # smiles/mol/y/dates/targets/sequences, so source_col can't be resolved here
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
                DegenerateClusterWarning(
                    f"SourceSplitter: largest source holds {largest_frac:.1%} of records"
                )
            )
        return {
            "n_sources": int(n_sources),
            "source_sizes": sizes_arr.tolist(),
            "largest_source_frac": largest_frac,
            "n_missing_source": int(ctx.extra.get("_lineage_source_n_missing", 0)),
        }


class FidelitySplitter(BaseSplitter):
    """Train on low-fidelity measurements, test on the highest fidelity (``fidelity``).

    Records carry a fidelity level, e.g. ``"HTS"`` < ``"confirmatory"`` < ``"dose-response"``,
    or a numeric tier. Whole levels are assigned from the highest down -- test first, then
    valid, with the lowest left for train -- so realised sizes follow level boundaries.
    Discarding structure leakage matches molecules by canonical SMILES and keeps only each
    one's highest-fidelity record, so the model never trains on a cheap label of a test
    molecule. Fully deterministic.

    :param fidelity: per-record fidelity levels. Required.
    :param levels: the levels ordered from lowest to highest fidelity, or ``None`` to sort the
        observed values. Every record's level must appear.
    :param structure_leakage: discard a molecule's lower-fidelity records, or allow them.
        Discarding needs molecular input.
    :param size_tolerance: how far a realised partition fraction may miss its target before a
        :class:`SizeToleranceWarning` is issued.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :ivar splitter_id: ``"fidelity"``.
    :raises ParameterError: if ``structure_leakage`` is unknown, or ``size_tolerance`` is
        outside ``[0, 1)``.
    :raises ConfigurationError: if ``fidelity`` is missing, or ``structure_leakage="discard"``
        is combined with non-molecular input.
    :raises InputError: at split time, if the fidelity labels are the wrong length or name a
        level outside ``levels``.
    :raises EmptyPartitionError: at split time, if fewer than two levels survive.

    Advantages
    ----------
    - Measures the realistic multi-fidelity case: learn from abundant cheap data, predict the
      scarce expensive measurement.
    - Whole levels stay together, so assay-protocol artefacts specific to one level cannot leak
      across the split.
    - `structure_leakage="discard"` removes the most common multi-fidelity leak, a compound's
      own low-fidelity label sitting in train.
    - No seed, and no free parameter beyond the level order.

    Pitfalls
    --------
    - **Sizes are coarse.** With few levels the realised sizes can sit far from the targets;
      `metadata["level_partition"]` shows where they landed.
    - Confounds the label's fidelity with the chemistry measured at each level, since
      high-fidelity assays run on already-optimised compounds. The drop is real, its cause
      is not isolated.
    - Low- and high-fidelity labels may be on different scales, or be different quantities
      altogether. The split does not harmonise them.
    - Needs at least two levels, and `structure_leakage="discard"` can empty train when most
      molecules are measured at every level.

    References
    ----------
    .. [1] Buterez, D.; Janet, J. P.; Kiddle, S. J.; Oglic, D.; Liò, P. Transfer Learning with
       Graph Neural Networks for Improved Molecular Property Prediction in the Multi-Fidelity
       Setting. *Nat. Commun.* **2024**, 15, 1517.
       https://doi.org/10.1038/s41467-024-45566-8
    """

    splitter_id: ClassVar[str] = "fidelity"
    family: ClassVar[str] = "lineage"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = False
    requires_dates: ClassVar[bool] = False
    requires_targets: ClassVar[bool] = False
    accepts: ClassVar[tuple[str, ...]] = ("smiles", "mol", "features", "interactions", "sequences")
    extras: ClassVar[tuple[str, ...]] = ()
    deterministic_without_seed: ClassVar[bool] = True
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = True

    def __init__(
        self,
        *,
        fidelity: Any | None = None,
        levels: Any | None = None,
        structure_leakage: Literal["discard", "allow"] = "discard",
        size_tolerance: float = 0.05,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.fidelity = fidelity
        self.levels = levels
        self.structure_leakage = structure_leakage
        self.size_tolerance = size_tolerance
        if structure_leakage not in ("discard", "allow"):
            raise ParameterError(f"invalid structure_leakage: {structure_leakage!r}")
        if not (isinstance(size_tolerance, (int, float)) and 0 <= size_tolerance < 1):
            raise ParameterError(f"size_tolerance must be in [0, 1), got {size_tolerance!r}")
        if levels is not None and len(set(levels)) != len(list(levels)):
            raise ParameterError("levels must not contain duplicates")

    def _level_ranks(self, ctx: _Context) -> tuple[list[Any], np.ndarray]:
        if self.fidelity is None:
            raise InputError(
                "FidelitySplitter requires `fidelity` (a per-record fidelity level sequence)"
            )
        fidelity = list(self.fidelity)
        if len(fidelity) != ctx.n:
            raise InputError(f"len(fidelity)={len(fidelity)} does not match n={ctx.n}")
        if self.levels is not None:
            levels = list(self.levels)
            unknown = sorted({str(v) for v in fidelity} - {str(v) for v in levels})
            if unknown:
                raise ParameterError(f"fidelity values not listed in levels: {unknown}")
        else:
            try:
                levels = sorted(set(fidelity))
            except TypeError:
                raise ParameterError(
                    "fidelity values are not mutually orderable; pass `levels`"
                ) from None
        rank = {str(level): r for r, level in enumerate(levels)}
        return levels, np.asarray([rank[str(v)] for v in fidelity], dtype=np.int64)

    def _structure_keys(self, ctx: _Context) -> list[str]:
        if ctx.mols is not None:
            from rdkit import Chem

            return [
                Chem.MolToSmiles(m) if m is not None else f"<unparsed:{i}>"
                for i, m in enumerate(ctx.mols)
            ]
        if ctx.smiles is not None:
            return list(ctx.smiles)
        raise ParameterError(
            "FidelitySplitter(structure_leakage='discard') needs SMILES or molecules to match "
            "structures; pass structure_leakage='allow' for other inputs"
        )

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        n = ctx.n
        levels, ranks = self._level_ranks(ctx)
        present = sorted(set(ranks.tolist()))
        if len(present) < 2:
            raise ConstraintUnsatisfiableError(
                f"FidelitySplitter: all {n} records share one fidelity level"
            )
        counts = {r: int(np.sum(ranks == r)) for r in present}
        level_partition: dict[int, str] = {}
        filled = {"test": 0, "valid": 0}
        targets = {"test": ctx.sizes.n_test, "valid": ctx.sizes.n_valid}
        for r in reversed(present):
            if r == present[0]:
                level_partition[r] = "train"
            elif filled["test"] < targets["test"] or not filled["test"]:
                level_partition[r] = "test"
                filled["test"] += counts[r]
            elif filled["valid"] < targets["valid"]:
                level_partition[r] = "valid"
                filled["valid"] += counts[r]
            else:
                level_partition[r] = "train"
        part = np.asarray([level_partition[int(r)] for r in ranks], dtype=object)
        discard = np.zeros(n, dtype=bool)
        if self.structure_leakage == "discard":
            keys = self._structure_keys(ctx)
            later = {"train": ("valid", "test"), "valid": ("test",)}
            for lower, uppers in later.items():
                held = {keys[i] for i in range(n) if part[i] in uppers}
                discard |= (part == lower) & np.asarray([k in held for k in keys], dtype=bool)
        buckets = {
            name: np.flatnonzero((part == name) & ~discard).astype(np.int64)
            for name in ("train", "valid", "test")
        }
        if buckets["train"].size == 0:
            raise EmptyPartitionError(
                "FidelitySplitter: train is empty after removing shared structures"
            )
        realised = {k: int(v.size) for k, v in buckets.items()}
        wanted = {"train": ctx.sizes.n_train, "valid": ctx.sizes.n_valid, "test": ctx.sizes.n_test}
        if any(abs(realised[k] - wanted[k]) / n > self.size_tolerance for k in wanted):
            from chemsplit.exceptions import SizeToleranceWarning

            warn_with_details(
                SizeToleranceWarning(
                    f"FidelitySplitter: realised sizes {realised} differ from targets "
                    f"{wanted} by more than size_tolerance={self.size_tolerance} (whole "
                    "fidelity levels are kept together)",
                    details={"realised": realised, "targets": wanted},
                )
            )
        result = SplitResult(
            train=buckets["train"],
            valid=buckets["valid"],
            test=buckets["test"],
            discard=np.flatnonzero(discard).astype(np.int64),
            groups=None,
            splitter_id=self.splitter_id,
            params=self.get_params(),
            n_records=n,
            metadata={
                "level_partition": {str(levels[r]): level_partition[r] for r in present},
                "n_structure_discards": int(discard.sum()),
                "realised_sizes": realised,
            },
        )
        return [result]


class PartySplitter(GroupSplitter):
    """Partitions across data owners for federated evaluation, with deliberately non-IID parties.

    Leave-one-party-out is not the usual ``assign_groups`` bucketing, so :meth:`_partition` is
    overridden entirely.

    :param party: per-record owner labels, required by ``synthesis="given"``.
    :param n_parties: how many parties to synthesise.
    :param synthesis: use the supplied labels, or synthesise parties by a Dirichlet draw over
        clusters, by whole clusters, or by label skew.
    :param dirichlet_alpha: concentration of the Dirichlet draw. Lower is more non-IID.
    :param clusterer: clusterer, by registry id or instance, behind the cluster-based synthesis
        modes.
    :param held_out_party: which party to hold out, or ``"each"`` for one fold per party.
    :param base: forwarded to :class:`chemsplit.base.GroupSplitter`.
    :raises ParameterError: if ``n_parties`` is below 2, ``dirichlet_alpha`` is not positive,
        ``held_out_party`` is out of range, or ``synthesis`` is unknown.
    :raises ConfigurationError: if ``synthesis="given"`` without ``party``, or ``party`` is
        supplied alongside another mode.
    :raises LabelError: at split time, if ``synthesis="label_skew"`` and ``y`` is missing.

    Advantages
    ----------
    - The only way to check whether a federated or consortium model helps *each* participant
      rather than only the largest contributor.
    - The synthesis modes let a public dataset stand in for a consortium at an explicit,
      tunable non-IID severity.
    - `chemical_overlap_matrix` quantifies how different the parties really are, which is the
      precondition for interpreting any federated result.

    Pitfalls
    --------
    - A pooled average across parties is dominated by the largest one and hides a model that is
      useless to everyone else. Per-party scores, weighted by `party_sizes`, are the result.
    - Synthesised parties model heterogeneity rather than reproducing it. Real pharma datasets
      differ in assay protocol and target selection, not just chemistry.
    - `dirichlet_alpha` has no natural value, and results at `alpha=0.1` and `alpha=1.0` are
      not comparable.
    - Party sizes are usually very unequal, so held-out fold sizes vary enormously.
    - Says nothing about the privacy properties of the training scheme. This is a data split,
      not a privacy guarantee.

    References
    ----------
    .. [1] Hsu, T.-M. H.; Qi, H.; Brown, M. Measuring the Effects of Non-Identical Data
       Distribution for Federated Visual Classification. arXiv preprint, **2019**. No DOI;
       https://arxiv.org/abs/1909.06335 (origin of the Dirichlet(alpha) non-IID partition
       convention that ``synthesis="dirichlet"`` and ``dirichlet_alpha`` follow)
    .. [2] Heyndrickx, W.; Mervin, L.; Morawietz, T. et al. MELLODDY: Cross-Pharma Federated
       Learning at Unprecedented Scale Unlocks Benefits in QSAR without Compromising
       Proprietary Information. *J. Chem. Inf. Model.* **2024**, 64 (7), 2331-2344.
       https://doi.org/10.1021/acs.jcim.3c00799
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
        party: Any | None = None,
        n_parties: int = 3,
        synthesis: Literal["given", "dirichlet", "cluster", "label_skew"] = "given",
        dirichlet_alpha: float = 0.5,
        clusterer: str | BaseSplitter = "butina",
        held_out_party: int | Literal['each'] = "each",
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
        if (
            isinstance(n_parties, bool)
            or not isinstance(n_parties, (int, np.integer))
            or n_parties < 1
        ):
            raise ParameterError(f"n_parties must be a positive int, got {n_parties!r}")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Report how many splits will be yielded.

        :param X: ignored, as are ``y`` and ``groups``; the signature is sklearn\'s.
        :return: the party count when ``held_out_party="each"``, else ``1``.
        """
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
        # Butina reads only radius-neighbour lists, which stream; the dense matrix is only built
        # when it fits and is cheaper.
        if dense_matrix_fits(ctx.n, 2 * 1024**3):
            F = ctx.get_features(featurizer)
            clusters = butina(pairwise_distances(F, metric="tanimoto"), cutoff=0.35)
        else:
            clusters = butina_from_neighbors(
                compute_neighbor_lists(
                    ctx, featurizer, "tanimoto", 0.35, eps=_CLUSTER_EPS, n_jobs=self.n_jobs
                ),
                ctx.n,
                reorder=False,
            )

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
                # parties are atomic, so take the one closest in size to valid_size
                target_valid = ctx.sizes.n_valid
                valid_party = min(
                    train_parties,
                    key=lambda q: (abs(int(party_sizes[q]) - target_valid), int(party_sizes[q]), q),
                )
                valid = np.sort(np.nonzero(labels == valid_party)[0].astype(np.int64))
                train = np.array(sorted(set(train.tolist()) - set(valid.tolist())), dtype=np.int64)

            metadata = {
                "n_parties": n_parties_actual,
                "party_sizes": party_sizes.tolist(),
                "held_out_party": int(p),
                "synthesis": self.synthesis,
                "dirichlet_alpha": self.dirichlet_alpha if self.synthesis == "dirichlet" else None,
                "per_party_label_mean": (
                    [
                        float(np.mean(np.asarray(ctx.y, dtype=float)[labels == q]))
                        for q in range(n_parties_actual)
                    ]
                    if ctx.y is not None
                    else None
                ),
                "chemical_overlap_matrix": overlap,
                "realised_sizes": {
                "train": int(len(train)),
                "valid": int(len(valid)),
                "test": int(len(test)),
            },
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
    ) -> list[list[float]] | None:
        if ctx.mols is None:
            return None
        from chemsplit.featurizers import get_featurizer
        from chemsplit.metrics import pairwise_distances

        featurizer = get_featurizer("ecfp4")
        dense = dense_matrix_fits(ctx.n, 2 * 1024**3)
        S = None
        if dense:
            S = 1.0 - pairwise_distances(ctx.get_features(featurizer), metric="tanimoto")
        mat = [[0.0] * n_parties_actual for _ in range(n_parties_actual)]
        for a in range(n_parties_actual):
            idx_a = np.nonzero(labels == a)[0]
            for b in range(n_parties_actual):
                idx_b = np.nonzero(labels == b)[0]
                if len(idx_a) == 0 or len(idx_b) == 0:
                    continue
                # a per-row maximum, so blocking the rows is exact
                row_max = (
                    S[np.ix_(idx_a, idx_b)].max(axis=1)
                    if S is not None
                    else blocked_max_similarity_to(
                        ctx, featurizer, "tanimoto", idx_b, rows=idx_a, n_jobs=self.n_jobs
                    )
                )
                mat[a][b] = float(np.mean(row_max))
        return mat
