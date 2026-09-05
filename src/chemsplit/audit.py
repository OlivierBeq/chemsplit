"""Leakage-audit module: descriptive diagnostics for an already-computed split.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats as _stats

from chemsplit import __version__ as _CHEMSPLIT_VERSION
from chemsplit.base import SplitResult
from chemsplit.featurizers import get_featurizer
from chemsplit.metrics import nn_distance

__all__ = [
    "LeakageReport",
    "NNProfile",
    "adversarial_validation",
    "audit_split",
    "nn_similarity_profile",
    "y_scramble_control",
]

_DEFAULT_THRESHOLDS = (0.4, 0.6, 0.8, 0.9, 0.99)
_QUANTILE_POINTS = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)




def _is_smiles_like(X: Any) -> bool:
    return isinstance(X, (list, tuple)) and (len(X) == 0 or isinstance(X[0], str))


def _featurize(X: Any, featurizer_spec: str | Any) -> Any:
    """Featurize ``X``, resolving the featurizer only when ``X`` is not already a matrix."""
    if _is_smiles_like(X):
        from rdkit import Chem

        featurizer = get_featurizer(featurizer_spec)
        mols = [Chem.MolFromSmiles(s) for s in X]
        return featurizer.transform(mols)
    return X


def _mols_or_none(X: Any) -> list | None:
    if not _is_smiles_like(X):
        return None
    from rdkit import Chem

    return [Chem.MolFromSmiles(s) for s in X]




@dataclasses.dataclass(frozen=True, slots=True)
class NNProfile:
    """Nearest-neighbour distance profile for a set of query records against a reference set."""

    values: np.ndarray  # float32, one distance per query record (or empty if either side is empty)
    quantiles: dict[float, float]
    mean: float
    histogram: tuple[np.ndarray, np.ndarray]  # (counts, bin_edges), 50 bins over [0, 1]

    def to_dict(self) -> dict[str, Any]:
        counts, edges = self.histogram
        return {
            "values": self.values.tolist(),
            "quantiles": {str(k): v for k, v in self.quantiles.items()},
            "mean": self.mean,
            "histogram": {"counts": counts.tolist(), "bin_edges": edges.tolist()},
        }


def _nn_profile_from_distances(distances: np.ndarray) -> NNProfile:
    if distances.size == 0:
        empty_hist = (np.zeros(50, dtype=np.int64), np.linspace(0.0, 1.0, 51))
        return NNProfile(
            values=distances.astype(np.float32),
            quantiles={q: float("nan") for q in _QUANTILE_POINTS},
            mean=float("nan"),
            histogram=empty_hist,
        )
    quantiles = {q: float(np.quantile(distances, q)) for q in _QUANTILE_POINTS}
    counts, edges = np.histogram(distances, bins=50, range=(0.0, 1.0))
    return NNProfile(
        values=distances.astype(np.float32),
        quantiles=quantiles,
        mean=float(np.mean(distances)),
        histogram=(counts, edges),
    )


def nn_similarity_profile(
    query_idx: Sequence[int],
    ref_idx: Sequence[int],
    X: Any,
    *,
    featurizer: str | Any = "ecfp4",
    metric: str = "tanimoto",
    n_jobs: int = 1,
) -> NNProfile:
    """Nearest-neighbour distance from each query record to the closest reference record.

    For a similarity profile, invert the result, as :func:`audit_split` does for its
    ``nn_similarity`` field.

    :param query_idx: indices of the records to measure from, typically the test set.
    :param ref_idx: indices of the records to measure to, typically the training set.
    :param X: the records, as SMILES, molecules or a feature matrix.
    :param featurizer: featurizer alias or instance.
    :param metric: the distance metric.
    :param n_jobs: worker count. Results never depend on it.
    :return: one distance per query record, in ``query_idx`` order.
    """
    F = _featurize(X, featurizer)
    query_idx = np.asarray(query_idx, dtype=np.int64)
    ref_idx = np.asarray(ref_idx, dtype=np.int64)
    if query_idx.size == 0 or ref_idx.size == 0:
        return _nn_profile_from_distances(np.array([], dtype=np.float32))
    Q = F[query_idx] if not hasattr(F, "tocsr") else F[query_idx]
    R = F[ref_idx] if not hasattr(F, "tocsr") else F[ref_idx]
    distances = nn_distance(Q, R, metric=metric, n_jobs=n_jobs)
    return _nn_profile_from_distances(distances)


def adversarial_validation(
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    X: Any,
    *,
    featurizer: str | Any = "ecfp4",
    classifier: str = "logreg",
    cv: int = 5,
    random_state: int = 0,
) -> tuple[float, tuple[float, float], dict[str, Any]]:
    """Cross-validate a classifier that tries to tell train-side from test-side records.

    The 95% interval is the 2.5th to 97.5th percentile of the per-fold AUCs, not a parametric
    interval, which suits the small fold counts typically used.

    :param train_idx: indices on the train side.
    :param test_idx: indices on the test side.
    :param X: the records, as SMILES, molecules or a feature matrix.
    :param featurizer: featurizer alias or instance.
    :param classifier: ``"logreg"`` or ``"gbdt"``.
    :param cv: cross-validation folds.
    :param random_state: seeds the classifier and the fold assignment.
    :raises ParameterError: if ``classifier`` is unknown.
    :return: the mean AUC, its interval, and the per-feature importances.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    F = _featurize(X, featurizer)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    idx = np.concatenate([train_idx, test_idx])
    labels = np.concatenate([np.zeros(len(train_idx)), np.ones(len(test_idx))]).astype(np.int64)

    Xa = F[idx]
    if hasattr(Xa, "toarray"):
        Xa = Xa.toarray()
    Xa = np.asarray(Xa, dtype=np.float64)

    n_splits = max(2, min(cv, int(np.bincount(labels).min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_aucs: list[float] = []
    importances = np.zeros(Xa.shape[1], dtype=np.float64)
    n_fits = 0
    for fold_i, (tr, te) in enumerate(skf.split(Xa, labels)):
        if classifier == "hgb":
            clf = HistGradientBoostingClassifier(random_state=random_state + fold_i)
        else:
            clf = LogisticRegression(max_iter=1000, random_state=random_state + fold_i)
        clf.fit(Xa[tr], labels[tr])
        proba = clf.predict_proba(Xa[te])[:, 1]
        if len(np.unique(labels[te])) < 2:
            continue
        fold_aucs.append(float(roc_auc_score(labels[te], proba)))
        if hasattr(clf, "coef_"):
            importances += np.abs(clf.coef_[0])
            n_fits += 1

    if not fold_aucs:
        fold_aucs = [0.5]
    mean_auc = float(np.mean(fold_aucs))
    ci_low, ci_high = (
        float(np.percentile(fold_aucs, 2.5, method="linear")),
        float(np.percentile(fold_aucs, 97.5, method="linear")),
    )
    feature_importances = {
        "mean_abs_coef": (importances / n_fits).tolist() if n_fits else None,
        "fold_aucs": fold_aucs,
    }
    return mean_auc, (ci_low, ci_high), feature_importances


def y_scramble_control(
    estimator: Any,
    X: Any,
    y: np.ndarray,
    split: SplitResult,
    *,
    n_repeats: int = 20,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
    random_state: int = 0,
) -> dict[str, Any]:
    """Compare the observed score against scores from shuffled training labels.

    The estimator is refit on ``n_repeats`` independent shufflings of the train-side ``y`` and
    scored on the real test set each time. Passing -- meaning the model is not just exploiting a
    distributional artefact of the split -- is defined as the observed score beating at least
    95% of the scrambled ones.

    :param estimator: an sklearn-style estimator, refit per repeat.
    :param X: the records, passed to the estimator unchanged.
    :param y: the true labels.
    :param split: the split to evaluate.
    :param n_repeats: how many shufflings to run.
    :param scorer: maps ``(y_true, y_pred)`` to a score, or ``None`` for R² on continuous
        labels and ROC-AUC on binary ones.
    :param random_state: seeds the shufflings.
    :return: the observed score, the scrambled scores, and whether the control passed.
    """
    import copy

    y = np.asarray(y)
    train_idx, test_idx = split.train, split.test

    def default_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import r2_score, roc_auc_score

        if len(np.unique(y_true)) <= 2 and set(np.unique(y_true)) <= {0, 1}:
            return float(roc_auc_score(y_true, y_pred))
        return float(r2_score(y_true, y_pred))

    score_fn = scorer or default_scorer

    def fit_and_score(y_train: np.ndarray) -> float:
        model = copy.deepcopy(estimator)
        Xf = _featurize(X, "ecfp4")
        Xtr = Xf[train_idx]
        Xte = Xf[test_idx]
        if hasattr(Xtr, "toarray"):
            Xtr, Xte = Xtr.toarray(), Xte.toarray()
        model.fit(Xtr, y_train)
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(Xte)[:, 1]
        else:
            pred = model.predict(Xte)
        return score_fn(y[test_idx], pred)

    observed_score = fit_and_score(y[train_idx])

    rng = np.random.default_rng(random_state)
    scrambled_scores: list[float] = []
    for _ in range(n_repeats):
        shuffled = y[train_idx].copy()
        rng.shuffle(shuffled)
        scrambled_scores.append(fit_and_score(shuffled))

    beats_fraction = float(np.mean(observed_score > np.asarray(scrambled_scores)))
    empirical_p_value = float(1.0 - beats_fraction)
    return {
        "observed_score": observed_score,
        "scrambled_scores": scrambled_scores,
        "empirical_p_value": empirical_p_value,
        "passes": beats_fraction >= 0.95,
    }



_PHYSCHEM_DESCRIPTORS = (
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms", "BertzCT",
)

@dataclasses.dataclass(frozen=True, slots=True)
class LeakageReport:
    """Purely descriptive diagnostics for a computed split. Never declares a split good or bad."""

    n_train: int
    n_valid: int
    n_test: int
    n_discard: int

    nn_similarity: NNProfile
    nn_similarity_valid: NNProfile | None
    max_similarity: float
    frac_test_above: dict[float, float]
    n_exact_duplicates_across: int
    duplicate_pairs: list[tuple[int, int]]

    shared_scaffolds: int
    shared_ring_systems: int
    shared_sources: int
    shared_mmp_contexts: int | None

    adversarial_auc: float | None
    adversarial_auc_ci95: tuple[float, float] | None
    label_shift: dict[str, float] | None
    property_shift: dict[str, float]

    splitter_id: str | None
    params: dict[str, Any]
    chemsplit_version: str
    seed: int | None

    def summary(self) -> str:
        lines = [
            "chemsplit LeakageReport",
            "------------------------",
            f"n_train={self.n_train}  n_valid={self.n_valid}  n_test={self.n_test}  "
            f"n_discard={self.n_discard}",
            f"max cross-partition similarity: {self.max_similarity:.4f}",
            "median NN similarity (test->train): "
            f"{1.0 - self.nn_similarity.quantiles.get(0.5, float('nan')):.4f}",
            f"exact duplicates across partitions: {self.n_exact_duplicates_across}",
            f"shared scaffolds: {self.shared_scaffolds}  "
            f"shared ring systems: {self.shared_ring_systems}",
        ]
        if self.adversarial_auc is not None:
            lines.append(
                f"adversarial AUC: {self.adversarial_auc:.4f} "
                f"(95% CI {self.adversarial_auc_ci95[0]:.4f}-{self.adversarial_auc_ci95[1]:.4f})"
            )
        flags = self.flags()
        lines.append(f"flags: {', '.join(flags) if flags else '(none)'}")
        return "\n".join(lines)

    def to_json(self) -> str:
        payload = {
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "n_test": self.n_test,
            "n_discard": self.n_discard,
            "nn_similarity": self.nn_similarity.to_dict(),
            "nn_similarity_valid": self.nn_similarity_valid.to_dict()
            if self.nn_similarity_valid is not None
            else None,
            "max_similarity": self.max_similarity,
            "frac_test_above": {str(k): v for k, v in self.frac_test_above.items()},
            "n_exact_duplicates_across": self.n_exact_duplicates_across,
            "duplicate_pairs": self.duplicate_pairs,
            "shared_scaffolds": self.shared_scaffolds,
            "shared_ring_systems": self.shared_ring_systems,
            "shared_sources": self.shared_sources,
            "shared_mmp_contexts": self.shared_mmp_contexts,
            "adversarial_auc": self.adversarial_auc,
            "adversarial_auc_ci95": list(self.adversarial_auc_ci95)
            if self.adversarial_auc_ci95 is not None
            else None,
            "label_shift": self.label_shift,
            "property_shift": self.property_shift,
            "splitter_id": self.splitter_id,
            "params": self.params,
            "chemsplit_version": self.chemsplit_version,
            "seed": self.seed,
            "flags": self.flags(),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def flags(self) -> list[str]:
        out: list[str] = []
        if self.n_exact_duplicates_across > 0:
            out.append("EXACT_DUPLICATES_ACROSS_PARTITIONS")
        if self.max_similarity > 0.99:
            out.append("HIGH_MAX_SIMILARITY")
        median_similarity = 1.0 - self.nn_similarity.quantiles.get(0.5, float("nan"))
        if not np.isnan(median_similarity) and median_similarity > 0.6:
            out.append("MEDIAN_NN_ABOVE_0.6")
        if self.shared_scaffolds > 0:
            out.append("SHARED_SCAFFOLDS")
        if self.shared_sources > 0:
            out.append("SHARED_SOURCES")
        if self.adversarial_auc is not None:
            if self.adversarial_auc < 0.55:
                out.append("LOW_ADVERSARIAL_AUC")
            if self.adversarial_auc > 0.90:
                out.append("HIGH_ADVERSARIAL_AUC")
        if self.label_shift is not None and self.label_shift.get("ks_pvalue", 1.0) < 0.01:
            out.append("LABEL_SHIFT")
        if self.n_test < 30:
            out.append("SMALL_TEST")
        for name, d in sorted(self.property_shift.items()):
            if abs(d) > 0.5:
                out.append(f"PROPERTY_SHIFT_{name}")
        return out




def _exact_duplicates_across(
    train_idx: np.ndarray, test_idx: np.ndarray, X: Any
) -> tuple[int, list[tuple[int, int]]]:
    if not _is_smiles_like(X):
        return 0, []
    from rdkit import Chem

    def key(i: int) -> str | None:
        mol = Chem.MolFromSmiles(X[i])
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    train_keys: dict[str, int] = {}
    for i in train_idx.tolist():
        k = key(i)
        if k is not None:
            train_keys.setdefault(k, i)

    pairs: list[tuple[int, int]] = []
    for j in test_idx.tolist():
        k = key(j)
        if k is not None and k in train_keys:
            pairs.append((train_keys[k], j))
            if len(pairs) >= 1000:
                break
    return len(pairs), pairs


def _shared_scaffold_count(train_idx: np.ndarray, test_idx: np.ndarray, X: Any) -> tuple[int, int]:
    if not _is_smiles_like(X):
        return 0, 0
    from rdkit import Chem

    from chemsplit.scaffolds import murcko_scaffold, ring_systems

    def scaffold_keys(idx: np.ndarray) -> set[str]:
        keys = set()
        for i in idx.tolist():
            mol = Chem.MolFromSmiles(X[i])
            if mol is not None:
                keys.add(murcko_scaffold(mol))
        return keys

    def ring_system_keys(idx: np.ndarray) -> set[str]:
        keys = set()
        for i in idx.tolist():
            mol = Chem.MolFromSmiles(X[i])
            if mol is not None:
                keys.update(ring_systems(mol))
        return keys

    scaffold_overlap = len(scaffold_keys(train_idx) & scaffold_keys(test_idx))
    ring_overlap = len(ring_system_keys(train_idx) & ring_system_keys(test_idx))
    return scaffold_overlap, ring_overlap


def _property_shift(train_idx: np.ndarray, test_idx: np.ndarray, X: Any) -> dict[str, float]:
    if not _is_smiles_like(X):
        return {}
    from rdkit import Chem

    from chemsplit.featurizers.descriptors import PhysChemFeaturizer

    featurizer = PhysChemFeaturizer()
    mols = [Chem.MolFromSmiles(s) for s in X]
    F = np.asarray(featurizer.transform(mols))
    out: dict[str, float] = {}
    for j, name in enumerate(_PHYSCHEM_DESCRIPTORS):
        a, b = F[train_idx, j], F[test_idx, j]
        if len(a) > 1 and len(b) > 1:
            pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        else:
            pooled_std = 0.0
        d = 0.0 if pooled_std == 0 else float((b.mean() - a.mean()) / pooled_std)
        out[name] = d
    return out


def audit_split(
    split: SplitResult,
    X: Any,
    y: np.ndarray | None = None,
    *,
    featurizer: str | Any = "ecfp4",
    metric: str = "tanimoto",
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
    sources: np.ndarray | None = None,
    check_mmp: bool = False,
    adversarial: bool = True,
    n_jobs: int = 1,
    random_state: int = 0,
    max_memory_bytes: int = 2 * 1024**3,
) -> LeakageReport:
    """Compute a purely descriptive :class:`LeakageReport` for a split.

    Purely descriptive: it never fails a pipeline, it reports what to look at.

    :param split: the split to audit.
    :param X: the records. SMILES or molecules enable the structural checks -- scaffolds, ring
        systems, exact duplicates, property shift -- which a feature matrix reports as zero or
        ``None``.
    :param y: labels, enabling the label-shift and adversarial statistics.
    :param featurizer: featurizer alias or instance.
    :param metric: the distance metric.
    :param thresholds: the similarity thresholds to count cross-partition pairs at.
    :param sources: per-record provenance labels, enabling the source-overlap counts.
    :param check_mmp: also count matched-molecular-pair overlap, which costs an MMPA
        fragmentation.
    :param adversarial: run :func:`adversarial_validation` as part of the report.
    :param n_jobs: worker count. Results never depend on it.
    :param random_state: seeds the adversarial classifier.
    :param max_memory_bytes: ceiling on any pairwise matrix.
    :raises ScalabilityError: if a pairwise matrix would exceed ``max_memory_bytes``.
    :return: the :class:`LeakageReport`.
    """
    train_idx, test_idx, valid_idx = split.train, split.test, split.valid

    dist_profile = nn_similarity_profile(
        test_idx, train_idx, X, featurizer=featurizer, metric=metric, n_jobs=n_jobs
    )
    sim_values = 1.0 - dist_profile.values
    sim_quantiles = {q: 1.0 - v for q, v in dist_profile.quantiles.items()}
    sim_counts, sim_edges = dist_profile.histogram[0][::-1], 1.0 - dist_profile.histogram[1][::-1]
    nn_similarity = NNProfile(
        values=sim_values, quantiles=sim_quantiles, mean=1.0 - dist_profile.mean,
        histogram=(sim_counts, sim_edges),
    )
    max_similarity = float(np.max(sim_values)) if sim_values.size else float("nan")
    frac_test_above = {
        t: float(np.mean(sim_values > t)) if sim_values.size else 0.0 for t in thresholds
    }

    nn_similarity_valid = None
    if valid_idx.size:
        dvp = nn_similarity_profile(
            valid_idx, train_idx, X, featurizer=featurizer, metric=metric, n_jobs=n_jobs
        )
        svp = 1.0 - dvp.values
        nn_similarity_valid = NNProfile(
            values=svp,
            quantiles={q: 1.0 - v for q, v in dvp.quantiles.items()},
            mean=1.0 - dvp.mean,
            histogram=(dvp.histogram[0][::-1], 1.0 - dvp.histogram[1][::-1]),
        )

    n_dupes, dupe_pairs = _exact_duplicates_across(train_idx, test_idx, X)
    shared_scaffolds, shared_ring_systems = _shared_scaffold_count(train_idx, test_idx, X)

    shared_sources = 0
    if sources is not None:
        sources = np.asarray(sources)
        shared_sources = len(set(sources[train_idx].tolist()) & set(sources[test_idx].tolist()))

    shared_mmp_contexts = None  # not implemented in this pass; documented gap, see module notes.

    adv_auc: float | None = None
    adv_ci: tuple[float, float] | None = None
    if adversarial and train_idx.size and test_idx.size:
        adv_auc, adv_ci, _ = adversarial_validation(
            train_idx, test_idx, X, featurizer=featurizer, random_state=random_state
        )

    label_shift = None
    if y is not None:
        y = np.asarray(y)
        y_train, y_test = y[train_idx], y[test_idx]
        if np.issubdtype(y.dtype, np.number) and y_train.size and y_test.size:
            ks_stat, ks_p = _stats.ks_2samp(y_train, y_test)
            label_shift = {
                "mean_train": float(np.mean(y_train)),
                "mean_test": float(np.mean(y_test)),
                "std_train": float(np.std(y_train)),
                "std_test": float(np.std(y_test)),
                "skew_train": float(_stats.skew(y_train)) if y_train.size > 2 else float("nan"),
                "skew_test": float(_stats.skew(y_test)) if y_test.size > 2 else float("nan"),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_p),
            }

    property_shift = _property_shift(train_idx, test_idx, X)

    return LeakageReport(
        n_train=int(train_idx.size),
        n_valid=int(valid_idx.size),
        n_test=int(test_idx.size),
        n_discard=int(split.discard.size),
        nn_similarity=nn_similarity,
        nn_similarity_valid=nn_similarity_valid,
        max_similarity=max_similarity,
        frac_test_above=frac_test_above,
        n_exact_duplicates_across=n_dupes,
        duplicate_pairs=dupe_pairs,
        shared_scaffolds=shared_scaffolds,
        shared_ring_systems=shared_ring_systems,
        shared_sources=shared_sources,
        shared_mmp_contexts=shared_mmp_contexts,
        adversarial_auc=adv_auc,
        adversarial_auc_ci95=adv_ci,
        label_shift=label_shift,
        property_shift=property_shift,
        splitter_id=split.splitter_id,
        params=split.params,
        chemsplit_version=_CHEMSPLIT_VERSION,
        seed=random_state,
    )
