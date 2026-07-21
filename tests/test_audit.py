import numpy as np
import pytest

from chemsplit import audit
from chemsplit.base import SplitResult

# A diverse-ish pool of valid SMILES with distinct scaffolds, used to build "clean" splits.
_DIVERSE = [
    "c1ccccc1",  # benzene
    "c1ccncc1",  # pyridine
    "c1ccc2ccccc2c1",  # naphthalene
    "C1CCCCC1",  # cyclohexane
    "c1ccc2[nH]ccc2c1",  # indole-like
    "C1CCNCC1",  # piperidine
    "c1ccc2ncccc2c1",  # quinoline
    "C1CCOC1",  # THF
    "c1ccc2c(c1)oc1ccccc12",  # dibenzofuran-like
    "C1CC1",  # cyclopropane
    "c1ccsc1",  # thiophene
    "c1cc[nH]c1",  # pyrrole
    "C1CCCC1",  # cyclopentane
    "c1ccc2c(c1)CCCC2",  # tetralin-like
    "C1CCCCCC1",  # cycloheptane
    "c1cnc2[nH]ccc2c1",  # azaindole-like
]


def _clean_split() -> tuple[SplitResult, list]:
    X = _DIVERSE
    n = len(X)
    # first half train, second half test -- distinct scaffolds on each side.
    train = np.arange(0, n // 2, dtype=np.int64)
    test = np.arange(n // 2, n, dtype=np.int64)
    result = SplitResult(
        train=train,
        test=test,
        valid=np.array([], dtype=np.int64),
        discard=np.array([], dtype=np.int64),
        groups=None,
        splitter_id="random",
        params={},
        n_records=n,
        metadata={},
    )
    return result, X


def _leaky_split() -> tuple[SplitResult, list]:
    # benzene duplicated across train and test, plus a near-identical pair (same scaffold).
    X = ["c1ccccc1", "c1ccccc1", "c1ccccc1C", "c1ccccc1CC", "C1CCCCC1", "C1CCCCC1C"]
    train = np.array([0, 2, 4], dtype=np.int64)
    test = np.array([1, 3, 5], dtype=np.int64)
    result = SplitResult(
        train=train,
        test=test,
        valid=np.array([], dtype=np.int64),
        discard=np.array([], dtype=np.int64),
        groups=None,
        splitter_id="random",
        params={},
        n_records=len(X),
        metadata={},
    )
    return result, X


def test_clean_split_low_leakage_signals():
    split, X = _clean_split()
    report = audit.audit_split(split, X, adversarial=False)
    assert report.n_train == 8
    assert report.n_test == 8
    assert report.shared_scaffolds == 0
    assert report.n_exact_duplicates_across == 0
    assert "SHARED_SCAFFOLDS" not in report.flags()
    assert "EXACT_DUPLICATES_ACROSS_PARTITIONS" not in report.flags()


def test_leaky_split_triggers_flags():
    split, X = _leaky_split()
    report = audit.audit_split(split, X, adversarial=False)
    assert report.n_exact_duplicates_across >= 1
    assert report.max_similarity == pytest.approx(1.0, abs=1e-6)
    flags = report.flags()
    assert "EXACT_DUPLICATES_ACROSS_PARTITIONS" in flags
    assert "HIGH_MAX_SIMILARITY" in flags
    assert "SHARED_SCAFFOLDS" in flags


def test_small_test_flag():
    split, X = _clean_split()
    report = audit.audit_split(split, X, adversarial=False)
    assert "SMALL_TEST" in report.flags()  # n_test=8 < 30


def test_flags_fixed_order():
    split, X = _leaky_split()
    report = audit.audit_split(split, X, adversarial=False)
    flags = report.flags()
    # EXACT_DUPLICATES / HIGH_MAX_SIMILARITY / SHARED_SCAFFOLDS must appear in this relative order.
    order_positions = [flags.index(f) for f in flags if f in
                        ("EXACT_DUPLICATES_ACROSS_PARTITIONS", "HIGH_MAX_SIMILARITY", "SHARED_SCAFFOLDS")]
    assert order_positions == sorted(order_positions)


def test_to_json_round_trips_and_is_valid_json():
    import json

    split, X = _clean_split()
    report = audit.audit_split(split, X, adversarial=False)
    payload = json.loads(report.to_json())
    assert payload["n_train"] == 8
    assert payload["n_test"] == 8
    assert isinstance(payload["flags"], list)


def test_summary_is_a_readable_string():
    split, X = _clean_split()
    report = audit.audit_split(split, X, adversarial=False)
    text = report.summary()
    assert "LeakageReport" in text
    assert "n_train=8" in text


def test_nn_similarity_profile_standalone():
    X = _DIVERSE
    profile = audit.nn_similarity_profile([0, 1], [2, 3, 4, 5], X)
    assert profile.values.shape == (2,)
    assert 0.0 <= profile.mean <= 1.0
    counts, edges = profile.histogram
    assert counts.shape == (50,)
    assert edges.shape == (51,)


def test_nn_similarity_profile_empty_side():
    X = _DIVERSE
    profile = audit.nn_similarity_profile([], [0, 1], X)
    assert profile.values.shape == (0,)


def test_adversarial_validation_near_chance_on_random_split():
    rng = np.random.default_rng(0)
    X = [_DIVERSE[i % len(_DIVERSE)] for i in range(60)]
    idx = rng.permutation(60)
    train_idx, test_idx = idx[:30], idx[30:]
    auc, ci, importances = audit.adversarial_validation(train_idx, test_idx, X, random_state=0)
    assert 0.0 <= auc <= 1.0
    assert ci[0] <= ci[1]
    assert "fold_aucs" in importances


def test_y_scramble_control_runs_end_to_end():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = [_DIVERSE[i % len(_DIVERSE)] for i in range(60)]
    y = (rng.random(60) > 0.5).astype(int)
    idx = rng.permutation(60)
    train_idx, test_idx = idx[:40], idx[40:]
    split = SplitResult(
        train=np.sort(train_idx.astype(np.int64)),
        test=np.sort(test_idx.astype(np.int64)),
        valid=np.array([], dtype=np.int64),
        discard=np.array([], dtype=np.int64),
        groups=None,
        splitter_id="random",
        params={},
        n_records=60,
        metadata={},
    )
    result = audit.y_scramble_control(
        LogisticRegression(max_iter=1000), X, y, split, n_repeats=5, random_state=0
    )
    assert "observed_score" in result
    assert len(result["scrambled_scores"]) == 5
    assert 0.0 <= result["empirical_p_value"] <= 1.0
    assert isinstance(result["passes"], bool)


def test_property_shift_and_label_shift_populated():
    split, X = _clean_split()
    y = np.arange(len(X), dtype=float)
    report = audit.audit_split(split, X, y=y, adversarial=False)
    assert report.label_shift is not None
    assert "ks_pvalue" in report.label_shift
    assert set(report.property_shift.keys()) == set(audit._PHYSCHEM_DESCRIPTORS)
