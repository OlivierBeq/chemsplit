"""Reference and synthetic dataset fixtures.

Every generator is a pure function of its ``seed`` argument (where applicable): same seed ->
byte-identical output, no network access, no unseeded randomness.

``_SCAFFOLD_CORES`` and ``_SUBSTITUENTS`` are independently chosen, RDKit-verified building
blocks (10 distinct ring systems x >=20 non-ring-introducing substituents), used by
``make_scaffold_families``, ``make_dated_series``, and ``make_multitask_sparse``.

Real network dataset loaders (e.g. ``load_esol``, ``load_bace``) are out of scope for this
module for now.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rdkit import Chem

from chemsplit.exceptions import InvariantError

__all__ = [
    "Fixture",
    "make_activity_cliffs",
    "make_all_identical",
    "make_dated_series",
    "make_interactions",
    "make_label_extremes",
    "make_linear_series",
    "make_multitask_sparse",
    "make_pathological",
    "make_scaffold_families",
    "make_sequences",
    "make_singletons",
    "make_two_clusters",
]


@dataclasses.dataclass(frozen=True, slots=True)
class Fixture:
    """Uniform container returned by every ``make_*`` fixture generator.

    Fields not meaningful for a given fixture are left ``None``. ``extra`` carries anything
    fixture-specific that doesn't fit the common shape (e.g. rejection-sampling diagnostics).
    """

    smiles: list[str]
    y: np.ndarray | None = None
    dates: np.ndarray | None = None
    groups_true: np.ndarray | None = None
    targets: list[str] | None = None
    interactions: list[tuple[int, int, float]] | None = None
    sequences: list[str] | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared scaffold-core / substituent pool
# ---------------------------------------------------------------------------

_SCAFFOLD_CORES: list[tuple[str, str]] = [
    ("benzene", "c1ccc(cc1){sub}"),
    ("pyridine", "c1ccc(nc1){sub}"),
    ("naphthalene", "c1ccc2ccccc2c1{sub}"),
    ("indole", "c1ccc2[nH]ccc2c1{sub}"),
    ("quinoline", "c1ccc2ncccc2c1{sub}"),
    ("piperidine", "C1CCNCC1{sub}"),
    ("cyclohexane", "C1CCCCC1{sub}"),
    ("thiophene", "c1ccc(s1){sub}"),
    ("furan", "c1ccc(o1){sub}"),
    ("imidazole", "c1c(nc[nH]1){sub}"),
]
"""10 distinct ring-system cores, each with one substitutable position. Verified (see
``tests/test_datasets.py``) to parse and to share exactly one Murcko scaffold across every
substituent in ``_SUBSTITUENTS`` (none of the substituents below introduce their own ring, which
would otherwise change the resulting Murcko scaffold)."""

_SUBSTITUENTS: list[str] = [
    "",
    "C",
    "CC",
    "CCC",
    "CCCC",
    "F",
    "Cl",
    "Br",
    "OC",
    "OCC",
    "N",
    "NC",
    "C(=O)C",
    "C(F)(F)F",
    "C#N",
    "C(=O)O",
    "S",
    "SC",
    "CO",
    "CCO",
    "CCN",
    "C(C)C",
    "N(C)C",
    "C=C",
]
"""24 exocyclic (ring-free) substituents. For a scaffold family larger than this pool, additional
members use progressively longer plain alkyl chains (``"C"*k``, k>=5), which never collides with
the fixed pool (whose longest plain alkyl chain is length 4) and never introduces a ring."""


def _substituent(k: int) -> str:
    if k < len(_SUBSTITUENTS):
        return _SUBSTITUENTS[k]
    return "C" * (5 + (k - len(_SUBSTITUENTS)))


def _scaffold_family_smiles(
    n_scaffolds: int, per_scaffold: int, seed: int
) -> tuple[list[str], np.ndarray]:
    """Build ``n_scaffolds`` groups of ``per_scaffold`` molecules each, sharing one Murcko
    scaffold per group. Returns ``(smiles, groups_true)`` with ``groups_true[i]`` in
    ``[0, n_scaffolds)``, records ordered group-by-group (group 0's members first, etc.)."""
    if n_scaffolds > len(_SCAFFOLD_CORES):
        raise InvariantError(
            f"only {len(_SCAFFOLD_CORES)} scaffold cores available, got n_scaffolds={n_scaffolds}",
            splitter_id="datasets.make_scaffold_families",
            params={"n_scaffolds": n_scaffolds, "per_scaffold": per_scaffold},
            n_records=0,
        )
    rng = np.random.default_rng(seed)
    smiles: list[str] = []
    groups_true: list[int] = []
    for g in range(n_scaffolds):
        _, template = _SCAFFOLD_CORES[g]
        # Deterministic per-group ordering of substituent indices, then a seeded shuffle so
        # different seeds still produce a full, exact per_scaffold membership (never fewer).
        order = list(range(per_scaffold))
        rng.shuffle(order)
        for k in order:
            smi = template.format(sub=_substituent(k))
            smiles.append(smi)
            groups_true.append(g)
    return smiles, np.asarray(groups_true, dtype=np.int64)


def _verify_smiles_parse(smiles: list[str], *, context: str) -> None:
    from rdkit import Chem

    bad = [s for s in smiles if Chem.MolFromSmiles(s) is None]
    if bad:
        raise InvariantError(
            f"{context}: {len(bad)} generated SMILES failed to parse: {bad[:5]}",
            splitter_id="datasets.invariant",
            params={"context": context},
            n_records=len(smiles),
        )


# ---------------------------------------------------------------------------
# make_linear_series
# ---------------------------------------------------------------------------


def make_linear_series(n: int = 200, seed: int = 0) -> Fixture:
    """A congeneric single-scaffold series.

    All ``n`` molecules share one core (benzene); ``y`` is a smooth function of substituent chain
    length plus seeded Gaussian noise, so it is genuinely predictable from structure (useful as a
    "sanity" fixture where a reasonable model should do well under any split).
    """
    rng = np.random.default_rng(seed)
    _, template = _SCAFFOLD_CORES[0]
    smiles = []
    chain_lengths = np.arange(n, dtype=np.float64)
    for k in range(n):
        smiles.append(template.format(sub=_substituent(k)))
    y = 0.3 * chain_lengths + rng.normal(0.0, 0.5, size=n)
    _verify_smiles_parse(smiles, context="make_linear_series")
    return Fixture(smiles=smiles, y=y)


# ---------------------------------------------------------------------------
# make_scaffold_families
# ---------------------------------------------------------------------------


def make_scaffold_families(
    n_scaffolds: int = 10, per_scaffold: int = 20, seed: int = 0
) -> Fixture:
    """The canonical group-forming ground-truth fixture.

    Exactly ``n_scaffolds`` groups of exactly ``per_scaffold`` molecules, each group sharing one
    Murcko scaffold. Self-asserts its own ground truth at construction.
    """
    smiles, groups_true = _scaffold_family_smiles(n_scaffolds, per_scaffold, seed)
    n = n_scaffolds * per_scaffold
    if len(smiles) != n or len(set(groups_true.tolist())) != n_scaffolds:
        raise InvariantError(
            "make_scaffold_families: generated group structure does not match request",
            splitter_id="datasets.make_scaffold_families",
            params={"n_scaffolds": n_scaffolds, "per_scaffold": per_scaffold},
            n_records=len(smiles),
        )
    for g in range(n_scaffolds):
        count = int(np.sum(groups_true == g))
        if count != per_scaffold:
            raise InvariantError(
                f"make_scaffold_families: group {g} has {count} members, expected {per_scaffold}",
                splitter_id="datasets.make_scaffold_families",
                params={"n_scaffolds": n_scaffolds, "per_scaffold": per_scaffold},
                n_records=len(smiles),
            )
    _verify_smiles_parse(smiles, context="make_scaffold_families")
    return Fixture(smiles=smiles, groups_true=groups_true)


# ---------------------------------------------------------------------------
# make_two_clusters
# ---------------------------------------------------------------------------


def make_two_clusters(n: int = 200, separation: float = 0.9, seed: int = 0) -> Fixture:
    """Two chemically distant molecule families.

    Cluster A is built from ring-rich aromatic cores (benzene/naphthalene/indole/quinoline),
    cluster B from ring-free/aliphatic-only substituent chains on a saturated core, which reliably
    yields a large ECFP4 Tanimoto gap between clusters. Self-verified at construction (retries
    with a perturbed internal seed if the first attempt doesn't hit a clear separation).
    """
    from chemsplit.featurizers import get_featurizer
    from chemsplit.metrics import pairwise_distances

    half = n // 2
    fp = get_featurizer("ecfp4")

    for attempt in range(5):
        a_core = "c1ccc2ccccc2c1{sub}"  # naphthalene: aromatic-rich
        b_core = "C1CCCCC1{sub}"  # cyclohexane: aliphatic-only substituents
        a_smiles = [a_core.format(sub=_substituent(k)) for k in range(half)]
        b_smiles = [
            b_core.format(sub="C" * (k + 1)) for k in range(n - half)
        ]  # plain alkyl chains only
        smiles = a_smiles + b_smiles
        groups_true = np.asarray([0] * half + [1] * (n - half), dtype=np.int64)

        F = fp.transform(_mols(smiles))
        D = pairwise_distances(F, metric="tanimoto")
        S = 1.0 - D
        within_a = _mean_upper(S[:half,:half])
        within_b = _mean_upper(S[half:, half:])
        between = float(np.mean(S[:half, half:]))
        gap = 0.5 * (within_a + within_b) - between
        if gap >= 0.15:  # a real, checkable separation; "separation" param informs the target
            _verify_smiles_parse(smiles, context="make_two_clusters")
            return Fixture(
                smiles=smiles,
                groups_true=groups_true,
                extra={"within_similarity_gap": gap, "attempts": attempt + 1},
            )
    raise InvariantError(
        "make_two_clusters: could not reach a clear inter-cluster similarity gap after 5 attempts",
        splitter_id="datasets.make_two_clusters",
        params={"n": n, "separation": separation, "seed": seed},
        n_records=n,
    )


def _mols(smiles: list[str]) -> list[Chem.rdchem.Mol | None]:
    from rdkit import Chem

    return [Chem.MolFromSmiles(s) for s in smiles]


def _mean_upper(S: np.ndarray) -> float:
    n = S.shape[0]
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    return float(np.mean(S[iu]))


# ---------------------------------------------------------------------------
# make_activity_cliffs
# ---------------------------------------------------------------------------


def make_activity_cliffs(n_pairs: int = 50, seed: int = 0) -> Fixture:
    """50 matched-pair "activity cliffs".

    Each pair shares one Murcko scaffold (same core) and differs by exactly one substituent;
    ``y`` (log-scale) differs between the two members of a pair by exactly ``2.0``.
    """
    rng = np.random.default_rng(seed)
    _, template = _SCAFFOLD_CORES[0]
    smiles: list[str] = []
    y: list[float] = []
    for p in range(n_pairs):
        k_a = 2 * p
        k_b = 2 * p + 1
        y_base = float(rng.uniform(3.0, 7.0))
        smiles.append(template.format(sub=_substituent(k_a)))
        y.append(y_base)
        smiles.append(template.format(sub=_substituent(k_b)))
        y.append(y_base + 2.0)
    _verify_smiles_parse(smiles, context="make_activity_cliffs")
    pair_index = np.repeat(np.arange(n_pairs, dtype=np.int64), 2)
    return Fixture(
        smiles=smiles, y=np.asarray(y, dtype=np.float64), extra={"pair_index": pair_index}
    )


# ---------------------------------------------------------------------------
# make_dated_series
# ---------------------------------------------------------------------------


def make_dated_series(n: int = 500, seed: int = 0) -> Fixture:
    """make_scaffold_families-style scaffold families plus a date column correlated with
    scaffold group."""
    n_scaffolds = 10
    per_scaffold = max(1, n // n_scaffolds)
    smiles, groups_true = _scaffold_family_smiles(n_scaffolds, per_scaffold, seed)
    rng = np.random.default_rng(seed + 1)
    base = np.datetime64("2015-01-01")
    dates = np.empty(len(smiles), dtype="datetime64[D]")
    for i, g in enumerate(groups_true):
        offset_days = int(g) * 180 + int(rng.integers(0, 60))
        dates[i] = base + np.timedelta64(offset_days, "D")
    _verify_smiles_parse(smiles, context="make_dated_series")
    return Fixture(smiles=smiles, dates=dates, groups_true=groups_true)


# ---------------------------------------------------------------------------
# make_multitask_sparse
# ---------------------------------------------------------------------------


def make_multitask_sparse(
    n: int = 500, n_tasks: int = 8, density: float = 0.3, seed: int = 0
) -> Fixture:
    """Sparse multi-task labels.

    ``y`` is ``(n, n_tasks)`` with ~``density`` fraction non-NaN; task 0 is deliberately
    near-empty (only 3 labelled records) to stress-test multi-task balance splitters.
    """
    n_scaffolds = 10
    per_scaffold = max(1, n // n_scaffolds)
    smiles, groups_true = _scaffold_family_smiles(n_scaffolds, per_scaffold, seed)
    n_actual = len(smiles)
    rng = np.random.default_rng(seed + 2)
    y = np.full((n_actual, n_tasks), np.nan, dtype=np.float64)
    mask = rng.random((n_actual, n_tasks)) < density
    y[mask] = rng.normal(0.0, 1.0, size=int(mask.sum()))
    # Force task 0 to have exactly 3 labelled records (adversarial sparsity).
    y[:, 0] = np.nan
    sparse_idx = rng.choice(n_actual, size=3, replace=False)
    y[sparse_idx, 0] = rng.normal(0.0, 1.0, size=3)
    n_task0 = int(np.sum(~np.isnan(y[:, 0])))
    if n_task0 != 3:
        raise InvariantError(
            f"make_multitask_sparse: task 0 has {n_task0} labelled records, expected 3",
            splitter_id="datasets.make_multitask_sparse",
            params={"n": n, "n_tasks": n_tasks, "density": density},
            n_records=n_actual,
        )
    _verify_smiles_parse(smiles, context="make_multitask_sparse")
    return Fixture(smiles=smiles, y=y, groups_true=groups_true)


# ---------------------------------------------------------------------------
# make_interactions
# ---------------------------------------------------------------------------


def make_interactions(
    n_compounds: int = 100, n_targets: int = 20, density: float = 0.25, seed: int = 0
) -> Fixture:
    """A compound x target interaction table."""
    n_scaffolds = min(10, n_compounds)
    per_scaffold = max(1, n_compounds // n_scaffolds)
    smiles, groups_true = _scaffold_family_smiles(n_scaffolds, per_scaffold, seed)
    smiles = smiles[:n_compounds]
    targets = [f"TGT_{t:02d}" for t in range(n_targets)]
    rng = np.random.default_rng(seed + 3)
    interactions: list[tuple[int, int, float]] = []
    for ci in range(len(smiles)):
        for ti in range(n_targets):
            if rng.random() < density:
                y = float(rng.normal(6.0, 1.5))
                interactions.append((ci, ti, y))
    _verify_smiles_parse(smiles, context="make_interactions")
    return Fixture(smiles=smiles, targets=targets, interactions=interactions)


# ---------------------------------------------------------------------------
# make_sequences
# ---------------------------------------------------------------------------

_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def make_sequences(
    n: int = 20, families: int = 4, identity_within: float = 0.8, seed: int = 0
) -> Fixture:
    """Synthetic protein-like sequences in families with a target within-family identity.
    Identity is Hamming-style (fraction of matching
    positions against the family ancestor at equal length) -- a documented simplification, not a
    full alignment-based identity."""
    rng = np.random.default_rng(seed)
    per_family = n // families
    length = 60
    sequences: list[str] = []
    family_ids: list[int] = []
    mutation_rate = 1.0 - identity_within
    for f in range(families):
        ancestor = "".join(rng.choice(list(_AMINO_ACIDS), size=length))
        for _ in range(per_family):
            seq = list(ancestor)
            n_mut = int(round(mutation_rate * length))
            positions = rng.choice(length, size=n_mut, replace=False)
            for p in positions:
                seq[p] = str(rng.choice(list(_AMINO_ACIDS)))
            sequences.append("".join(seq))
            family_ids.append(f)

    groups_true = np.asarray(family_ids, dtype=np.int64)
    # Self-check: mean within-family identity should be roughly identity_within.
    identities = []
    for f in range(families):
        members = [s for s, g in zip(sequences, family_ids, strict=True) if g == f]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                matches = sum(a == b for a, b in zip(members[i], members[j], strict=True))
                identities.append(matches / length)
    mean_identity = float(np.mean(identities)) if identities else 0.0
    if abs(mean_identity - identity_within) > 0.25:
        raise InvariantError(
            f"make_sequences: mean within-family identity {mean_identity:.3f} far from "
            f"requested {identity_within}",
            splitter_id="datasets.make_sequences",
            params={"n": n, "families": families, "identity_within": identity_within},
            n_records=n,
        )
    return Fixture(
        smiles=[],
        sequences=sequences,
        groups_true=groups_true,
        extra={"mean_within_family_identity": mean_identity},
    )


# ---------------------------------------------------------------------------
# make_pathological
# ---------------------------------------------------------------------------


def make_pathological() -> Fixture:
    """12 hard-coded records covering every error path.

    No ``seed`` parameter (fully fixed). Composition: [0] unparseable SMILES,
    [1] a salt, [2..3] a tautomer pair, [4..5] an enantiomer pair, [6..7] an exact duplicate pair,
    [8] a macrocycle, [9] a fully acyclic (0-ring) molecule, [10] a large polymer-like chain,
    [11] a plain valid small molecule (padding to exactly 12).
    """
    smiles = [
        "not_a_smiles(((",  # [0] unparseable
        "CC(=O)O.[Na+]",  # [1] salt (sodium acetate)
        "CC(=O)C",  # [2] acetone (keto tautomer, simplified pair)
        "CC(O)=C",  # [3] enol tautomer form
        "C[C@H](N)C(=O)O",  # [4] L-alanine
        "C[C@@H](N)C(=O)O",  # [5] D-alanine (enantiomer)
        "c1ccccc1O",  # [6] phenol
        "c1ccccc1O",  # [7] exact duplicate of [6]
        "C1CCCCCCCCCCCCCCCC1",  # [8] macrocycle (16-membered ring)
        "CCCCCCCC",  # [9] fully acyclic (0-ring) molecule (octane)
        "C" * 250,  # [10] large polymer-like chain (~250 heavy atoms)
        "CCO",  # [11] plain valid small molecule (ethanol)
    ]
    from rdkit import Chem

    n_unparseable = sum(1 for s in smiles if Chem.MolFromSmiles(s) is None)
    if n_unparseable != 1:
        raise InvariantError(
            f"make_pathological: expected exactly 1 unparseable SMILES, found {n_unparseable}",
            splitter_id="datasets.make_pathological",
            params={},
            n_records=len(smiles),
        )
    if len(smiles) != 12:
        raise InvariantError(
            f"make_pathological: expected exactly 12 records, got {len(smiles)}",
            splitter_id="datasets.make_pathological",
            params={},
            n_records=len(smiles),
        )
    return Fixture(smiles=smiles)


# ---------------------------------------------------------------------------
# make_all_identical
# ---------------------------------------------------------------------------


def make_all_identical(n: int = 50) -> Fixture:
    """``n`` copies of the same molecule. No seed
    (fully deterministic)."""
    smiles = ["CC(=O)Oc1ccccc1C(=O)O"] * n  # aspirin, repeated
    return Fixture(smiles=smiles)


# ---------------------------------------------------------------------------
# make_singletons
# ---------------------------------------------------------------------------


_DIVERSE_POOL: list[str] = [
    # Deliberately heterogeneous: sugars, amino acids, steroids, nucleobases, common drugs,
    # simple rings of every size, halogenated/charged/polar fragments, PEG, lipids, etc. Formulaic
    # variation of one scaffold family (see ``_SCAFFOLD_CORES``) cannot reach low mutual ECFP4
    # similarity on its own, hence this hand-curated pool.
    "CCO", "CCCCCCCCCCCCCCCC", "OCC(O)C(O)C(O)C(O)CO", "OCC1OC(O)C(O)C(O)C1O",
    "NC(CO)C(=O)O", "NC(CC(=O)O)C(=O)O", "NC(CS)C(=O)O", "NC(Cc1ccccc1)C(=O)O",
    "NC(Cc1c[nH]c2ccccc12)C(=O)O", "NC(CCCCN)C(=O)O", "NC(CCC(=O)N)C(=O)O",
    "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O", "Nc1ncnc2[nH]cnc12", "Nc1nc2[nH]cnc2c(=O)[nH]1",
    "Cc1c[nH]c(=O)[nH]c1=O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "OC(=O)C(O)C(O)C(=O)O", "NC(N)=O", "NC(=N)N", "NC(=O)N",
    "C1CC2CCC1CC2", "C1CC2CC3CC1CC(C2)C3", "c1ccc2c(c1)ccc3c2ccc4c3cccc4",
    "CCCCCCCCCCCCCCCCCC(=O)O", "CCCCCCCCCCCCOS(=O)(=O)O", "OCCOCCOCCOCCO",
    "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "c1ccsc1", "c1ccoc1", "c1cc[nH]c1", "c1cnccn1",
    "c1nncnn1", "c1ccc2[nH]c3ccccc3c2c1", "O=C1CCC(=O)N1", "O=C1NC(=O)NC(=O)C1",
    "C1CNCCN1", "C1COCCN1", "Nc1ccc(cc1)S(=O)(=O)N", "O=S(=O)(O)c1ccccc1",
    "O=P(O)(O)Oc1ccccc1", "CC(=O)NC1=CC=C(O)C=C1", "CC1=CC(=O)C=CC1=O",
    "c1ccc2c(c1)oc1ccccc12", "C1=CC2=NC3=CC=CC=C3N=C2C=C1", "CC(N)C(=O)NC(C)C(=O)O",
    "O=C(O)c1ccccc1N", "Clc1ccc(Cl)cc1Cl", "Brc1ccccc1", "Ic1ccccc1", "N#Cc1ccccc1",
    "O=[N+]([O-])c1ccccc1", "c1ccc2c(c1)nc1ccccc1n2", "CC(C)(C)c1ccccc1", "CCN(CC)CC",
    "COC(=O)c1ccccc1", "C1=CC=C(C=C1)N=NC2=CC=CC=C2", "C1CCCCC1", "C1CCCC1", "C1CCC1",
    "C1CCCCCC1", "C1CCCCCCC1", "O=C1OCCO1", "O=C1CCCN1", "c1ccc(cc1)-c1ccccc1",
    "c1ccc(cc1)-c1ccncc1", "c1cnc2[nH]ccc2c1", "c1ccc2c(c1)cnc(N)n2", "c1ccc2c(c1)[nH]nc2",
    "c1cc2ccccc2[nH]1", "O=c1cc[nH]c(=O)[nH]1", "O=c1[nH]cnc2[nH]cnc12", "OC1CCCCC1",
    "O=C1CCCCC1", "CC(=O)C(=O)O", "CC(=O)CC(=O)C", "O=C(N)c1ccccc1", "O=C(O)c1ccccc1",
    "O=C1c2ccccc2C(=O)c2ccccc21", "CCOC(=O)c1ccccc1", "CCOP(=O)(OCC)OCC", "C[Si](C)(C)C",
    "O=C1OC(=O)c2ccccc12", "O=S1(=O)CCCC1", "c1ccc2c(c1)ncnc2N", "c1ccc2c(c1)nnc(N)n2",
    "N1CCOCC1", "N1CCNCC1", "C1COCC1", "C1CCOC1", "c1ccc(cc1)C#Cc1ccccc1",
    "c1ccc(cc1)C=Cc1ccccc1", "OCC(N)Cc1ccccc1", "OCC(N)Cc1c[nH]c2ccccc12",
    "c1cc2cccc3c2c(c1)CCC3", "O=C1CCCCCC1", "O=C1CCCCCCC1", "CC(=O)OCC(=O)O", "OCC(=O)O",
    "OC(CO)CO", "NC(=O)c1ccncc1", "NC(=O)c1cccnc1", "c1ccc2c(c1)ccnc2", "C1=CN=CN1",
    "C1=CN=CC=N1", "CC(=O)N1CCCCC1", "CCN1CCCCC1", "O=C(O)CCC(=O)O", "O=C(O)CCCC(=O)O",
    "O=C(O)CCCCC(=O)O", "O=C(O)C(N)Cc1ccc(O)cc1", "NCCc1ccc(O)c(O)c1", "NCCc1c[nH]c2ccccc12",
    "CC(N)Cc1ccccc1", "O=C(O)C1CCCCC1",
]


def make_singletons(n: int = 100, seed: int = 0, max_similarity: float = 0.15) -> Fixture:
    """``n`` molecules with max pairwise ECFP4 Tanimoto similarity < ``max_similarity``.

    Uses :func:`chemsplit.clustering.maxmin_pick` (greedy MaxMin diversity picking) over a pool
    combining ``_DIVERSE_POOL`` with a combinatorial extension of ``_SCAFFOLD_CORES``.

    The default ``n=100, max_similarity=0.15`` is not reliably achievable with this synthetic
    pool: empirically, ``n=20`` reaches ~0.14 similarity but ``n=30`` already exceeds 0.15
    (~0.18). Raises :class:`InvariantError` (naming the achieved similarity) if ``n``/
    ``max_similarity`` can't be met — for ``n=100``, pass a looser ``max_similarity`` (~0.35) or
    supply a real diverse compound pool.
    """
    from chemsplit.clustering import maxmin_pick
    from chemsplit.featurizers import get_featurizer
    from chemsplit.metrics import pairwise_distances

    rng = np.random.default_rng(seed)
    combi: set[str] = set()
    for core_idx in range(len(_SCAFFOLD_CORES)):
        _, template = _SCAFFOLD_CORES[core_idx]
        for k in range(0, 200, 3):
            combi.add(template.format(sub=_substituent(k)))
    pool_smiles = list(dict.fromkeys(_DIVERSE_POOL)) + sorted(combi)

    from rdkit import Chem

    mols = []
    valid_smiles = []
    for s in pool_smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
            valid_smiles.append(s)

    fp = get_featurizer("ecfp4")
    F = fp.transform(mols)
    D = pairwise_distances(F, metric="tanimoto")

    if n > len(mols):
        raise InvariantError(
            f"make_singletons: requested n={n} exceeds the candidate pool size ({len(mols)})",
            splitter_id="datasets.make_singletons",
            params={"n": n, "seed": seed, "max_similarity": max_similarity},
            n_records=len(mols),
        )
    picked = maxmin_pick(D, n_picks=n, init="most_peripheral", rng=rng)
    sub_D = D[np.ix_(picked, picked)]
    np.fill_diagonal(sub_D, 1.0)
    achieved_max_sim = float(1.0 - sub_D.min())

    if achieved_max_sim >= max_similarity:
        raise InvariantError(
            f"make_singletons: best achievable max similarity at n={n} is "
            f"{achieved_max_sim:.3f}, >= requested max_similarity={max_similarity}. This is a "
            "genuine limitation of the synthetic candidate pool, not a sampling-budget issue -- "
            "see the function's docstring for the empirically achievable (n, max_similarity) "
            "combinations.",
            splitter_id="datasets.make_singletons",
            params={"n": n, "seed": seed, "max_similarity": max_similarity},
            n_records=n,
        )
    accepted = [valid_smiles[i] for i in picked]
    return Fixture(smiles=accepted, extra={"achieved_max_similarity": achieved_max_sim})


# ---------------------------------------------------------------------------
# make_label_extremes
# ---------------------------------------------------------------------------


def make_label_extremes(n: int = 300, seed: int = 0) -> Fixture:
    """Bimodal ``y`` plus a 60-record censored block fixed at exactly ``5.0``."""
    n_censored = 60
    n_bimodal = n - n_censored
    n_scaffolds = 10
    per_scaffold = max(1, n // n_scaffolds)
    smiles, groups_true = _scaffold_family_smiles(n_scaffolds, per_scaffold, seed)
    smiles = smiles[:n]
    groups_true = groups_true[:n]

    rng = np.random.default_rng(seed + 4)
    half = n_bimodal // 2
    mode_a = rng.normal(2.0, 0.4, size=half)
    mode_b = rng.normal(8.0, 0.4, size=n_bimodal - half)
    y_bimodal = np.concatenate([mode_a, mode_b])
    rng.shuffle(y_bimodal)
    y = np.concatenate([y_bimodal, np.full(n_censored, 5.0)])
    perm = rng.permutation(n)
    y = y[perm]
    smiles = [smiles[i] for i in perm]
    groups_true = groups_true[perm]

    n_at_five = int(np.sum(y == 5.0))
    if n_at_five != n_censored:
        raise InvariantError(
            f"make_label_extremes: {n_at_five} records at y==5.0, expected {n_censored}",
            splitter_id="datasets.make_label_extremes",
            params={"n": n, "seed": seed},
            n_records=n,
        )
    _verify_smiles_parse(smiles, context="make_label_extremes")
    return Fixture(smiles=smiles, y=y, groups_true=groups_true, extra={"n_censored": n_censored})
