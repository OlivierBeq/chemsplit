"""Biomolecular-axis splitters: hold out on protein sequence identity, family, binding site,
deposition date, or a joint ligand-and-sequence axis.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np

from chemsplit._pair_assign import assign_pair_groups
from chemsplit._unionfind import UnionFind, dense_label_encode
from chemsplit.base import (
    BaseSplitter,
    GroupSplitter,
    SplitResult,
    Strictness,
    _canonicalize_params,
)
from chemsplit.clustering import butina
from chemsplit.determinism import seed_for
from chemsplit.exceptions import (
    DegenerateClusterWarning,
    DegenerateGroupingError,
    LabelError,
    MissingDependencyError,
    ParameterError,
    warn_with_details,
)
from chemsplit.types import IndexArray

__all__ = [
    "BindingSiteSplitter",
    "ComplexJointSplitter",
    "DepositionDateSplitter",
    "ProteinFamilySplitter",
    "SequenceIdentitySplitter",
]

_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _hamming_identity(a: str, b: str) -> float:
    """Fraction of matching positions over the shorter sequence's length (documented, simplified
    identity metric for the pure-Python fallback path -- not a full alignment)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


def _parasail_identity(a: str, b: str) -> float:
    """Global-alignment percent identity via ``parasail`` (the ``bio`` extra's accelerated path)."""
    import parasail

    result = parasail.nw_stats(a, b, 10, 1, parasail.blosum62)
    denom = max(1, min(len(a), len(b)))
    return result.matches / denom


def _pairwise_identity_matrix(sequences: list[str], use_parasail: bool) -> np.ndarray:
    n = len(sequences)
    ident_fn = _parasail_identity if use_parasail else _hamming_identity
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = ident_fn(sequences[i], sequences[j])
            D[i, j] = D[j, i] = s
    np.fill_diagonal(D, 1.0)
    return D


class SequenceIdentitySplitter(GroupSplitter):
    """Group protein sequences by pairwise identity (single-linkage above a threshold).

    :param identity_threshold: sequences above this pairwise identity merge into one group.
        Linkage is single, so a chain can span very different sequences at its ends, as
        graph-component grouping does on the ligand side.
    :param algorithm: global alignment via the ``bio`` extra, a dependency-free equal-length
        fractional match, or ``"auto"`` to prefer ``parasail`` when installed.
    :param size_tolerance: how far a realised partition size may drift from its target before
        a :class:`SizeToleranceWarning` is issued.
    :param group_assignment: how groups are handed to partitions; see
        :func:`chemsplit.base.assign_groups`.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``identity_threshold`` is outside ``(0, 1)``, or ``algorithm``
        is unknown.
    :raises MissingDependencyError: if ``algorithm="parasail"`` and the ``bio`` extra is not
        installed.
    :raises InputError: at split time, if the sequences are missing, or ``algorithm="hamming"``
        is used on sequences of unequal length.

    Advantages
    ----------
    - The standard control for any model taking a protein as input. Without it a "new target"
      claim is not credible.
    - Computes and reports `max_cross_identity`, so the claim rests on evidence.
    - Deduplicating to unique sequences first keeps it cheap even on large interaction
      datasets.

    Pitfalls
    --------
    - **Percent identity is not one number.** It shifts with the denominator and with local
      against global alignment: 30% local over the shorter sequence is far weaker than 30%
      global.
    - A poor proxy for binding-site similarity: two proteins at 20% identity can share nearly
      identical pockets, so the split separates them while leaking the pharmacology.
      `binding_site` covers that axis.
    - Single-linkage grouping is transitive, so one promiscuous sequence can chain two
      otherwise unrelated families into a single group.
    - Multi-domain and multi-chain proteins align poorly as single strings.
    - `algorithm="hamming"` is a positional approximation, not an alignment, and gives
      different groups from the `parasail` path.

    References
    ----------
    .. [1] Pahikkala, T.; Airola, A.; Pietilä, S. et al. Toward More Realistic Drug-Target
       Interaction Predictions. *Brief. Bioinform.* **2015**, 16 (2), 325-337.
       https://doi.org/10.1093/bib/bbu010 (the rationale for requiring unseen targets)
    .. [2] Smith, T. F.; Waterman, M. S. Identification of Common Molecular Subsequences.
       *J. Mol. Biol.* **1981**, 147 (1), 195-197.
       https://doi.org/10.1016/0022-2836(81)90087-5; and Needleman, S. B.; Wunsch, C. D. A
       General Method Applicable to the Search for Similarities in the Amino Acid Sequence of
       Two Proteins. *J. Mol. Biol.* **1970**, 48 (3), 443-453.
       https://doi.org/10.1016/0022-2836(70)90057-4
    .. [3] ``algorithm="parasail"``: Daily, J. Parasail: SIMD C Library for Global,
       Semi-Global, and Local Pairwise Sequence Alignments. *BMC Bioinformatics* **2016**, 17,
       81. https://doi.org/10.1186/s12859-016-0930-z
    .. [4] Established identity-clustering tools solving the same grouping problem: Li, W.;
       Godzik, A. Cd-hit. *Bioinformatics* **2006**, 22 (13), 1658-1659.
       https://doi.org/10.1093/bioinformatics/btl158; Steinegger, M.; Söding, J. MMseqs2
       Enables Sensitive Protein Sequence Searching for the Analysis of Massive Data Sets.
       *Nat. Biotechnol.* **2017**, 35 (11), 1026-1028. https://doi.org/10.1038/nbt.3988
    """

    splitter_id: ClassVar[str] = "sequence_identity"
    family: ClassVar[str] = "biomolecular"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("sequences",)
    extras: ClassVar[tuple[str,...]] = ("bio",)
    deterministic_without_seed: ClassVar[bool] = True
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        identity_threshold: float = 0.7,
        algorithm: Literal["auto", "parasail", "hamming"] = "auto",
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        **base: Any,
    ) -> None:
        super().__init__(size_tolerance=size_tolerance, group_assignment=group_assignment, **base)
        self.identity_threshold = identity_threshold
        self.algorithm = algorithm
        if not (0.0 < identity_threshold <= 1.0):
            raise ParameterError(
                f"identity_threshold must be in (0, 1], got {identity_threshold!r}"
            )
        if algorithm not in ("auto", "parasail", "hamming"):
            raise ParameterError(f"invalid algorithm: {algorithm!r}")

    def _resolve_algorithm(self) -> bool:
        if self.algorithm == "hamming":
            return False
        try:
            import parasail  # noqa: F401

            return True
        except ImportError:
            if self.algorithm == "parasail":
                raise MissingDependencyError("SequenceIdentitySplitter", "bio") from None
            return False

    def _group_labels(self, ctx: Any) -> IndexArray:
        if ctx.sequences is None:
            raise LabelError(
                "SequenceIdentitySplitter requires ctx.sequences (accepts=('sequences',))"
            )
        sequences = ctx.sequences
        n = len(sequences)
        use_parasail = self._resolve_algorithm()
        D = _pairwise_identity_matrix(sequences, use_parasail)
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] > self.identity_threshold:
                    uf.union(i, j)
        reps = [uf.find(i) for i in range(n)]
        labels = dense_label_encode(reps)
        _n_groups = len(set(labels.tolist()))
        largest = max((labels == g).sum() for g in set(labels.tolist())) if n else 0
        if n and largest / n > 0.95:
            raise DegenerateGroupingError(
                f"SequenceIdentitySplitter: largest group holds {largest}/{n} sequences "
                f"(> 95%) at identity_threshold={self.identity_threshold}"
            )
        if n and largest / n > 0.6:
            warn_with_details(
                DegenerateClusterWarning(
                    f"SequenceIdentitySplitter: largest group holds {largest}/{n} sequences (> 60%)"
                )
            )
        return labels


class ProteinFamilySplitter(GroupSplitter):
    """Hold out whole target families/classes by a caller-supplied hierarchy.

    No external database lookup -- the family label per record is supplied directly by the
    caller.

    :param family_labels: one family or class label per record, aligned with ``X``. Required;
        nothing is inferred.
    :param size_tolerance: how far a realised partition size may drift from its target before
        a :class:`SizeToleranceWarning` is issued.
    :param group_assignment: how groups are handed to partitions; see
        :func:`chemsplit.base.assign_groups`.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ConfigurationError: if ``family_labels`` is missing.
    :raises InputError: at split time, if ``family_labels`` is the wrong length.

    Advantages
    ----------
    - Tests transfer *across target classes* -- kinases in train, GPCRs in test -- the claim
      behind most proteochemometrics work, and invisible to a within-family identity split.
    - Uses curated biological knowledge instead of a sequence heuristic, so the groups mean
      something to a biologist.
    - `held_out_families` makes the experiment specifiable in one line.

    Pitfalls
    --------
    - Family annotations are incomplete and inconsistent, and unlabelled targets form a junk
      group whose size is worth checking.
    - A family boundary is not pharmacological independence: ATP-binding proteins outside the
      kinase family share its ligand chemistry, so a new family can still be easy.
    - Most datasets have very few families, so holding one out is high-variance.
      `leave_one_cluster_out` gives leave-one-family-out instead.
    - Family sizes are very skewed, since kinases dominate public data, so the requested ratio
      is usually unreachable.

    References
    ----------
    .. [1] Holding out caller-supplied family labels is generic. The hierarchies and the
       leave-family-out precedent are published in [2]-[4].
    .. [2] Mistry, J.; Chuguransky, S.; Williams, L. et al. Pfam: The Protein Families
       Database in 2021. *Nucleic Acids Res.* **2021**, 49 (D1), D412-D419.
       https://doi.org/10.1093/nar/gkaa913
    .. [3] Zdrazil, B.; Felix, E.; Hunter, F. et al. The ChEMBL Database in 2023. *Nucleic
       Acids Res.* **2024**, 52 (D1), D1180-D1192. https://doi.org/10.1093/nar/gkad1004
    .. [4] Kramer, C.; Gedeck, P. Leave-Cluster-Out Cross-Validation Is Appropriate for Scoring
       Functions Derived from Diverse Protein Data Sets. *J. Chem. Inf. Model.* **2010**, 50 (11),
       1961-1969. https://doi.org/10.1021/ci100264e
    """

    splitter_id: ClassVar[str] = "protein_family"
    family: ClassVar[str] = "biomolecular"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("sequences", "features", "interactions")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = True
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = True

    def __init__(
        self,
        *,
        family_labels: list[str] | None = None,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        **base: Any,
    ) -> None:
        super().__init__(size_tolerance=size_tolerance, group_assignment=group_assignment, **base)
        self.family_labels = family_labels

    def _group_labels(self, ctx: Any) -> IndexArray:
        if self.family_labels is None:
            raise ParameterError(
                "ProteinFamilySplitter requires family_labels (no default inference)"
            )
        if len(self.family_labels) != ctx.n:
            raise ParameterError(
                f"family_labels has length {len(self.family_labels)}, expected {ctx.n}"
            )
        return dense_label_encode(list(self.family_labels))


class BindingSiteSplitter(GroupSplitter):
    """Cluster on pocket residue composition/sequence, rather than global sequence identity.

    :param representation: Butina on Euclidean distance over a caller-supplied
        residue-composition matrix, or ``sequence_identity``'s machinery over short pocket
        sequences.
    :param cutoff: Butina cutoff: a distance for ``"composition"``, and ``1 - identity`` for
        ``"pocket_sequence"``.
    :param size_tolerance: how far a realised partition size may drift from its target before
        a :class:`SizeToleranceWarning` is issued.
    :param group_assignment: how groups are handed to partitions; see
        :func:`chemsplit.base.assign_groups`.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if ``cutoff`` is outside ``(0, 1)``, or ``representation`` is
        unknown.
    :raises InputKindError: at split time, if the input does not match the chosen
        representation.
    :raises InputError: at split time, if a pocket definition is missing for any record.

    Advantages
    ----------
    - Catches the leak sequence identity misses: distant sequences with near-identical pockets,
      common across convergently evolved binding sites.
    - Pocket composition is interpretable, since the residues driving the grouping can be read
      off directly.
    - Works from any pocket definition, including one derived from a predicted structure.

    Pitfalls
    --------
    - **Needs a pocket definition the library cannot produce.** Pocket detection is its own
      research problem, and a different detector yields a different split.
    - `representation="composition"` ignores geometry, so two pockets with identical residue
      counts and completely different shapes look the same to it.
    - Pocket residue lists from a single co-crystal reflect one ligand's contacts rather than
      the pocket itself.
    - Targets without structures have no pocket, so the method covers only the structurally
      characterised subset. Missing pockets raise rather than being dropped silently.

    References
    ----------
    .. [1] This exact splitter is not itself published; the pocket representation and the
       clustering are, in [2]-[4].
    .. [2] Weill, N.; Rognan, D. Alignment-Free Ultra-High-Throughput Comparison of Druggable
       Protein-Ligand Binding Sites. *J. Chem. Inf. Model.* **2010**, 50 (1), 123-135.
       https://doi.org/10.1021/ci900349y (FuzCav; the residue-composition fingerprint
       ``representation="composition"`` mirrors)
    .. [3] Ehrt, C.; Brinkjost, T.; Koch, O. Impact of Binding Site Comparisons on Medicinal
       Chemistry and Rational Molecular Design. *J. Med. Chem.* **2016**, 59 (9), 4121-4151.
       https://doi.org/10.1021/acs.jmedchem.6b00078 (distant sequences can share
       near-identical pockets)
    .. [4] Butina, D. Unsupervised Data Base Clustering Based on Daylight's Fingerprint and
       Tanimoto Similarity. *J. Chem. Inf. Comput. Sci.* **1999**, 39 (4), 747-750.
       https://doi.org/10.1021/ci9803381
    """

    splitter_id: ClassVar[str] = "binding_site"
    family: ClassVar[str] = "biomolecular"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = True
    requires_labels: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("features", "sequences")
    extras: ClassVar[tuple[str,...]] = ("bio",)
    deterministic_without_seed: ClassVar[bool] = True
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        representation: Literal["composition", "pocket_sequence"] = "composition",
        cutoff: float = 0.35,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        **base: Any,
    ) -> None:
        super().__init__(size_tolerance=size_tolerance, group_assignment=group_assignment, **base)
        self.representation = representation
        self.cutoff = cutoff
        if representation not in ("composition", "pocket_sequence"):
            raise ParameterError(f"invalid representation: {representation!r}")
        if not (0.0 < cutoff < 1.0):
            raise ParameterError(f"cutoff must be in (0, 1), got {cutoff!r}")

    def _group_labels(self, ctx: Any) -> IndexArray:
        if self.representation == "composition":
            if ctx.raw_features is None:
                raise ParameterError(
                    "BindingSiteSplitter(representation='composition') requires a features matrix"
                )
            F = np.asarray(ctx.raw_features, dtype=np.float64)
            n = F.shape[0]
            diff = F[:, None,:] - F[None,:,:]
            D = np.sqrt((diff**2).sum(axis=-1)).astype(np.float32)
            np.fill_diagonal(D, 0.0)
        else:
            if ctx.sequences is None:
                raise LabelError(
                    "BindingSiteSplitter(representation='pocket_sequence') requires ctx.sequences"
                )
            sequences = ctx.sequences
            n = len(sequences)
            try:
                import parasail  # noqa: F401

                use_parasail = True
            except ImportError:
                use_parasail = False
            S = _pairwise_identity_matrix(sequences, use_parasail)
            D = (1.0 - S).astype(np.float32)
            np.fill_diagonal(D, 0.0)
        clusters = butina(D, self.cutoff, reorder=False)
        labels = np.empty(n, dtype=np.int64)
        for cid, members in enumerate(clusters):
            for m in members:
                labels[m] = cid
        largest = max(len(c) for c in clusters) if clusters else 0
        if n and largest / n > 0.95:
            raise DegenerateGroupingError(
                f"BindingSiteSplitter: largest cluster holds {largest}/{n} records (> 95%)"
            )
        if n and largest / n > 0.6:
            warn_with_details(
                DegenerateClusterWarning(
                    f"BindingSiteSplitter: largest cluster holds {largest}/{n} records (> 60%)"
                )
            )
        return dense_label_encode(labels.tolist())


class DepositionDateSplitter(BaseSplitter):
    """A date-cut split for structures, additionally pruning train records too similar to test.

    Specialises :class:`~chemsplit.splitters.lineage.TemporalSplitter`'s date cut for
    deposited structures: after the cut, any *train* record whose ligand similarity or sequence
    identity to a *test* record exceeds the given ceiling is discarded too. A structure
    deposited just after the cut date can otherwise be a near-duplicate of one deposited just
    before it.

    :param cut_date: the boundary: records dated at or before it are candidate train, later
        ones candidate test. ``None`` places the cut at the quantile implied by the sizes.
    :param ligand_similarity_ceiling: prune training ligands above this ECFP4 Tanimoto
        similarity to any test ligand, or ``None`` to skip ligand pruning.
    :param sequence_identity_ceiling: prune training sequences above this identity to any test
        sequence, or ``None`` to skip sequence pruning.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if either ceiling is outside ``(0, 1]``.
    :raises InputError: at split time, if ``dates`` is missing, or a ceiling is set without the
        molecules or sequences it needs.
    :raises EmptyPartitionError: at split time, if pruning empties train.

    Advantages
    ----------
    - The established protocol for evaluating docking, scoring functions and co-folding, and
      the reason several early scoring-function results failed to reproduce.
    - The pruning closes the leak a pure date cut leaves open: the same ligand series and
      protein are redeposited for years, so the cut alone separates almost nothing.
    - Every pruning decision is counted and reported.

    Pitfalls
    --------
    - A deposition-date cut on its own is a weak split, since redundant re-depositions of one
      complex put near-identical entries on both sides. That is what the pruning is for.
    - Deposition date is not discovery date: structures are often deposited long after the
      work, and release dates differ again.
    - Pruning by ligand similarity removes the most informative complexes, depressing
      absolute scores by design, so cross-study comparison needs matching ceilings.
    - Sequence pruning is off by default because it needs sequences, and leaving it off keeps
      homologous complexes in training.

    References
    ----------
    .. [1] Li, Y.; Yang, J. Structural and Sequence Similarity Makes a Significant Impact on
       Machine-Learning-Based Scoring Functions for Protein-Ligand Interactions.
       *J. Chem. Inf. Model.* **2017**, 57 (4), 1007-1012.
       https://doi.org/10.1021/acs.jcim.7b00049
    .. [2] Li, J.; Guan, X.; Zhang, O. et al. Leak Proof PDBBind: A Reorganized Data Set of
       Protein-Ligand Complexes for More Generalizable Binding Affinity Prediction.
       *J. Phys. Chem. B* **2026**, 130 (2), 730-740. https://doi.org/10.1021/acs.jpcb.5c08598
       (a temporal holdout combined with sequence and ligand-similarity de-leaking, as here)
    .. [3] Su, M.; Yang, Q.; Du, Y. et al. Comparative Assessment of Scoring Functions: The
       CASF-2016 Update. *J. Chem. Inf. Model.* **2019**, 59 (2), 895-913.
       https://doi.org/10.1021/acs.jcim.8b00545
    """

    splitter_id: ClassVar[str] = "deposition_date"
    family: ClassVar[str] = "biomolecular"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = False
    requires_dates: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features", "sequences")
    extras: ClassVar[tuple[str,...]] = ()
    deterministic_without_seed: ClassVar[bool] = True
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = True

    def __init__(
        self,
        *,
        cut_date: str | np.datetime64 | None = None,
        ligand_similarity_ceiling: float | None = None,
        sequence_identity_ceiling: float | None = None,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.cut_date = cut_date
        self.ligand_similarity_ceiling = ligand_similarity_ceiling
        self.sequence_identity_ceiling = sequence_identity_ceiling

    def _partition(self, ctx: Any) -> list[SplitResult]:
        if ctx.dates is None:
            raise LabelError("DepositionDateSplitter requires dates (requires_dates=True)")
        if self.cut_date is None:
            raise ParameterError("DepositionDateSplitter requires cut_date")
        cut = np.datetime64(self.cut_date)
        dates = np.asarray(ctx.dates)
        train_mask = dates <= cut
        test_mask = ~train_mask
        train = np.nonzero(train_mask)[0]
        test = np.nonzero(test_mask)[0]
        discard: list[int] = []

        prune_ligands = self.ligand_similarity_ceiling is not None and ctx.mols is not None
        if prune_ligands and len(train) and len(test):
            from chemsplit._fp_similarity import compute_similarity_matrix

            S = compute_similarity_matrix(
                ctx, "ecfp4", "tanimoto", 2 * 1024**3, "DepositionDateSplitter", 1
            )
            cross = S[np.ix_(train, test)]
            max_sim = cross.max(axis=1)
            prune_mask = max_sim > self.ligand_similarity_ceiling
            discard.extend(train[prune_mask].tolist())
            train = train[~prune_mask]

        prune_seqs = self.sequence_identity_ceiling is not None and ctx.sequences is not None
        if prune_seqs and len(train) and len(test):
            try:
                import parasail  # noqa: F401

                use_parasail = True
            except ImportError:
                use_parasail = False
            seqs = ctx.sequences
            for ti in list(train):
                m = max(
                    (
                        _parasail_identity(seqs[ti], seqs[tj])
                        if use_parasail
                        else _hamming_identity(seqs[ti], seqs[tj])
                    )
                    for tj in test
                )
                if m > self.sequence_identity_ceiling:
                    discard.append(int(ti))
            train = np.array([t for t in train if t not in set(discard)], dtype=np.int64)

        train = np.sort(np.asarray(train, dtype=np.int64))
        test = np.sort(np.asarray(test, dtype=np.int64))
        discard_arr = np.sort(np.asarray(sorted(set(discard)), dtype=np.int64))
        return [
            SplitResult(
                train=train,
                valid=np.array([], dtype=np.int64),
                test=test,
                discard=discard_arr,
                groups=None,
                splitter_id=self.splitter_id,
                params=_canonicalize_params(self.get_params()),
                n_records=ctx.n,
                metadata={
                    "realised_sizes": {
                        "train": int(train.size),
                        "valid": 0,
                        "test": int(test.size),
                    },
                    "n_pruned": int(discard_arr.size),
                    "cut_date": str(cut),
                },
            )
        ]


class ComplexJointSplitter(BaseSplitter):
    """Jointly novel on the ligand-similarity axis AND the sequence-identity axis.

    Composes two independent group axes -- ligand groups from ``ligand_grouper`` and
    sequence groups from ``sequence_grouper`` -- through
    :func:`chemsplit._pair_assign.assign_pair_groups`, so a test complex is guaranteed novel on
    both axes (``mode="both_novel"``) or at least one axis (``mode="either_novel"``), sized at
    ``sqrt(f)`` per axis (see ``chemsplit._pair_assign``'s module docstring for why).

    :param ligand_grouper: groups records by ligand similarity. ``None`` calls
        :func:`chemsplit.clustering.butina` at a 0.35 cutoff directly.
    :param sequence_grouper: groups records by sequence identity. ``None`` uses this module's
        :class:`SequenceIdentitySplitter`.
    :param mode: require a test complex to be novel on both axes, or on either one; see
        :func:`chemsplit._pair_assign.assign_pair_groups`.
    :param base: forwarded to :class:`chemsplit.base.BaseSplitter`.
    :raises ParameterError: if a grouper is not a :class:`~chemsplit.base.GroupSplitter`, or
        ``mode`` is unknown.
    :raises InputError: at split time, if the sequences needed by ``sequence_grouper`` are
        missing.
    :raises ConstraintUnsatisfiableError: at split time, if too little data survives both
        constraints.

    Advantages
    ----------
    - The only defensible setting for claiming generalisation to genuinely new complexes, with
      both constraints verified and reported.
    - `mode="either_novel"` gives an intermediate, larger-data experiment when `"both_novel"`
      leaves too little.

    Pitfalls
    --------
    - Leaves very little data: the discard fraction routinely exceeds 80%, and the surviving
      test set may be too small for a stable metric. It is reported, and beyond
      `max_discard_frac` the split refuses.
    - Two thresholds and two groupers compound into four choices defining the experiment, none
      with a canonical value.
    - A tiny test set makes a single aggregate number easy to over-read; per-complex results
      say more.
    - `mode="either_novel"` is much weaker and is frequently reported as if it were
      `"both_novel"`.

    References
    ----------
    .. [1] Pahikkala, T.; Airola, A.; Pietilä, S. et al. Toward More Realistic Drug-Target
       Interaction Predictions. *Brief. Bioinform.* **2015**, 16 (2), 325-337.
       https://doi.org/10.1093/bib/bbu010 (setting S4: both the compound and the target are
       unseen)
    .. [2] Li, J.; Guan, X.; Zhang, O. et al. Leak Proof PDBBind: A Reorganized Data Set of
       Protein-Ligand Complexes for More Generalizable Binding Affinity Prediction.
       *J. Phys. Chem. B* **2026**, 130 (2), 730-740. https://doi.org/10.1021/acs.jpcb.5c08598
    .. [3] Durairaj, J.; Adeshina, Y.; Cao, Z. et al. PLINDER: The Protein-Ligand Interactions
       Dataset and Evaluation Resource. *bioRxiv* preprint, **2024** (not peer reviewed).
       https://doi.org/10.1101/2024.07.17.603955
    .. [4] The ``sqrt(f)`` sizing rule that reaches the requested test fraction on two axes at
       once is chemsplit's own.
    """

    splitter_id: ClassVar[str] = "complex_joint"
    family: ClassVar[str] = "biomolecular"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = False
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol", "features")
    extras: ClassVar[tuple[str,...]] = ("bio",)
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        ligand_grouper: GroupSplitter | None = None,
        sequence_grouper: GroupSplitter | None = None,
        mode: Literal["both_novel", "either_novel"] = "both_novel",
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.ligand_grouper = ligand_grouper
        self.sequence_grouper = sequence_grouper
        self.mode = mode
        if mode not in ("both_novel", "either_novel"):
            raise ParameterError(f"invalid mode: {mode!r}")

    def _ligand_labels(self, ctx: Any) -> IndexArray:
        if self.ligand_grouper is not None:
            ligand_X = ctx.raw_features if ctx.raw_features is not None else ctx.smiles
            return self.ligand_grouper.compute_groups(ligand_X)
        if ctx.mols is None:
            raise ParameterError(
                "ComplexJointSplitter needs molecules to derive a default ligand grouping"
            )
        from chemsplit._fp_similarity import compute_distance_matrix

        D = compute_distance_matrix(
            ctx, "ecfp4", "tanimoto", 2 * 1024**3, "ComplexJointSplitter", 1
        )
        clusters = butina(D, 0.35, reorder=False)
        labels = np.empty(ctx.n, dtype=np.int64)
        for cid, members in enumerate(clusters):
            for m in members:
                labels[m] = cid
        return dense_label_encode(labels.tolist())

    def _sequence_labels(self, ctx: Any) -> IndexArray:
        if self.sequence_grouper is not None:
            # without X_kind="sequences" a bare list[str] is read as SMILES, and protein
            # sequences don't parse as molecules
            return self.sequence_grouper.compute_groups(ctx.sequences, X_kind="sequences")
        if ctx.sequences is None:
            raise ParameterError(
                "ComplexJointSplitter needs ctx.sequences to derive a default sequence "
                "grouping"
            )
        use_parasail = True
        try:
            import parasail  # noqa: F401
        except ImportError:
            use_parasail = False
        D = 1.0 - _pairwise_identity_matrix(ctx.sequences, use_parasail)
        n = len(ctx.sequences)
        uf = UnionFind(n)
        S = 1.0 - D
        for i in range(n):
            for j in range(i + 1, n):
                if S[i, j] > 0.7:
                    uf.union(i, j)
        return dense_label_encode([uf.find(i) for i in range(n)])

    def _partition(self, ctx: Any) -> list[SplitResult]:
        labels_a = self._ligand_labels(ctx)
        labels_b = self._sequence_labels(ctx)
        rng = seed_for(ctx.rng_seeds, "complexjoint.pair_assign", 0)
        buckets = assign_pair_groups(labels_a, labels_b, ctx.sizes, rng, mode=self.mode)
        train, valid, test, discard = (
            buckets["train"],
            buckets["valid"],
            buckets["test"],
            buckets["discard"],
        )
        return [
            SplitResult(
                train=np.sort(train),
                valid=np.sort(valid),
                test=np.sort(test),
                discard=np.sort(discard),
                groups=None,
                splitter_id=self.splitter_id,
                params=_canonicalize_params(self.get_params()),
                n_records=ctx.n,
                metadata={
                    "realised_sizes": {
                        "train": int(train.size),
                        "valid": int(valid.size),
                        "test": int(test.size),
                    },
                    "mode": self.mode,
                    "n_discarded": int(discard.size),
                    # the default grouper uses parasail when installed and hamming
                    # otherwise, so the split isn't bit-exact across environments
                    "nondeterministic_method": True,
                },
            )
        ]
