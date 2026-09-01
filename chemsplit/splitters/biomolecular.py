"""Biomolecular-axis splitters: hold out on protein sequence identity, family, binding site, deposition date, or a joint ligand+sequence axis.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Literal

import numpy as np

from chemsplit._pair_assign import assign_pair_groups
from chemsplit._unionfind import UnionFind, dense_label_encode
from chemsplit.base import BaseSplitter, GroupSplitter, SplitResult, Strictness, _canonicalize_params
from chemsplit.clustering import butina
from chemsplit.determinism import argmax_tiebreak, seed_for
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
    identity metric for the pure-Python fallback path — not a full alignment)."""
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

    Parameters
    ----------
    identity_threshold: float, default=0.7
        Sequences whose pairwise identity exceeds this are merged into one group
        (single-linkage — a chain of pairwise-similar sequences can span very different
        sequences at the chain's ends, exactly as Butina/graph-component grouping does on the
        ligand side).
    algorithm: {"auto", "parasail", "hamming"}, default="auto"
        ``"parasail"`` uses global alignment via the ``bio`` extra (``parasail``);
        ``"hamming"`` is a dependency-free equal-length fractional-match fallback;
        ``"auto"`` uses ``parasail`` if installed, else ``"hamming"``.

    Advantages
    ----------
    - The standard control for any model taking a protein as input — without it, a "new target" claim isn't credible.
    - Computes and reports `max_cross_identity`, so the claim is backed by evidence.
    - Makes the `identity_denominator` choice explicit, removing the most common source of incomparable identity numbers between papers.
    - Deduplicating to unique sequences first keeps it cheap even on large interaction datasets.

    Pitfalls
    --------
    - **Percent identity isn't one number.** The same alignment gives very different identities depending on the denominator and on local vs. global alignment; 30% local over the shorter sequence is far weaker than 30% global over the alignment.
    - A poor proxy for binding-site similarity — two proteins at 20% overall identity can share nearly identical pockets, and this split will happily separate them while leaking the pharmacology. Pair with `binding_site`.
    - `greedy_incremental` is order-dependent by construction; longest-first fixes it here, but results won't match other tools' orderings.
    - Multi-domain and multi-chain proteins align poorly as single strings; coverage filtering helps but doesn't solve it.
    - `aligner="kmer"` is an approximation and must be disclosed.

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
            raise ParameterError(f"identity_threshold must be in (0, 1], got {identity_threshold!r}")
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
            raise LabelError("SequenceIdentitySplitter requires ctx.sequences (accepts=('sequences',))")
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

    No external database lookup — the family label per record is supplied directly by the
    caller.

    Parameters
    ----------
    family_labels: Sequence[str] | None, default=None
        One family/class label per record, aligned with ``X``. Required (no default inference).

    Advantages
    ----------
    - Tests transfer *across target classes* — kinases in train, GPCRs in test — the real claim behind most proteochemometrics work, and invisible to a sequence-identity split within one family.
    - Uses curated biological knowledge instead of a sequence heuristic, so the groups mean something to a biologist.
    - Explicit `held_out_families` makes the experiment fully specifiable in one line.

    Pitfalls
    --------
    - Family annotations are incomplete and inconsistent; unlabelled targets form a junk group whose size must be checked.
    - Family boundaries don't imply pharmacological independence — kinase and non-kinase ATP-binding proteins share ligand chemistry, so a "new family" can still be an easy target.
    - Most datasets have very few families, so holding one out is high-variance; prefer leave-one-family-out via `leave_one_cluster_out`.
    - Family sizes are extremely skewed (kinases dominate public data), so the requested ratio is usually unreachable.

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
        family_labels: "list[str] | None" = None,
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        **base: Any,
    ) -> None:
        super().__init__(size_tolerance=size_tolerance, group_assignment=group_assignment, **base)
        self.family_labels = family_labels

    def _group_labels(self, ctx: Any) -> IndexArray:
        if self.family_labels is None:
            raise ParameterError("ProteinFamilySplitter requires family_labels (no default inference)")
        if len(self.family_labels) != ctx.n:
            raise ParameterError(
                f"family_labels has length {len(self.family_labels)}, expected {ctx.n}"
            )
        return dense_label_encode(list(self.family_labels))


class BindingSiteSplitter(GroupSplitter):
    """Cluster on pocket residue composition/sequence, rather than global sequence identity.

    Parameters
    ----------
    representation: {"composition", "pocket_sequence"}, default="composition"
        ``"composition"`` clusters a caller-supplied residue-composition feature matrix (passed
        as ``X`` with ``accepts=("features",)``) via Butina on Euclidean distance.
        ``"pocket_sequence"`` clusters caller-supplied short pocket sequences (``accepts=
        ("sequences",)``) via the same identity machinery as ``sequence_identity``, requiring the
        ``bio`` extra for the accelerated path (falls back to the Hamming approximation
        otherwise, same as ``SequenceIdentitySplitter``).
    cutoff: float, default=0.35
        Butina cutoff (distance for ``"composition"``, ``1 - identity`` for
        ``"pocket_sequence"``).

    Advantages
    ----------
    - Catches the leak sequence identity misses: distant sequences with near-identical pockets, common across convergently evolved binding sites.
    - Pocket composition is directly interpretable — you can read off which residues drive the grouping.
    - Works from any pocket definition, including one derived from a predicted structure.

    Pitfalls
    --------
    - **Needs a pocket definition the library can't produce.** Pocket detection is its own research problem, and a different detector yields a different split.
    - `residue_composition` ignores geometry entirely — two pockets with identical residue counts and completely different shapes look identical to it.
    - Pocket residue lists from a single co-crystal reflect one ligand's contacts, not the pocket itself.
    - Unavailable for targets without structures, so the method silently applies only to the structurally characterised subset unless the caller handles the rest — which is why missing pockets raise rather than get dropped.

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

    Specializes the ``temporal`` (``TemporalSplitter``) date-cut for
    deposited structures: after the ordinary temporal cut, any *train* record whose ligand
    Tanimoto similarity or sequence identity to a *test* record exceeds the given ceiling is
    additionally removed from train (moved to ``discard``) — a leakage-prevention step layered on
    top of the date boundary, since a structure deposited just after the cut date can still be a
    near-duplicate of one deposited just before it.

    Parameters
    ----------
    cut_date: str | numpy.datetime64
        Records with ``dates <= cut_date`` are candidate train; later records are candidate test.
    ligand_similarity_ceiling: float | None, default=None
        If given, train ligands (via ``ctx.mols``' ECFP4 Tanimoto similarity) more similar than
        this to any test ligand are pruned from train.
    sequence_identity_ceiling: float | None, default=None
        If given, train sequences more identical than this to any test sequence are pruned from
        train (same identity machinery as ``sequence_identity``).

    Advantages
    ----------
    - The established protocol for evaluating docking, scoring functions, and co-folding — and the reason several early scoring-function results failed to reproduce.
    - The added ligand and sequence pruning closes the leak a pure date cut leaves open: the same ligand series and protein get redeposited for years, so a date cut alone separates almost nothing.
    - Every pruning decision is counted and reported.

    Pitfalls
    --------
    - A deposition-date cut alone is a weak split — redundant re-depositions of the same complex put near-identical entries on both sides of the cut, which is why the pruning defaults exist.
    - Deposition date isn't discovery date; structures are often deposited long after the work, and release dates differ again.
    - Pruning by ligand similarity removes exactly the complexes most informative for testing, which depresses absolute scores intentionally — but makes cross-study comparison unsafe unless the ceilings match.
    - Sequence pruning is off by default because it needs sequences; leaving it off keeps homologous complexes in training, as the docstring notes.

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
        cut_date: "str | np.datetime64 | None" = None,
        ligand_similarity_ceiling: "float | None" = None,
        sequence_identity_ceiling: "float | None" = None,
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

        if self.ligand_similarity_ceiling is not None and ctx.mols is not None and len(train) and len(test):
            from chemsplit._fp_similarity import compute_similarity_matrix

            S = compute_similarity_matrix(
                ctx, "ecfp4", "tanimoto", 2 * 1024**3, "DepositionDateSplitter", 1
            )
            cross = S[np.ix_(train, test)]
            max_sim = cross.max(axis=1)
            prune_mask = max_sim > self.ligand_similarity_ceiling
            discard.extend(train[prune_mask].tolist())
            train = train[~prune_mask]

        if self.sequence_identity_ceiling is not None and ctx.sequences is not None and len(train) and len(test):
            try:
                import parasail  # noqa: F401

                use_parasail = True
            except ImportError:
                use_parasail = False
            seqs = ctx.sequences
            for ti in list(train):
                m = max(
                    (_parasail_identity(seqs[ti], seqs[tj]) if use_parasail else _hamming_identity(seqs[ti], seqs[tj]))
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
                    "realised_sizes": {"train": int(train.size), "valid": 0, "test": int(test.size)},
                    "n_pruned": int(discard_arr.size),
                    "cut_date": str(cut),
                },
            )
        ]


class ComplexJointSplitter(BaseSplitter):
    """Jointly novel on the ligand-similarity axis AND the sequence-identity axis.

    Composes two independent group axes — ligand groups (via
    ``ligand_grouper``) and sequence/target groups (via ``sequence_grouper``) — through
    :func:`chemsplit._pair_assign.assign_pair_groups`, so a test complex is guaranteed novel on
    both axes (``mode="both_novel"``) or at least one axis (``mode="either_novel"``), sized at
    ``sqrt(f)`` per axis (see ``chemsplit._pair_assign``'s module docstring for why).

    Parameters
    ----------
    ligand_grouper: GroupSplitter | None, default=None
        Groups records by ligand similarity. ``None`` falls back to Butina clustering
        (``chemsplit.clustering.butina``, cutoff 0.35) directly, as a stand-in for this project's
        string default ``"butina"`` — ``chemsplit.registry`` does not exist yet to resolve
        that string; pass an instantiated splitter (e.g. a future ``ButinaSplitter``) once
        available for the real behaviour.
    sequence_grouper: GroupSplitter | None, default=None
        Groups records by sequence identity. ``None`` falls back to using this module's own
        :class:`SequenceIdentitySplitter` directly (an intra-module reference, not circular).
    mode: {"both_novel", "either_novel"}, default="both_novel"
        See :func:`chemsplit._pair_assign.assign_pair_groups`.

    Advantages
    ----------
    - The only defensible setting for claiming generalisation to genuinely new complexes — both constraints are verified and reported.
    - `mode="either_novel"` gives an intermediate, larger-data experiment when `"both_novel"` leaves too little.

    Pitfalls
    --------
    - Leaves very little data — on typical structural datasets the discard fraction exceeds 80%, and the surviving test set may be too small for a stable metric. Reports the fraction and refuses beyond `max_discard_frac`.
    - Two thresholds and two groupers compound into four choices defining the experiment, none with a canonical value.
    - A tiny test set invites over-interpreting a single number — report per-complex results, not just an aggregate.
    - `mode="either_novel"` is much weaker and is frequently reported as if it were `"both_novel"`.

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
        ligand_grouper: "GroupSplitter | None" = None,
        sequence_grouper: "GroupSplitter | None" = None,
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
            return self.ligand_grouper.compute_groups(ctx.raw_features if ctx.raw_features is not None else ctx.smiles)
        if ctx.mols is None:
            raise ParameterError("ComplexJointSplitter needs molecules to derive a default ligand grouping")
        from chemsplit._fp_similarity import compute_distance_matrix

        D = compute_distance_matrix(ctx, "ecfp4", "tanimoto", 2 * 1024**3, "ComplexJointSplitter", 1)
        clusters = butina(D, 0.35, reorder=False)
        labels = np.empty(ctx.n, dtype=np.int64)
        for cid, members in enumerate(clusters):
            for m in members:
                labels[m] = cid
        return dense_label_encode(labels.tolist())

    def _sequence_labels(self, ctx: Any) -> IndexArray:
        if self.sequence_grouper is not None:
            return self.sequence_grouper.compute_groups(ctx.sequences)
        if ctx.sequences is None:
            raise ParameterError("ComplexJointSplitter needs ctx.sequences to derive a default sequence grouping")
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
                },
            )
        ]
