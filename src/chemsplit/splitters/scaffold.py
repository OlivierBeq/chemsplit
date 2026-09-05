"""The ``scaffold``/substructure splitter family.

All records are molecules; grouping is computed from a per-record structural key (scaffold,
generic framework, ring system, matched-molecular-series context) except
:class:`ActivityCliffSplitter`, which is not group-forming.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMMPA

from chemsplit import scaffolds as _scaffolds
from chemsplit._unionfind import UnionFind, dense_label_encode
from chemsplit.base import BaseSplitter, GroupSplitter, SplitResult, Strictness, _Context
from chemsplit.determinism import argmax_tiebreak, seed_for
from chemsplit.exceptions import (
    ConstraintUnsatisfiableError,
    DegenerateClusterWarning,
    DegenerateGroupingError,
    ParameterError,
    ScalabilityError,
    warn_with_details,
)
from chemsplit.types import IndexArray

__all__ = [
    "MurckoScaffoldSplitter",
    "GenericScaffoldSplitter",
    "ScaffoldTreeSplitter",
    "RingSystemSplitter",
    "MatchedMolecularSeriesSplitter",
    "ActivityCliffSplitter",
]

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_degenerate(n_groups: int, n_records: int, class_name: str) -> None:
    """Raise :class:`DegenerateGroupingError` when a grouping produced exactly 1 group, or every
    (kept) record is its own group — the generic degenerate condition every scaffold-family
    splitter documents."""
    if n_records == 0:
        return
    if n_groups == 1 or n_groups == n_records:
        raise DegenerateGroupingError(
            f"{class_name}: grouping produced {n_groups} group(s) over {n_records} record(s) "
            "(degenerate: exactly 1 group, or every record its own group). Group-count histogram "
            f"(first 5): {_histogram_head(n_groups, n_records)}"
        )


def _histogram_head(n_groups: int, n_records: int) -> str:
    if n_groups == 1:
        return f"[{n_records}]"
    return f"[{'1, ' * min(5, n_records)}...]" if n_records else "[]"


def _apply_on_empty_scaffold(
    keys: list[str],
    on_empty_scaffold: str,
    class_name: str,
) -> tuple[list[str], list[int]]:
    """Rewrite empty-scaffold (``""``) keys per ``on_empty_scaffold`` policy.

    Returns ``(rewritten_keys, forced_discard_indices)``. ``"own_group"`` gives each empty-key
    record a unique key (own singleton group); ``"shared_group"`` leaves them sharing ``""``;
    ``"discard"`` returns their indices for the caller to fold into ``ctx.extra["forced_discard"]``;
    ``"raise"`` raises immediately.
    """
    empty_idx = [i for i, k in enumerate(keys) if k == ""]
    if not empty_idx:
        return keys, []
    if on_empty_scaffold == "own_group":
        keys = list(keys)
        for i in empty_idx:
            keys[i] = f"\x00EMPTY_SINGLETON_{i}\x00"
        return keys, []
    if on_empty_scaffold == "shared_group":
        return keys, []
    if on_empty_scaffold == "discard":
        return keys, empty_idx
    if on_empty_scaffold == "raise":
        raise DegenerateGroupingError(
            f"{class_name}: {len(empty_idx)} record(s) have an empty scaffold key and "
            "on_empty_scaffold='raise'"
        )
    raise ParameterError(f"invalid on_empty_scaffold: {on_empty_scaffold!r}")


def _merge_forced_discard(ctx: _Context, new_indices: list[int]) -> None:
    if not new_indices:
        return
    existing = set(ctx.extra.get("forced_discard", []))
    ctx.extra["forced_discard"] = sorted(existing | set(new_indices))


def _group_size_metadata(labels: IndexArray, keep_mask: np.ndarray) -> dict[str, Any]:
    kept = labels[keep_mask]
    if kept.size == 0:
        return {"n_groups": 0, "group_sizes": [], "largest_group_frac": 0.0, "singleton_frac": 0.0}
    counts = np.bincount(kept)
    counts = counts[counts > 0]
    sizes_sorted = sorted(counts.tolist(), reverse=True)
    return {
        "n_groups": int(len(sizes_sorted)),
        "group_sizes": sizes_sorted,
        "largest_group_frac": float(sizes_sorted[0]) / float(kept.size),
        "singleton_frac": float(sum(1 for s in sizes_sorted if s == 1)) / float(len(sizes_sorted)),
    }


def _require_mols(ctx: _Context, class_name: str) -> list:
    if ctx.mols is None:
        raise ParameterError(f"{class_name} requires molecule input (smiles/mol), got features")
    return ctx.mols


# The full BaseSplitter + GroupSplitter constructor surface, spelled out explicitly on every
# leaf class below (never `**kwargs`) — sklearn's BaseEstimator.get_params() silently drops any
# parameter that only appears behind a **kwargs catch-all in the LEAF class's own __init__
# signature (verified directly: sklearn.base.BaseEstimator._get_param_names() filters out
# VAR_KEYWORD parameters rather than raising), which would otherwise make SplitResult.params
# silently incomplete. Every concrete splitter in every family must follow this pattern.


class _ScaffoldFamilyBase(GroupSplitter):
    """Shared validation for the family's ``include_chirality``/``on_empty_scaffold`` block.

    A private implementation convenience so the six leaf classes below don't
    each repeat the same two ``if`` checks. Leaf classes still declare their own full, explicit
    ``__init__`` signature (see the module-level note above on sklearn's ``get_params()``).
    """

    family: ClassVar[str] = "scaffold"
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol")
    deterministic_without_seed: ClassVar[bool] = False  # group_assignment="random" depends on seed
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def _validate_shared(self) -> None:
        if not isinstance(self.include_chirality, bool):
            raise ParameterError(f"include_chirality must be bool, got {self.include_chirality!r}")
        if self.on_empty_scaffold not in ("own_group", "shared_group", "discard", "raise"):
            raise ParameterError(f"invalid on_empty_scaffold: {self.on_empty_scaffold!r}")


# ---------------------------------------------------------------------------
# MurckoScaffoldSplitter
# ---------------------------------------------------------------------------


class MurckoScaffoldSplitter(_ScaffoldFamilyBase):
    """Group records by Murcko scaffold.

    Deterministic without a seed (unless ``group_assignment="random"``), O(n), and requires no
    distance matrix — the default comparability baseline in the scaffold-split literature.

    Advantages
    ----------
    - Seed-free, `O(n)`, needs no distance matrix, and scales to millions of molecules.
    - The group key is a chemist-readable SMILES, so a disputed assignment can be inspected by eye — no clustering split offers that.
    - Harder than random and far cheaper than any similarity split, making it the default comparability baseline in the literature.
    - Directly targets the "new chemotype" question when a dataset genuinely spans many distinct frameworks.

    Pitfalls
    --------
    - **Systematically weaker than it looks.** Two molecules differing by one atom can land in different Murcko scaffolds (a benzene-to-pyridine swap in a fused system), so cross-boundary Tanimoto similarity routinely stays above 0.6. Always confirm with `audit.nn_similarity_profile`.
    - On large diverse sets, singletons dominate the scaffold distribution — often over 50% of records — so the split degenerates toward random for most of the data. Read `metadata["singleton_frac"]`.
    - On a focused project set, one scaffold can hold most of the data, making the requested ratio unreachable; the splitter warns rather than fails, so read the warning.
    - The default `greedy_desc` ordering puts the largest scaffold groups in train, systematically enriching the test set in rare chemotypes. `group_assignment="random"` removes this bias at the cost of more variance.
    - Side chains are discarded entirely, so two molecules with the same core but very different substituents land in the same group — this can put genuinely dissimilar compounds on the same side of the boundary.
    - `on_empty_scaffold="shared_group"` silently lumps all acyclic molecules into one huge group — a common source of surprise, and the historical default in some libraries.

    """

    splitter_id: ClassVar[str] = "murcko_scaffold"
    strictness: ClassVar[Strictness] = Strictness.MODERATE

    def __init__(
        self,
        *,
        include_chirality: bool = False,
        on_empty_scaffold: Literal["own_group", "shared_group", "discard", "raise"] = "own_group",
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.include_chirality = include_chirality
        self.on_empty_scaffold = on_empty_scaffold
        self._validate_shared()

    def _group_labels(self, ctx: _Context) -> IndexArray:
        mols = _require_mols(ctx, type(self).__name__)
        keys = [
            _scaffolds.murcko_scaffold(m, self.include_chirality) if m is not None else ""
            for m in mols
        ]
        keys, forced = _apply_on_empty_scaffold(keys, self.on_empty_scaffold, type(self).__name__)
        _merge_forced_discard(ctx, forced)
        labels_full = dense_label_encode(keys)
        keep_mask = np.ones(ctx.n, dtype=bool)
        if forced:
            keep_mask[forced] = False
        _check_degenerate(
            len(set(labels_full[keep_mask].tolist())), int(keep_mask.sum()), type(self).__name__
        )
        self._last_keys = keys
        return labels_full

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        forced = ctx.extra.get("forced_discard", [])
        keep_mask = np.ones(ctx.n, dtype=bool)
        if forced:
            keep_mask[forced] = False
        meta = _group_size_metadata(labels, keep_mask)
        keys = getattr(self, "_last_keys", None)
        if keys is not None:
            # Report the scaffold SMILES of each group's first-seen member.
            seen: dict[int, str] = {}
            for i, lab in enumerate(labels.tolist()):
                if lab not in seen:
                    seen[lab] = keys[i]
            meta["scaffold_smiles"] = [seen[g] for g in sorted(seen)]
        return meta


# ---------------------------------------------------------------------------
# GenericScaffoldSplitter
# ---------------------------------------------------------------------------


class GenericScaffoldSplitter(_ScaffoldFamilyBase):
    """Group records by generic (heteroatoms->carbon) scaffold framework.

    Advantages
    ----------
    - Collapses heteroatom variants onto one topology, so benzene/pyridine/pyrimidine analogues can't straddle the boundary — closing the biggest leak in `murcko_scaffold`.
    - Produces fewer, larger groups, making the test set genuinely novel in ring topology.
    - Still seed-free and `O(n)`.

    Pitfalls
    --------
    - Groups get coarse enough that on small datasets the achievable train/test ratio drifts badly — expect `SizeToleranceWarning`.
    - Higher metric variance, since a single large group landing in test can dominate the score.
    - Topological identity isn't chemical identity — a generic scaffold can merge compounds with unrelated electronics and binding modes, making it too strict in a way with no medicinal-chemistry meaning.
    - Sanitisation fallbacks silently mix two key types — check `metadata["generic_fallbacks"]` before reporting.

    """

    splitter_id: ClassVar[str] = "generic_scaffold"
    strictness: ClassVar[Strictness] = Strictness.STRICT

    def __init__(
        self,
        *,
        include_chirality: bool = False,
        on_empty_scaffold: Literal["own_group", "shared_group", "discard", "raise"] = "own_group",
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        if include_chirality:
            raise ParameterError(
                "GenericScaffoldSplitter: include_chirality must be False — a generic scaffold "
                "has no stereochemistry"
            )
        super().__init__(
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.include_chirality = include_chirality
        self.on_empty_scaffold = on_empty_scaffold
        self._validate_shared()

    def _group_labels(self, ctx: _Context) -> IndexArray:
        mols = _require_mols(ctx, type(self).__name__)
        keys = [_scaffolds.generic_scaffold(m) if m is not None else "" for m in mols]
        keys, forced = _apply_on_empty_scaffold(keys, self.on_empty_scaffold, type(self).__name__)
        _merge_forced_discard(ctx, forced)
        labels_full = dense_label_encode(keys)
        keep_mask = np.ones(ctx.n, dtype=bool)
        if forced:
            keep_mask[forced] = False
        _check_degenerate(
            len(set(labels_full[keep_mask].tolist())), int(keep_mask.sum()), type(self).__name__
        )
        return labels_full

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        forced = ctx.extra.get("forced_discard", [])
        keep_mask = np.ones(ctx.n, dtype=bool)
        if forced:
            keep_mask[forced] = False
        return _group_size_metadata(labels, keep_mask)


# ---------------------------------------------------------------------------
# ScaffoldTreeSplitter
# ---------------------------------------------------------------------------


class ScaffoldTreeSplitter(_ScaffoldFamilyBase):
    """Group records by scaffold-tree node at a chosen pruning ``level``.

    ``level=0`` is exactly :class:`MurckoScaffoldSplitter`. Higher levels remove peripheral rings
    one at a time (see :func:`chemsplit.scaffolds.scaffold_tree_levels` — its ``"scaffold_tree"``
    prune rule is a documented best-effort reconstruction, not verified against the original
    Scaffold Tree publication's exact rule text).

    Advantages
    ----------
    - Keeps a whole scaffold lineage together, so a test compound can't be a ring-truncated relative of a training compound — a leak plain Murcko splitting misses entirely.
    - The `level` knob gives a tunable strictness dial with actual chemical meaning, unlike an abstract distance cutoff.
    - Deterministic and seed-free.

    Pitfalls
    --------
    - `level` is a free parameter that silently controls difficulty — a paper reporting "scaffold tree split" without the level and prune rule isn't reproducible, so the value is always written into `metadata` and `SplitResult.params`.
    - Ring pruning isn't canonical across toolkits — the underlying chemistry-based priorities differ, making cross-library comparison unsafe.
    - High `level` collapses everything into a handful of one-ring keys, giving huge groups and unreachable size targets.
    - Bridged, spiro, and macrocyclic systems frequently trigger `prune_failures`, ending up grouped at a different effective level than the rest.

    """

    splitter_id: ClassVar[str] = "scaffold_tree"
    strictness: ClassVar[Strictness] = Strictness.STRICT

    def __init__(
        self,
        *,
        level: int = 1,
        prune_rule: Literal["scaffold_tree", "min_rings", "peripheral_first"] = "scaffold_tree",
        max_rings: int = 12,
        include_chirality: bool = False,
        on_empty_scaffold: Literal["own_group", "shared_group", "discard", "raise"] = "own_group",
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.level = level
        self.prune_rule = prune_rule
        self.max_rings = max_rings
        self.include_chirality = include_chirality
        self.on_empty_scaffold = on_empty_scaffold
        self._validate_shared()
        if not (0 <= level <= max_rings):
            raise ParameterError(f"level must satisfy 0 <= level <= max_rings, got {level!r}")
        if prune_rule not in ("scaffold_tree", "min_rings", "peripheral_first"):
            raise ParameterError(f"invalid prune_rule: {prune_rule!r}")
        if max_rings < 1:
            raise ParameterError(f"max_rings must be >= 1, got {max_rings!r}")

    def _group_labels(self, ctx: _Context) -> IndexArray:
        mols = _require_mols(ctx, type(self).__name__)
        if self.level == 0:
            keys = [
                _scaffolds.murcko_scaffold(m, self.include_chirality) if m is not None else ""
                for m in mols
            ]
            self._prune_failures: list[int] = []
            self._too_complex: list[int] = []
        else:
            keys = []
            prune_failures = []
            too_complex = []
            for i, m in enumerate(mols):
                if m is None:
                    keys.append("")
                    continue
                smi, prune_failed = _scaffolds.scaffold_tree_levels(
                    m,
                    self.level,
                    prune_rule=self.prune_rule,
                    max_rings=self.max_rings,
                    include_chirality=self.include_chirality,
                )
                if smi == _scaffolds.SENTINEL_TOO_COMPLEX:
                    too_complex.append(i)
                if prune_failed:
                    prune_failures.append(i)
                keys.append(smi)
            self._prune_failures = prune_failures
            self._too_complex = too_complex
        keys, forced = _apply_on_empty_scaffold(keys, self.on_empty_scaffold, type(self).__name__)
        _merge_forced_discard(ctx, forced)
        labels_full = dense_label_encode(keys)
        keep_mask = np.ones(ctx.n, dtype=bool)
        if forced:
            keep_mask[forced] = False
        _check_degenerate(
            len(set(labels_full[keep_mask].tolist())), int(keep_mask.sum()), type(self).__name__
        )
        return labels_full

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        forced = ctx.extra.get("forced_discard", [])
        keep_mask = np.ones(ctx.n, dtype=bool)
        if forced:
            keep_mask[forced] = False
        meta = _group_size_metadata(labels, keep_mask)
        meta["level"] = self.level
        meta["prune_failures"] = getattr(self, "_prune_failures", [])
        meta["n_too_complex"] = len(getattr(self, "_too_complex", []))
        if self.level == 0:
            meta["delegated_to"] = "murcko_scaffold"
        return meta


# ---------------------------------------------------------------------------
# RingSystemSplitter
# ---------------------------------------------------------------------------


class RingSystemSplitter(_ScaffoldFamilyBase):
    """Group records by shared individual ring system(s), transitively.

    ``linkage="any_shared"`` unions any two molecules sharing >= 1 ring-system key via Union-Find
    (a molecule can belong to a component through a chain of shared rings, not just a direct
    pairwise match). ``linkage="all_shared"`` instead hashes the full sorted multiset of a
    molecule's ring-system keys — molecules join only if their ring-system content is identical.

    Advantages
    ----------
    - Targets novel ring chemistry directly — the exact claim behind most "scaffold hopping" results.
    - `any_shared` linkage catches a leak framework-level splits can't see: two molecules with different overall scaffolds sharing one highly characteristic ring system.
    - `csk` and `ring_size_profile` give progressively coarser, stricter variants without changing the algorithm.

    Pitfalls
    --------
    - With `any_shared` linkage, ubiquitous rings (benzene, pyridine, piperazine) can merge most of the dataset into one component — always check `metadata["largest_component_frac"]`.
    - Transitive grouping isn't a distance — two molecules in the same group can be entirely dissimilar, linked only through a chain of shared rings.
    - Excluding common rings to avoid the giant component changes what the split means; do it only if you say so, since it's no longer a plain ring-system split.
    - Spiro-merging policy affects group identity — fixed here at >=1 shared atom, but other toolkits differ.

    """

    splitter_id: ClassVar[str] = "ring_system"
    strictness: ClassVar[Strictness] = Strictness.MODERATE

    def __init__(
        self,
        *,
        key: Literal["ring_system", "csk", "ring_size_profile"] = "ring_system",
        linkage: Literal["any_shared", "all_shared"] = "any_shared",
        min_ring_size: int = 3,
        max_ring_size: int = 20,
        include_chirality: bool = False,
        on_empty_scaffold: Literal["own_group", "shared_group", "discard", "raise"] = "own_group",
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.key = key
        self.linkage = linkage
        self.min_ring_size = min_ring_size
        self.max_ring_size = max_ring_size
        self.include_chirality = include_chirality
        self.on_empty_scaffold = on_empty_scaffold
        self._validate_shared()
        if key not in ("ring_system", "csk", "ring_size_profile"):
            raise ParameterError(f"invalid key: {key!r}")
        if linkage not in ("any_shared", "all_shared"):
            raise ParameterError(f"invalid linkage: {linkage!r}")

    def _ring_keys_for_mol(self, mol: Chem.rdchem.Mol) -> set[str]:
        if self.key == "ring_system":
            return set(
                _scaffolds.ring_systems(mol, self.min_ring_size, self.max_ring_size)
            )
        if self.key == "csk":
            # CSK applied per individual ring system: extract each ring system as its own
            # submolecule (reusing ring_systems' fused/spiro merge), then generic+saturate it.
            keys = set()
            for smi in _scaffolds.ring_systems(mol, self.min_ring_size, self.max_ring_size):
                sub = Chem.MolFromSmiles(smi)
                if sub is None:
                    continue
                keys.add(_scaffolds.csk(sub))
            return keys
        # ring_size_profile: sorted tuple of ring sizes per ring system, stringified.
        ri = mol.GetRingInfo()
        rings = [
            set(r) for r in ri.AtomRings() if self.min_ring_size <= len(r) <= self.max_ring_size
        ]
        if not rings:
            return set()
        uf = UnionFind(len(rings))
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                if rings[i] & rings[j]:
                    uf.union(i, j)
        comps = uf.components()
        out = set()
        for members in comps.values():
            sizes = sorted(len(rings[m]) for m in members)
            out.add(str(tuple(sizes)))
        return out

    def _group_labels(self, ctx: _Context) -> IndexArray:
        mols = _require_mols(ctx, type(self).__name__)
        n = ctx.n
        ring_keys: list[set[str]] = []
        for m in mols:
            ring_keys.append(self._ring_keys_for_mol(m) if m is not None else set())

        if self.linkage == "all_shared":
            key_strs = [str(tuple(sorted(ks))) for ks in ring_keys]
            key_strs, forced = _apply_on_empty_scaffold(
                key_strs, self.on_empty_scaffold, type(self).__name__
            )
            _merge_forced_discard(ctx, forced)
            labels_full = dense_label_encode(key_strs)
        else:
            empty_idx = [i for i, ks in enumerate(ring_keys) if not ks]
            _, forced = _apply_on_empty_scaffold(
                ["" for _ in empty_idx], self.on_empty_scaffold, type(self).__name__
            )
            forced_set = {empty_idx[k] for k in range(len(forced))} if forced else set()
            _merge_forced_discard(ctx, sorted(forced_set))

            uf = UnionFind(n)
            owner: dict[str, int] = {}
            for i in range(n):
                for k in sorted(ring_keys[i]):
                    if k in owner:
                        uf.union(i, owner[k])
                    else:
                        owner[k] = i
            reps = [uf.find(i) for i in range(n)]
            if self.on_empty_scaffold == "own_group":
                # Give each empty-ring-set record (which never unioned with anything) a unique key
                # so it truly becomes its own singleton, matching the shared block's contract.
                next_singleton = n
                for i in empty_idx:
                    reps[i] = next_singleton
                    next_singleton += 1
            labels_full = dense_label_encode(reps)

        keep_mask = np.ones(n, dtype=bool)
        forced_now = ctx.extra.get("forced_discard", [])
        if forced_now:
            keep_mask[forced_now] = False
        n_kept = int(keep_mask.sum())
        kept_labels = labels_full[keep_mask]
        if n_kept > 0:
            counts = np.bincount(kept_labels)
            largest_frac = float(counts.max()) / float(n_kept)
            if largest_frac > 0.95:
                raise DegenerateGroupingError(
                    f"{type(self).__name__}: largest component holds {largest_frac:.1%} of "
                    "records (>95%, degenerate)"
                )
            if largest_frac > 0.60:
                warn_with_details(
                    DegenerateClusterWarning(
                        f"{type(self).__name__}: largest component holds {largest_frac:.1%} of "
                        "records",
                        details={"largest_component_frac": largest_frac},
                    )
                )
            self._largest_component_frac = largest_frac
        else:
            self._largest_component_frac = 0.0
        _check_degenerate(len(set(kept_labels.tolist())), n_kept, type(self).__name__)
        return labels_full

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        forced = ctx.extra.get("forced_discard", [])
        keep_mask = np.ones(ctx.n, dtype=bool)
        if forced:
            keep_mask[forced] = False
        meta = _group_size_metadata(labels, keep_mask)
        meta["n_ring_systems"] = meta.pop("n_groups")
        meta["n_groups"] = meta["n_ring_systems"]
        meta["largest_component_frac"] = getattr(self, "_largest_component_frac", 0.0)
        meta["linkage"] = self.linkage
        return meta


# ---------------------------------------------------------------------------
# MatchedMolecularSeriesSplitter
# ---------------------------------------------------------------------------


def _heavy_atoms(smiles_frag: str) -> int:
    mol = Chem.MolFromSmiles(smiles_frag, sanitize=False)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


class MatchedMolecularSeriesSplitter(_ScaffoldFamilyBase):
    """Group molecules sharing a matched-molecular-pair constant context.

    Fragments every molecule on up to ``max_cuts`` acyclic single bonds via RDKit's MMPA
    (``rdkit.Chem.rdMMPA.FragmentMol``); a constant context shared by >= ``min_series_size``
    distinct molecules defines a series, unioned via Union-Find. ``enforce="discard_boundary"``
    instead runs a Murcko-scaffold split and then moves any test record sharing a context with a
    train record to ``discard``.

    Advantages
    ----------
    - The closest available test of "did the model learn the SAR, or memorise the series?" — it removes exactly the analogue pairs that make a nearest-neighbour baseline look competitive.
    - Chemically interpretable — every group can be explained as "these share a constant context".
    - `discard_boundary` mode keeps a familiar Murcko-split size profile while surgically removing the analogue leak, often the best practical compromise.

    Pitfalls
    --------
    - Enumeration cost is the real constraint — `max_cuts>1` on sets above roughly 50,000 molecules is impractical, and the ceiling exists so the failure is loud rather than a hung process.
    - Congeneric libraries collapse into one enormous group, making the requested ratio unreachable — check `largest_group_frac` before trusting the split.
    - MMP detection is sensitive to `min_constant_heavy_atoms` — too low and sharing a methyl counts as a "series", too high and real series are missed. The default of 5 is a convention, not a rule.
    - `discard_boundary` throws away exactly the records nearest the boundary, biasing the remaining test set to be harder than a uniform sample of held-out chemistry.
    - Ignores stereochemistry unless `include_chirality=True` — enantiomeric pairs otherwise look like the same molecule to the fragmenter.

    """

    splitter_id: ClassVar[str] = "matched_molecular_series"
    strictness: ClassVar[Strictness] = Strictness.STRICT
    extras: ClassVar[tuple[str,...]] = ("mmpa",)

    def __init__(
        self,
        *,
        max_cuts: int = 1,
        max_variable_heavy_atoms: int = 10,
        min_constant_heavy_atoms: int = 5,
        min_series_size: int = 2,
        fragment_symmetry: bool = True,
        enforce: Literal["group", "discard_boundary"] = "group",
        max_pairs: int | None = 5_000_000,
        include_chirality: bool = False,
        on_empty_scaffold: Literal["own_group", "shared_group", "discard", "raise"] = "own_group",
        size_tolerance: float = 0.05,
        group_assignment: Literal["greedy_desc", "balanced", "random"] = "greedy_desc",
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            size_tolerance=size_tolerance,
            group_assignment=group_assignment,
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.max_cuts = max_cuts
        self.max_variable_heavy_atoms = max_variable_heavy_atoms
        self.min_constant_heavy_atoms = min_constant_heavy_atoms
        self.min_series_size = min_series_size
        self.fragment_symmetry = fragment_symmetry
        self.enforce = enforce
        self.max_pairs = max_pairs
        self.include_chirality = include_chirality
        self.on_empty_scaffold = on_empty_scaffold
        self._validate_shared()
        if not (1 <= max_cuts <= 3):
            raise ParameterError(f"max_cuts must satisfy 1 <= max_cuts <= 3, got {max_cuts!r}")
        if min_series_size < 2:
            raise ParameterError(f"min_series_size must be >= 2, got {min_series_size!r}")
        if enforce not in ("group", "discard_boundary"):
            raise ParameterError(f"invalid enforce: {enforce!r}")
        if max_pairs is not None and max_pairs < 1:
            raise ParameterError(f"max_pairs must be >= 1 or None, got {max_pairs!r}")

    def _canonicalise_context(self, constant: str) -> str:
        if not self.fragment_symmetry:
            return constant
        # Canonicalise attachment-point ([*]) ordering by round-tripping through RDKit's
        # canonical SMILES, which already normalises equivalent attachment-point numbering.
        mol = Chem.MolFromSmiles(constant, sanitize=False)
        if mol is None:
            return constant
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return constant

    def _build_contexts(self, mols: list) -> dict[str, list[tuple[int, str]]]:
        contexts: dict[str, list[tuple[int, str]]] = {}
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            try:
                fragments = rdMMPA.FragmentMol(
                    mol, maxCuts=self.max_cuts, resultsAsMols=False
                )
            except Exception:
                continue
            for pair in fragments:
                # rdMMPA.FragmentMol returns (core_smiles, chains_smiles) pairs. Verified directly
                # (not assumed): for a single-bond cut, `core` is EMPTY and `chains` packs BOTH
                # resulting fragments into one dot-joined SMILES (e.g. ('', 'C[*:1].c1ccc([*:1])
                # cc1')) -- there is no "remaining scaffold" concept for a single cut. Only for
                # multi-bond cuts (max_cuts>=2) does `core` become the genuine multi-attachment-
                # point shared context. Handle both shapes: single-cut splits `chains` on '.' and
                # picks the larger-by-heavy-atoms piece as the constant context (the smaller as the
                # variable substituent); multi-cut treats `core` as the constant directly.
                if not isinstance(pair, tuple) or len(pair) < 2:
                    continue
                core, chains = pair[0], pair[1]
                if core:
                    constant, variable = core, chains
                else:
                    pieces = chains.split(".")
                    if len(pieces) != 2:
                        continue
                    a, b = pieces
                    constant, variable = (a, b) if _heavy_atoms(a) >= _heavy_atoms(b) else (b, a)
                if not constant or not variable:
                    continue
                if _heavy_atoms(variable) > self.max_variable_heavy_atoms:
                    continue
                if _heavy_atoms(constant) < self.min_constant_heavy_atoms:
                    continue
                c = self._canonicalise_context(constant)
                contexts.setdefault(c, []).append((i, variable))
        return contexts

    def _group_labels(self, ctx: _Context) -> IndexArray:
        mols = _require_mols(ctx, type(self).__name__)
        contexts = self._build_contexts(mols)
        uf = UnionFind(ctx.n)
        n_pairs = 0
        n_series = 0
        context_examples: list[str] = []
        for c in sorted(contexts.keys()):
            members = sorted({idx for idx, _ in contexts[c]})
            if len(members) < self.min_series_size:
                continue
            n_series += 1
            if len(context_examples) < 20:
                context_examples.append(c)
            n_pairs += len(members) * (len(members) - 1) // 2
            if self.max_pairs is not None and n_pairs > self.max_pairs:
                raise ScalabilityError(
                    f"{type(self).__name__}: matched-pair enumeration exceeded max_pairs="
                    f"{self.max_pairs} (n={ctx.n}, max_cuts={self.max_cuts}). Reduce max_cuts, "
                    "raise min_constant_heavy_atoms, or raise max_pairs."
                )
            for m in members[1:]:
                uf.union(members[0], m)
        labels_full = dense_label_encode([uf.find(i) for i in range(ctx.n)])
        self._n_series = n_series
        self._n_pairs = n_pairs
        self._context_examples = context_examples
        if self.enforce == "group":
            _check_degenerate(len(set(labels_full.tolist())), ctx.n, type(self).__name__)
        return labels_full

    def _group_metadata(self, ctx: _Context, labels: IndexArray) -> dict[str, Any]:
        forced = ctx.extra.get("forced_discard", [])
        keep_mask = np.ones(ctx.n, dtype=bool)
        if forced:
            keep_mask[forced] = False
        meta = _group_size_metadata(labels, keep_mask)
        meta["n_series"] = getattr(self, "_n_series", 0)
        meta["n_pairs"] = getattr(self, "_n_pairs", 0)
        meta["context_examples"] = getattr(self, "_context_examples", [])
        meta["boundary_discarded"] = getattr(self, "_boundary_discarded", 0)
        return meta

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        if self.enforce == "group":
            return super()._partition(ctx)

        # "discard_boundary": run a Murcko-scaffold split, then move any test record sharing a
        # matched-molecular context with any train record to discard.
        mols = _require_mols(ctx, type(self).__name__)
        murcko_keys = [
            _scaffolds.murcko_scaffold(m, self.include_chirality) if m is not None else ""
            for m in mols
        ]
        murcko_keys, forced = _apply_on_empty_scaffold(
            murcko_keys, self.on_empty_scaffold, type(self).__name__
        )
        _merge_forced_discard(ctx, forced)
        murcko_labels = dense_label_encode(murcko_keys)

        forced_discard = np.asarray(sorted(ctx.extra.get("forced_discard", [])), dtype=np.int64)
        keep_mask = np.ones(ctx.n, dtype=bool)
        keep_mask[forced_discard] = False
        keep_idx = np.nonzero(keep_mask)[0]
        labels_kept = dense_label_encode(murcko_labels[keep_idx].tolist())

        from chemsplit.base import assign_groups

        rng = self._rng_for_group_assignment(ctx)
        buckets_local = assign_groups(labels_kept, ctx.sizes, self.group_assignment, rng)
        train = keep_idx[buckets_local["train"]]
        valid = keep_idx[buckets_local.get("valid", np.array([], dtype=np.int64))]
        test = keep_idx[buckets_local["test"]]

        contexts = self._build_contexts(mols)
        train_set = set(train.tolist())
        boundary_records: set[int] = set()
        for members_list in contexts.values():
            members = {idx for idx, _ in members_list}
            if members & train_set and (members - train_set):
                boundary_records |= (members - train_set) & set(test.tolist())

        if boundary_records:
            test = np.array(sorted(set(test.tolist()) - boundary_records), dtype=np.int64)
            forced_discard = np.array(
                sorted(set(forced_discard.tolist()) | boundary_records), dtype=np.int64
            )
        self._boundary_discarded = len(boundary_records)

        if test.size == 0 and ctx.sizes.n_test > 0:
            from chemsplit.exceptions import EmptyPartitionError

            raise EmptyPartitionError(
                f"{type(self).__name__}: discard_boundary removed the entire test set"
            )

        self._n_series = sum(
            1
            for members_list in contexts.values()
            if len({idx for idx, _ in members_list}) >= self.min_series_size
        )
        self._n_pairs = 0
        self._context_examples = list(contexts.keys())[:20]

        result = self._build_result(
            ctx,
            train=train,
            valid=valid,
            test=test,
            discard=forced_discard,
            groups=murcko_labels,
            extra_metadata=self._group_metadata(ctx, murcko_labels),
        )
        self._check_size_tolerance(result, ctx)
        return [result]


# ---------------------------------------------------------------------------
# ActivityCliffSplitter (NOT group-forming)
# ---------------------------------------------------------------------------


class ActivityCliffSplitter(BaseSplitter):
    """Deliberately place activity-cliff compounds (similar structure, very different potency) in the
    test set and tag them for separate scoring.

    This is a diagnostic, not a general-purpose split — see Pitfalls.

    Advantages
    ----------
    - Isolates the failure mode that matters most in lead optimisation: two nearly identical molecules with very different potency.
    - `cliff_mask` lets the caller report cliff and non-cliff performance separately — the comparison, not the aggregate, is the point.
    - Four independent definitions of "structurally similar" let you check a conclusion isn't just a fingerprint artefact.

    Pitfalls
    --------
    - This is a **diagnostic, not a general-purpose split**. Reporting only cliff-set performance understates a model that's otherwise fine, and reporting only aggregate performance hides the cliff failure entirely — report both.
    - Almost every descriptor-based model collapses to near-random on cliffs — that's the expected result, not a bug.
    - Cliff detection is extremely sensitive to `similarity_threshold` and `fold_change_threshold` — moving the threshold from 0.9 to 0.85 can multiply the cliff count several-fold.
    - Needs `y` on a log scale for the fold-change semantics to be meaningful; the implementation can't check this.
    - Assay noise can manufacture fake cliffs — a 10-fold "cliff" from two single-shot measurements in different papers is measurement error, not chemistry. Aggregate replicates first and prefer single-assay data.
    - With `keep_cliff_partners_together=False`, a test compound's cliff partner sits in train, which is a much easier and different experiment — don't compare numbers across this setting.

    """

    splitter_id: ClassVar[str] = "activity_cliff"
    family: ClassVar[str] = "scaffold"
    strictness: ClassVar[Strictness] = Strictness.EXTRAPOLATIVE
    group_forming: ClassVar[bool] = False
    requires_labels: ClassVar[bool] = True
    accepts: ClassVar[tuple[str,...]] = ("smiles", "mol")
    deterministic_without_seed: ClassVar[bool] = False
    deterministic_method: ClassVar[bool] = True
    order_invariant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.9,
        fold_change_threshold: float = 10.0,
        y_scale: Literal["log", "linear"] = "log",
        similarity: Literal["ecfp", "scaffold", "mmp", "substructure"] = "ecfp",
        featurizer: str | Any = "ecfp4",
        metric: str = "tanimoto",
        cliff_target: Literal["test", "balanced"] = "test",
        keep_cliff_partners_together: bool = True,
        max_memory_bytes: int = 2 * 1024**3,
        allow_slow: bool = False,
        n_splits: int = 1,
        train_size: Any = None,
        valid_size: Any = None,
        test_size: Any = None,
        random_state: int | np.random.Generator | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_splits=n_splits,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.similarity_threshold = similarity_threshold
        self.fold_change_threshold = fold_change_threshold
        self.y_scale = y_scale
        self.similarity = similarity
        self.featurizer = featurizer
        self.metric = metric
        self.cliff_target = cliff_target
        self.keep_cliff_partners_together = keep_cliff_partners_together
        self.max_memory_bytes = max_memory_bytes
        # `allow_slow` is a constructor param rather than a `split()`-time kwarg because
        # `BaseSplitter._run` only forwards a fixed set of named kwargs — functionally
        # equivalent, just set at construction instead of per-call.
        self.allow_slow = allow_slow
        if not (0.0 < similarity_threshold < 1.0):
            raise ParameterError(
                f"similarity_threshold must satisfy 0 < t < 1, got {similarity_threshold!r}"
            )
        if fold_change_threshold <= 1.0:
            raise ParameterError(
                f"fold_change_threshold must be > 1, got {fold_change_threshold!r}"
            )
        if similarity not in ("ecfp", "scaffold", "mmp", "substructure"):
            raise ParameterError(f"invalid similarity: {similarity!r}")
        if y_scale not in ("log", "linear"):
            raise ParameterError(f"invalid y_scale: {y_scale!r}")
        if cliff_target not in ("test", "balanced"):
            raise ParameterError(f"invalid cliff_target: {cliff_target!r}")

    def _candidate_pairs(self, ctx: _Context, mols: list) -> list[tuple[int, int, float]]:
        n = ctx.n
        if self.similarity == "ecfp":
            from chemsplit._fp_similarity import compute_similarity_matrix, guard_memory
            from chemsplit.featurizers import get_featurizer

            guard_memory(n, self.max_memory_bytes, type(self).__name__)
            feat = get_featurizer(self.featurizer)
            S = compute_similarity_matrix(
                ctx, feat, self.metric, self.max_memory_bytes, type(self).__name__, self.n_jobs
            )
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    if S[i, j] > self.similarity_threshold + _EPS:
                        pairs.append((i, j, float(S[i, j])))
            return pairs
        if self.similarity == "scaffold":
            keys = [
                _scaffolds.murcko_scaffold(m) if m is not None else "" for m in mols
            ]
            by_key: dict[str, list[int]] = {}
            for i, k in enumerate(keys):
                by_key.setdefault(k, []).append(i)
            pairs = []
            for members in by_key.values():
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        pairs.append((members[a], members[b], 1.0))
            return pairs
        if self.similarity == "mmp":
            contexts: dict[str, list[int]] = {}
            for i, mol in enumerate(mols):
                if mol is None:
                    continue
                try:
                    frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
                except Exception:
                    continue
                for pair in frags:
                    if not isinstance(pair, tuple) or len(pair) < 2:
                        continue
                    contexts.setdefault(pair[0], []).append(i)
            pairs = []
            for members in contexts.values():
                members = sorted(set(members))
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        pairs.append((members[a], members[b], 1.0))
            return pairs
        # substructure
        if n > 5000 and not self.allow_slow:
            raise ScalabilityError(
                f"{type(self).__name__}: similarity='substructure' is O(n^2) substructure "
                f"matches, refused for n={n} > 5000 unless allow_slow=True"
            )
        pairs = []
        for i in range(n):
            mi = mols[i]
            if mi is None:
                continue
            for j in range(i + 1, n):
                mj = mols[j]
                if mj is None:
                    continue
                if mi.HasSubstructMatch(mj) or mj.HasSubstructMatch(mi):
                    pairs.append((i, j, 1.0))
        return pairs

    def _label_gap(self, yi: float, yj: float) -> float:
        if self.y_scale == "log":
            return abs(yi - yj)
        lo, hi = min(yi, yj), max(yi, yj)
        return hi / lo if lo > 0 else float("inf")

    def _gap_threshold(self) -> float:
        import math

        if self.y_scale == "log":
            return math.log10(self.fold_change_threshold)
        return self.fold_change_threshold

    def _partition(self, ctx: _Context) -> list[SplitResult]:
        mols = _require_mols(ctx, type(self).__name__)
        y = np.asarray(ctx.y, dtype=np.float64)
        candidates = self._candidate_pairs(ctx, mols)
        gap_threshold = self._gap_threshold()
        cliffs = [
            (i, j, sim)
            for (i, j, sim) in candidates
            if self._label_gap(y[i], y[j]) >= gap_threshold - _EPS
        ]
        cliff_nodes = sorted({i for i, j, _ in cliffs} | {j for i, j, _ in cliffs})
        if not cliff_nodes:
            max_sim = max((s for _, _, s in candidates), default=0.0)
            max_gap = max(
                (self._label_gap(y[i], y[j]) for i, j, _ in candidates), default=0.0
            )
            raise ConstraintUnsatisfiableError(
                f"{type(self).__name__}: no activity cliffs found at similarity_threshold="
                f"{self.similarity_threshold}, fold_change_threshold={self.fold_change_threshold} "
                f"(observed max similarity among candidates={max_sim:.3f}, max label gap among "
                f"similar pairs={max_gap:.3f})"
            )

        cliff_set = set(cliff_nodes)
        if self.keep_cliff_partners_together:
            uf = UnionFind(ctx.n)
            for i, j, _ in cliffs:
                uf.union(i, j)
            # Build units: cliff components as-is; every other record its own singleton unit.
            units: dict[int, list[int]] = {}
            for i in range(ctx.n):
                r = uf.find(i) if i in cliff_set else i
                units.setdefault(r, []).append(i)
        else:
            units = {i: [i] for i in range(ctx.n)}

        cliff_unit_reps = sorted(
            {r for r, members in units.items() if set(members) & set(cliff_nodes)}
        )
        noncliff_unit_reps = sorted(set(units.keys()) - set(cliff_unit_reps))

        rng = seed_for(ctx.rng_seeds, "cliff.assign", 0)

        def unit_size(r: int) -> int:
            return len(units[r])

        from chemsplit.determinism import stable_sort

        cliff_order = stable_sort(cliff_unit_reps, key=unit_size, desc=True)
        noncliff_order = [
            int(r) for r in rng.permutation(np.asarray(noncliff_unit_reps, dtype=np.int64))
        ] if noncliff_unit_reps else []

        n_train, n_valid, n_test = ctx.sizes.n_train, ctx.sizes.n_valid, ctx.sizes.n_test
        train: list[int] = []
        valid: list[int] = []
        test: list[int] = []
        overflow = 0

        if self.cliff_target == "test":
            order = cliff_order + noncliff_order
            for r in order:
                members = units[r]
                if r in cliff_unit_reps and len(test) < n_test:
                    test.extend(members)
                elif len(test) < n_test:
                    test.extend(members)
                elif len(valid) < n_valid:
                    valid.extend(members)
                    if r in cliff_unit_reps:
                        overflow += len(members)
                else:
                    train.extend(members)
                    if r in cliff_unit_reps:
                        overflow += len(members)
        else:  # balanced
            buckets = [("train", n_train), ("valid", n_valid), ("test", n_test)]
            buckets = [b for b in buckets if b[1] > 0]
            counts = {name: 0 for name, _ in buckets}
            dest = {"train": train, "valid": valid, "test": test}
            for r in cliff_order + noncliff_order:
                members = units[r]
                cand_name = argmax_tiebreak(
                    lambda b: b[1] - counts[b[0]], buckets
                )[0]
                dest[cand_name].extend(members)
                counts[cand_name] += len(members)

        train_arr = np.sort(np.asarray(train, dtype=np.int64))
        valid_arr = np.sort(np.asarray(valid, dtype=np.int64))
        test_arr = np.sort(np.asarray(test, dtype=np.int64))
        assigned = set(train) | set(valid) | set(test)
        discard_arr = np.array(sorted(set(range(ctx.n)) - assigned), dtype=np.int64)

        cliff_mask = [i in cliff_set for i in range(ctx.n)]
        cliff_pairs_meta = [
            [int(i), int(j), float(sim), float(self._label_gap(y[i], y[j]))]
            for i, j, sim in sorted(cliffs)
        ]

        metadata = {
            "n_cliff_pairs": len(cliffs),
            "n_cliff_records": len(cliff_nodes),
            "cliff_mask": [bool(c) for c in cliff_mask],
            "cliff_pairs": cliff_pairs_meta,
            "cliff_overflow": int(overflow),
            "y_range": [float(np.min(y)), float(np.max(y))],
            "realised_sizes": {
                "train": int(train_arr.size),
                "valid": int(valid_arr.size),
                "test": int(test_arr.size),
            },
        }
        return [
            SplitResult(
                train=train_arr,
                valid=valid_arr,
                test=test_arr,
                discard=discard_arr,
                groups=None,
                splitter_id=self.splitter_id,
                params=self.get_params(),
                n_records=ctx.n,
                metadata=metadata,
            )
        ]
