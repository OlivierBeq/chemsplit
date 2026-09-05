"""Scaffold, generic-framework, wireframe (CSK), ring-system, and scaffold-tree computation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from rdkit import Chem

__all__ = [
    "csk",
    "generic_scaffold",
    "murcko_scaffold",
    "ring_systems",
    "scaffold_tree_levels",
]


def _mol_to_smiles(mol: Chem.rdchem.Mol | None, isomeric: bool) -> str:
    if mol is None or mol.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def murcko_scaffold(mol: Chem.rdchem.Mol, include_chirality: bool = False) -> str:
    """Canonical SMILES of ``scaffound.get_basic_scaffold(mol)``, or ``""`` if acyclic.

    Not guaranteed byte-identical to RDKit's classic Bemis-Murcko scaffold in all cases.
    """
    import scaffound

    scaf = scaffound.get_basic_scaffold(mol)
    return _mol_to_smiles(scaf, isomeric=include_chirality)


def generic_scaffold(mol: Chem.rdchem.Mol) -> str:
    """Canonical SMILES of ``scaffound.get_basic_framework(mol)`` (heteroatoms -> carbon).

    Always non-isomeric: a generic scaffold has no stereochemistry.
    """
    import scaffound

    fw = scaffound.get_basic_framework(mol)
    return _mol_to_smiles(fw, isomeric=False)


def csk(mol: Chem.rdchem.Mol) -> str:
    """Canonical SMILES of ``scaffound.get_basic_wireframe(mol)`` (generic AND saturated).

    This is the "Cyclic Skeleton Key" concept used by the ring-system splitter.
    """
    import scaffound

    wf = scaffound.get_basic_wireframe(mol)
    return _mol_to_smiles(wf, isomeric=False)


# --------------------------------------------------------------------------------------
# Per-molecule individual ring systems (fused/spiro merged), not covered by scaffound.
# --------------------------------------------------------------------------------------


def _union_find_merge(n: int, pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in pairs:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def ring_systems(
    mol: Chem.rdchem.Mol, min_ring_size: int = 3, max_ring_size: int = 20
) -> list[str]:
    """Canonical SMILES of each individual fused/spiro-merged ring system in ``mol``.

    Two SSSR rings merge into one system if they share >= 1 atom (covers both fused, sharing >= 2
    atoms, and spiro, sharing exactly 1 atom). Rings outside ``[min_ring_size, max_ring_size]`` are
    excluded before merging. Exocyclic double bonds (e.g. carbonyls) attached to a ring atom are
    kept in the extracted submolecule.
    """
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings() if min_ring_size <= len(r) <= max_ring_size]
    if not rings:
        return []

    pairs = [
        (i, j)
        for i in range(len(rings))
        for j in range(i + 1, len(rings))
        if rings[i] & rings[j]
    ]
    groups = _union_find_merge(len(rings), pairs)

    out = []
    for group in groups:
        atoms: set[int] = set()
        for ring_idx in group:
            atoms |= rings[ring_idx]

        # Keep exocyclic atoms attached to a ring atom by a double bond (preserves carbonyls etc.).
        extra: set[int] = set()
        for a_idx in atoms:
            atom = mol.GetAtomWithIdx(a_idx)
            for bond in atom.GetBonds():
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    other = bond.GetOtherAtomIdx(a_idx)
                    if other not in atoms:
                        extra.add(other)
        submol_atoms = atoms | extra

        bond_indices = [
            b.GetIdx()
            for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in submol_atoms and b.GetEndAtomIdx() in submol_atoms
        ]
        if not bond_indices:
            # Single-atom ring system edge case shouldn't occur for real rings, but guard anyway.
            continue
        submol = Chem.PathToSubmol(mol, bond_indices)
        try:
            Chem.SanitizeMol(submol)
        except Exception:
            pass
        out.append(Chem.MolToSmiles(submol, canonical=True))
    return sorted(out)


# --------------------------------------------------------------------------------------
# Leveled scaffold-tree pruning (best-effort reconstruction — see docstring).
# --------------------------------------------------------------------------------------

SENTINEL_TOO_COMPLEX = "\x00TOO_COMPLEX\x00"


def _ring_count(mol: Chem.rdchem.Mol) -> int:
    try:
        return mol.GetRingInfo().NumRings()
    except RuntimeError:
        # scaffound's submols aren't always sanitized enough to have ring info precomputed
        # (RDKit's "RingInfo not initialized" precondition); GetSSSR() computes and caches it.
        Chem.GetSSSR(mol)
        return mol.GetRingInfo().NumRings()


def _ring_clusters(mol: Chem.rdchem.Mol) -> list[tuple[frozenset, tuple]]:
    """Fused/spiro-merged ring clusters as (atom_frozenset, sorted_atom_tuple) pairs."""
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if not rings:
        return []
    pairs = [
        (i, j) for i in range(len(rings)) for j in range(i + 1, len(rings)) if rings[i] & rings[j]
    ]
    groups = _union_find_merge(len(rings), pairs)
    out = []
    for group in groups:
        atoms: set[int] = set()
        for idx in group:
            atoms |= rings[idx]
        out.append((frozenset(atoms), tuple(sorted(atoms))))
    return out


def _cluster_adjacency(mol: Chem.rdchem.Mol, clusters: list[tuple[frozenset, tuple]]) -> dict[int, set[int]]:
    """Two clusters are adjacent if they share atoms (already merged, so never here) or are
    connected by a path of non-ring atoms with no intermediate ring atoms from a third cluster.
    """
    all_ring_atoms: set[int] = set()
    for atoms, _ in clusters:
        all_ring_atoms |= atoms

    adjacency: dict[int, set[int]] = {i: set() for i in range(len(clusters))}
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            connected = False
            for a in clusters[i][0]:
                atom = mol.GetAtomWithIdx(a)
                for bond in atom.GetBonds():
                    other = bond.GetOtherAtomIdx(a)
                    if other in clusters[j][0]:
                        connected = True
                        break
                    if other not in all_ring_atoms:
                        # walk the linker chain (bounded length) looking for cluster j
                        seen = {a, other}
                        frontier = [other]
                        for _ in range(30):  # generous bound on linker length
                            nxt = []
                            for f in frontier:
                                for b2 in mol.GetAtomWithIdx(f).GetBonds():
                                    o2 = b2.GetOtherAtomIdx(f)
                                    if o2 in seen:
                                        continue
                                    if o2 in clusters[j][0]:
                                        connected = True
                                        break
                                    if o2 not in all_ring_atoms:
                                        seen.add(o2)
                                        nxt.append(o2)
                                if connected:
                                    break
                            if connected or not nxt:
                                break
                            frontier = nxt
                    if connected:
                        break
                if connected:
                    break
            if connected:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def _select_ring_to_remove(
    mol: Chem.rdchem.Mol, prune_rule: str
) -> tuple[frozenset, tuple] | None:
    clusters = _ring_clusters(mol)
    if len(clusters) <= 1:
        return None
    adjacency = _cluster_adjacency(mol, clusters)
    peripheral = [i for i in range(len(clusters)) if len(adjacency[i]) <= 1]
    if not peripheral:
        peripheral = list(range(len(clusters)))

    if prune_rule == "peripheral_first":
        best = min(peripheral, key=lambda i: clusters[i][1])
        return clusters[best]

    if prune_rule == "min_rings":
        best_smiles = None
        best_i = None
        for i in peripheral:
            candidate = _remove_cluster(mol, clusters[i][0])
            if candidate is None:
                continue
            smi = Chem.MolToSmiles(candidate, canonical=True)
            if best_smiles is None or smi < best_smiles:
                best_smiles = smi
                best_i = i
        if best_i is None:
            return None
        return clusters[best_i]

    # "scaffold_tree": best-effort Schuffenhauer-style heuristic reconstructed from commonly
    # published prioritisation: prefer removing smaller peripheral rings first, then smallest
    # atom-index tuple as the final tie-break. Finer-grained rules (e.g. linker- vs
    # fusion-attached preference) are not implemented with full confidence.
    ordered = sorted(peripheral, key=lambda i: (len(clusters[i][1]), clusters[i][1]))
    return clusters[ordered[0]]


def _remove_cluster(mol: Chem.rdchem.Mol, atoms_to_remove: frozenset) -> Chem.rdchem.Mol | None:
    rw = Chem.RWMol(mol)
    for idx in sorted(atoms_to_remove, reverse=True):
        rw.RemoveAtom(idx)
    candidate = rw.GetMol()
    frags = Chem.GetMolFrags(candidate, asMols=True, sanitizeFrags=False)
    # `sanitizeFrags=False` leaves each fragment's ring perception uninitialised -- calling
    # GetRingInfo() on one directly raises a RDKit "Pre-condition Violation" RuntimeError. Perceive
    # rings explicitly (this does not require the fragment to be otherwise sanitisable, unlike
    # Chem.SanitizeMol, which is deliberately deferred to the single survivor below).
    for f in frags:
        Chem.FastFindRings(f)
    kept = [f for f in frags if f.GetRingInfo().NumRings() > 0]
    if not kept:
        return None
    # Keep the fragment(s) still containing >= 1 ring; if several, keep the largest (by atom
    # count) as the "main" scaffold body — a dangling second ring-bearing fragment is not expected
    # for a scaffold derived from a single connected molecule, but guard defensively.
    best = max(kept, key=lambda f: f.GetNumAtoms())
    try:
        Chem.SanitizeMol(best)
    except Exception:
        return None
    return best


def scaffold_tree_levels(
    mol: Chem.rdchem.Mol,
    level: int,
    prune_rule: Literal["scaffold_tree", "min_rings", "peripheral_first"] = "scaffold_tree",
    max_rings: int = 12,
    include_chirality: bool = False,
) -> tuple[str, bool]:
    """Canonical SMILES of the scaffold after removing up to ``level`` peripheral rings.

    Returns ``(smiles, prune_failed)``. ``level=0`` is exactly :func:`murcko_scaffold`.

    .. warning::
       ``prune_rule="scaffold_tree"`` is a best-effort reconstruction of Schuffenhauer-style
       ring-removal prioritisation (peripheral rings first, then smaller rings, then smallest
       atom-index tuple) and should be treated as provisional, not normative.
    """
    import scaffound

    s0 = scaffound.get_basic_scaffold(mol)
    if s0 is None or _ring_count(s0) > max_rings:
        return SENTINEL_TOO_COMPLEX, False

    s = s0
    prune_failed = False
    for _ in range(level):
        if _ring_count(s) <= 1:
            break
        chosen = _select_ring_to_remove(s, prune_rule)
        if chosen is None:
            break
        candidate = _remove_cluster(s, chosen[0])
        if candidate is None:
            prune_failed = True
            break
        s = candidate

    return _mol_to_smiles(s, isomeric=include_chirality), prune_failed
