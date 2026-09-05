"""Scaffold, generic framework, wireframe (CSK), ring system and scaffold-tree computation."""

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
    """Murcko scaffold of ``mol``, via ``scaffound.get_basic_scaffold``.

    Not guaranteed byte-identical to RDKit's classic Bemis-Murcko scaffold in every case.

    :param mol: the molecule.
    :param include_chirality: keep stereochemistry in the output SMILES.
    :return: the scaffold's canonical SMILES, or ``""`` for an acyclic molecule.
    """
    import scaffound

    scaf = scaffound.get_basic_scaffold(mol)
    return _mol_to_smiles(scaf, isomeric=include_chirality)


def generic_scaffold(mol: Chem.rdchem.Mol) -> str:
    """Generic framework of ``mol``, with every heteroatom made carbon.

    :param mol: the molecule.
    :return: the framework's canonical SMILES, always non-isomeric, since a generic scaffold
        carries no stereochemistry.
    """
    import scaffound

    fw = scaffound.get_basic_framework(mol)
    return _mol_to_smiles(fw, isomeric=False)


def csk(mol: Chem.rdchem.Mol) -> str:
    """Cyclic skeleton of ``mol``: the framework made generic and saturated.

    :param mol: the molecule.
    :return: the skeleton's canonical SMILES.
    """
    import scaffound

    wf = scaffound.get_basic_wireframe(mol)
    return _mol_to_smiles(wf, isomeric=False)


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
    """Extract each fused or spiro-merged ring system in ``mol`` separately.

    Two SSSR rings merge into one system when they share at least one atom, which covers both
    fused rings, sharing two or more, and spiro rings, sharing exactly one. Exocyclic double
    bonds on a ring atom stay in the extracted submolecule.

    :param mol: the molecule.
    :param min_ring_size: rings smaller than this are excluded before merging.
    :param max_ring_size: rings larger than this are excluded before merging.
    :return: one canonical SMILES per ring system, in ascending first-atom order.
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
            # shouldn't happen for a real ring, but guard anyway
            continue
        submol = Chem.PathToSubmol(mol, bond_indices)
        try:
            Chem.SanitizeMol(submol)
        except Exception:
            pass
        out.append(Chem.MolToSmiles(submol, canonical=True))
    return sorted(out)


#: Stand-in key for a molecule with more rings than ``max_rings``, which is left unpruned.
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


def _cluster_adjacency(
    mol: Chem.rdchem.Mol, clusters: list[tuple[frozenset, tuple]]
) -> dict[int, set[int]]:
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

    # scaffold_tree: a best-effort Schuffenhauer-style heuristic. Smaller peripheral rings
    # go first, then the smallest atom-index tuple. Finer rules, such as preferring
    # linker-attached over fusion-attached rings, are not implemented.
    ordered = sorted(peripheral, key=lambda i: (len(clusters[i][1]), clusters[i][1]))
    return clusters[ordered[0]]


def _remove_cluster(mol: Chem.rdchem.Mol, atoms_to_remove: frozenset) -> Chem.rdchem.Mol | None:
    rw = Chem.RWMol(mol)
    for idx in sorted(atoms_to_remove, reverse=True):
        rw.RemoveAtom(idx)
    candidate = rw.GetMol()
    frags = Chem.GetMolFrags(candidate, asMols=True, sanitizeFrags=False)
    # sanitizeFrags=False leaves ring perception uninitialised, so GetRingInfo() would raise.
    # FastFindRings works on an otherwise unsanitisable fragment; full SanitizeMol waits for
    # the single survivor below.
    for f in frags:
        Chem.FastFindRings(f)
    kept = [f for f in frags if f.GetRingInfo().NumRings() > 0]
    if not kept:
        return None
    # keep the largest ring-bearing fragment; a second one shouldn't arise from a connected
    # molecule, but guard anyway
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
    """Prune up to ``level`` peripheral rings off the Murcko scaffold.

    .. warning::
       ``prune_rule="scaffold_tree"`` is a best-effort reconstruction of Schuffenhauer-style
       prioritisation -- peripheral rings first, then smaller rings, then the smallest
       atom-index tuple -- and is provisional rather than normative.

    :param mol: the molecule.
    :param level: how many rings to remove. ``0`` reproduces :func:`murcko_scaffold`.
    :param prune_rule: which ring to remove at each step.
    :param max_rings: molecules with more rings than this are left unpruned.
    :param include_chirality: keep stereochemistry in the output SMILES.
    :return: the pruned scaffold's canonical SMILES, and whether pruning fell short of
        ``level``.
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
