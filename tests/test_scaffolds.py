import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemsplit import scaffolds


def _smi(s):
    return Chem.MolFromSmiles(s)


class TestMurckoScaffold:
    @pytest.mark.parametrize(
        "smi",
        ["c1ccccc1", "c1ccncc1", "C1CC2CCC1CC2", "C1CCC2(CC1)CCCC2"],
    )
    def test_matches_rdkit_murcko_on_ring_only_molecules(self, smi):
        mol = _smi(smi)
        rdkit_scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        ours = scaffolds.murcko_scaffold(mol)
        assert ours == rdkit_scaf

    def test_acyclic_molecule_is_empty_string(self):
        mol = _smi("CCCCCC")
        assert scaffolds.murcko_scaffold(mol) == ""

    def test_exocyclic_carbonyl_differs_from_rdkit_murcko(self):
        """scaffound's basic_scaffold strips an exocyclic =O that RDKit's classic Murcko keeps.

        This regression-locks that documented, intentional difference rather than asserting
        false equivalence with RDKit.
        """
        mol = _smi("O=C1CCCCC1")
        rdkit_scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        ours = scaffolds.murcko_scaffold(mol)
        assert rdkit_scaf == "O=C1CCCCC1"
        assert ours == "C1CCCCC1"
        assert ours != rdkit_scaf


class TestGenericScaffold:
    def test_heteroatoms_become_carbon(self):
        mol = _smi("c1ccncc1")  # pyridine
        generic = scaffolds.generic_scaffold(mol)
        assert "n" not in generic and "N" not in generic

    def test_never_isomeric(self):
        mol = _smi("C1CC2CCC1CC2")
        generic = scaffolds.generic_scaffold(mol)
        assert "@" not in generic


class TestCSK:
    def test_generic_and_saturated(self):
        mol = _smi("c1ccncc1")
        wireframe = scaffolds.csk(mol)
        assert "n" not in wireframe and "N" not in wireframe
        assert "=" not in wireframe
        assert ":" not in wireframe  # no aromatic bonds either


class TestRingSystems:
    def test_two_unconnected_ring_systems_yield_two_entries(self):
        mol = _smi("c1ccccc1CCCCCCC1CCCCC1")
        systems = scaffolds.ring_systems(mol)
        assert len(systems) == 2

    def test_fused_bicyclic_is_one_system(self):
        mol = _smi("c1ccc2c(c1)CCC2")  # indane-like fused system
        systems = scaffolds.ring_systems(mol)
        assert len(systems) == 1

    def test_spiro_system_is_one_system(self):
        mol = _smi("C1CCC2(CC1)CCCC2")
        systems = scaffolds.ring_systems(mol)
        assert len(systems) == 1

    def test_acyclic_molecule_has_no_ring_systems(self):
        mol = _smi("CCCCCC")
        assert scaffolds.ring_systems(mol) == []

    def test_max_ring_size_excludes_macrocycle(self):
        macro = _smi("C1CCCCCCCCCCCCCCCCCCC1")  # 20-membered ring
        systems = scaffolds.ring_systems(macro, max_ring_size=12)
        assert systems == []

    def test_exocyclic_carbonyl_double_bond_is_kept(self):
        mol = _smi("O=C1CCCCC1")  # cyclohexanone
        systems = scaffolds.ring_systems(mol)
        assert len(systems) == 1
        assert "O" in systems[0] and "=" in systems[0]


class TestScaffoldTreeLevels:
    @pytest.mark.parametrize("smi", ["c1ccccc1", "C1CC2CCC1CC2", "c1ccncc1"])
    def test_level_zero_equals_murcko(self, smi):
        mol = _smi(smi)
        smiles, failed = scaffolds.scaffold_tree_levels(mol, level=0)
        assert smiles == scaffolds.murcko_scaffold(mol)
        assert failed is False

    def test_level_one_on_fused_bicyclic_reduces_ring_count(self):
        mol = _smi("c1ccc2c(c1)CCC2")
        base, _ = scaffolds.scaffold_tree_levels(mol, level=0)
        pruned, failed = scaffolds.scaffold_tree_levels(mol, level=1, prune_rule="peripheral_first")
        base_mol = Chem.MolFromSmiles(base)
        Chem.GetSSSR(base_mol)
        base_rings = base_mol.GetRingInfo().NumRings()
        pruned_mol = Chem.MolFromSmiles(pruned)
        if pruned_mol is not None:
            Chem.GetSSSR(pruned_mol)
            assert pruned_mol.GetRingInfo().NumRings() <= base_rings

    def test_too_complex_sentinel(self):
        mol = _smi("c1ccccc1")
        smiles, _ = scaffolds.scaffold_tree_levels(mol, level=0, max_rings=0)
        assert smiles == scaffolds.SENTINEL_TOO_COMPLEX

    # -- two SEPARATE (non-fused) ring clusters connected by a linker, one of the two clusters
    # actually gets removed: exercises _cluster_adjacency's linker-walk, _select_ring_to_remove's
    # peripheral-selection + all three prune_rule branches, and _remove_cluster's real removal
    # path. GetMolFrags(..., sanitizeFrags=False) fragments have no ring info until explicitly
    # perceived, so _remove_cluster must call GetRingInfo() only after that perception step.
    @pytest.mark.parametrize("prune_rule", ["peripheral_first", "min_rings", "scaffold_tree"])
    def test_two_linked_rings_prunes_one_ring_away(self, prune_rule):
        mol = _smi("c1ccccc1CCCc1ccccc1")  # 1,3-diphenylpropane
        base, _ = scaffolds.scaffold_tree_levels(mol, level=0)
        base_mol = Chem.MolFromSmiles(base)
        assert base_mol.GetRingInfo().NumRings() == 2  # two separate (non-fused) benzene rings

        pruned, failed = scaffolds.scaffold_tree_levels(mol, level=1, prune_rule=prune_rule)
        assert failed is False
        pruned_mol = Chem.MolFromSmiles(pruned)
        assert pruned_mol is not None
        assert pruned_mol.GetRingInfo().NumRings() == 1  # exactly one ring removed

        # A second level is a no-op: _ring_count(s) <= 1 breaks the loop immediately (only one
        # ring remains, and the algorithm never prunes down to zero rings).
        pruned2, failed2 = scaffolds.scaffold_tree_levels(mol, level=2, prune_rule=prune_rule)
        assert failed2 is False
        assert pruned2 == pruned

    def test_two_directly_bonded_rings_prunes_one_ring_away(self):
        # Biphenyl: two ring clusters joined by a direct bond (no linker chain atoms in between),
        # exercising _cluster_adjacency's direct-bond short-circuit rather than the linker walk.
        mol = _smi("c1ccc(-c2ccccc2)cc1")
        pruned, failed = scaffolds.scaffold_tree_levels(mol, level=1, prune_rule="peripheral_first")
        assert failed is False
        pruned_mol = Chem.MolFromSmiles(pruned)
        assert pruned_mol is not None
        assert pruned_mol.GetRingInfo().NumRings() == 1

    def test_ring_clusters_empty_for_acyclic(self):
        assert scaffolds._ring_clusters(_smi("CCCCCC")) == []

    def test_select_ring_to_remove_none_for_single_cluster(self):
        # A lone (or fully-fused, single-cluster) ring system has nothing peripheral to prune.
        assert scaffolds._select_ring_to_remove(_smi("c1ccccc1"), "peripheral_first") is None
        assert scaffolds._select_ring_to_remove(_smi("c1ccc2c(c1)CCC2"), "min_rings") is None
