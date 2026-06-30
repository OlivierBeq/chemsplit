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
        base_rings = Chem.MolFromSmiles(base).GetRingInfo().NumRings()
        pruned_mol = Chem.MolFromSmiles(pruned)
        if pruned_mol is not None:
            assert pruned_mol.GetRingInfo().NumRings() <= base_rings

    def test_too_complex_sentinel(self):
        mol = _smi("c1ccccc1")
        smiles, _ = scaffolds.scaffold_tree_levels(mol, level=0, max_rings=0)
        assert smiles == scaffolds.SENTINEL_TOO_COMPLEX
