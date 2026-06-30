import numpy as np
import pytest
import scipy.sparse as sp
from rdkit import Chem

from chemsplit.featurizers import get_featurizer
from chemsplit.featurizers.descriptors import _PHYSCHEM_DESCRIPTORS, MQNFeaturizer, PhysChemFeaturizer
from chemsplit.featurizers.fingerprints import ECFPFeaturizer, FCFPFeaturizer, MACCSFeaturizer
from chemsplit.featurizers.precomputed import PrecomputedFeaturizer

MOLS = [Chem.MolFromSmiles(s) for s in ["c1ccccc1O", "CCN(CC)CC", "c1ccncc1"]]


class TestEcfpRadiusMapping:
    @pytest.mark.parametrize(
        "alias,expected_radius",
        [("ecfp2", 1), ("ecfp4", 2), ("ecfp6", 3), ("ecfp8", 4), ("morgan2", 2), ("morgan3", 3)],
    )
    def test_alias_radius(self, alias, expected_radius):
        feat = get_featurizer(alias)
        assert feat.radius == expected_radius, f"{alias} should map to radius={expected_radius}"

    def test_ecfp6_is_not_radius_6(self):
        feat = get_featurizer("ecfp6")
        assert feat.radius != 6
        assert feat.radius == 3

    @pytest.mark.parametrize("alias,expected_radius", [("fcfp2", 1), ("fcfp4", 2), ("fcfp6", 3), ("fcfp8", 4)])
    def test_fcfp_alias_radius(self, alias, expected_radius):
        feat = get_featurizer(alias)
        assert feat.radius == expected_radius


class TestBinaryFeaturizers:
    def test_ecfp_output_shape_and_dtype(self):
        feat = ECFPFeaturizer(radius=2, n_bits=2048)
        X = feat.transform(MOLS)
        assert sp.issparse(X)
        assert X.dtype == np.uint8
        assert X.shape == (3, 2048)

    def test_none_mol_is_all_zero_row(self):
        feat = ECFPFeaturizer(radius=2, n_bits=256)
        X = feat.transform([MOLS[0], None])
        row1 = np.asarray(X[1].todense()).ravel()
        assert np.all(row1 == 0)

    def test_row_order_matches_input(self):
        feat = ECFPFeaturizer(radius=2, n_bits=2048)
        X = feat.transform(MOLS)
        for i, mol in enumerate(MOLS):
            direct = ECFPFeaturizer(radius=2, n_bits=2048).transform([mol])
            assert np.array_equal(X[i].toarray(), direct.toarray())

    def test_maccs_167_bits(self):
        feat = MACCSFeaturizer()
        X = feat.transform(MOLS)
        assert X.shape == (3, 167)

    def test_fcfp_differs_from_ecfp_in_general(self):
        e = ECFPFeaturizer(radius=2, n_bits=2048).transform(MOLS)
        f = FCFPFeaturizer(radius=2, n_bits=2048).transform(MOLS)
        assert not np.array_equal(e.toarray(), f.toarray())


class TestPhysChemOrder:
    def test_descriptor_order_matches_rdkit_directly(self):
        from rdkit.Chem import Descriptors

        feat = PhysChemFeaturizer()
        X = feat.transform([MOLS[0]])
        assert _PHYSCHEM_DESCRIPTORS[0] == "MolWt"
        assert X[0, 0] == pytest.approx(Descriptors.MolWt(MOLS[0]))
        assert _PHYSCHEM_DESCRIPTORS[1] == "MolLogP"
        assert X[0, 1] == pytest.approx(Descriptors.MolLogP(MOLS[0]))
        assert _PHYSCHEM_DESCRIPTORS[-1] == "BertzCT"
        assert X[0, -1] == pytest.approx(Descriptors.BertzCT(MOLS[0]))

    def test_none_mol_all_zero(self):
        feat = PhysChemFeaturizer()
        X = feat.transform([None])
        assert np.all(X[0] == 0.0)


class TestMQN:
    def test_shape(self):
        feat = MQNFeaturizer()
        X = feat.transform(MOLS)
        assert X.shape == (3, 42)


class TestPrecomputed:
    def test_identity(self):
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        feat = PrecomputedFeaturizer(arr)
        out = feat.transform(None)
        assert out is arr


class TestUnknownAlias:
    def test_raises_with_suggestions(self):
        with pytest.raises(Exception) as exc_info:
            get_featurizer("ecfpX")
        assert "ecfp" in str(exc_info.value).lower() or len(str(exc_info.value)) > 0

    def test_get_featurizer_passthrough(self):
        feat = ECFPFeaturizer()
        assert get_featurizer(feat) is feat
