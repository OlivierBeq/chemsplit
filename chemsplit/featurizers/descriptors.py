"""Continuous descriptor featurizers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

#: Fixed 12-descriptor order — do not reorder.
_PHYSCHEM_DESCRIPTORS = (
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms", "BertzCT",
)


class PhysChemFeaturizer:
    is_binary = False
    name = "physchem"
    n_features = len(_PHYSCHEM_DESCRIPTORS)

    def transform(self, mols: Sequence[Any]) -> np.ndarray:
        from rdkit.Chem import Descriptors, rdMolDescriptors

        def num_heteroatoms(mol):
            return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))

        fns = {
            "MolWt": Descriptors.MolWt,
            "MolLogP": Descriptors.MolLogP,
            "TPSA": Descriptors.TPSA,
            "NumHDonors": Descriptors.NumHDonors,
            "NumHAcceptors": Descriptors.NumHAcceptors,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
            "RingCount": rdMolDescriptors.CalcNumRings,
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
            "HeavyAtomCount": lambda m: m.GetNumHeavyAtoms(),
            "NumHeteroatoms": num_heteroatoms,
            "BertzCT": Descriptors.BertzCT,
        }

        rows = []
        for mol in mols:
            if mol is None:
                rows.append(np.zeros(self.n_features, dtype=np.float64))
            else:
                rows.append(np.array([fns[d](mol) for d in _PHYSCHEM_DESCRIPTORS], dtype=np.float64))
        return np.vstack(rows)

    def get_params(self) -> dict[str, Any]:
        return {}


class MQNFeaturizer:
    is_binary = False
    name = "mqn"
    n_features = 42

    def transform(self, mols: Sequence[Any]) -> np.ndarray:
        from rdkit.Chem import rdMolDescriptors

        rows = []
        for mol in mols:
            if mol is None:
                rows.append(np.zeros(42, dtype=np.float64))
            else:
                rows.append(np.array(rdMolDescriptors.MQNs_(mol), dtype=np.float64))
        return np.vstack(rows)

    def get_params(self) -> dict[str, Any]:
        return {}


__all__ = ["MQNFeaturizer", "PhysChemFeaturizer"]
