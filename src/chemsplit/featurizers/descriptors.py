"""Continuous descriptor featurizers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rdkit import Chem

#: Fixed 12-descriptor order -- do not reorder.
_PHYSCHEM_DESCRIPTORS = (
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms", "BertzCT",
)


class PhysChemFeaturizer:
    is_binary = False
    name = "physchem"
    n_features = len(_PHYSCHEM_DESCRIPTORS)

    def transform(self, mols: Sequence[Any]) -> np.ndarray:
        """Featurize a batch of molecules.

        :param mols: the molecules. A ``None`` entry becomes an all-zero row.
        :return: the descriptor matrix, one row per molecule.
        """
        from rdkit.Chem import Descriptors, rdMolDescriptors

        def num_heteroatoms(mol: Chem.rdchem.Mol) -> int:
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
                rows.append(
                    np.array([fns[d](mol) for d in _PHYSCHEM_DESCRIPTORS], dtype=np.float64)
                )
        return np.vstack(rows)

    def get_params(self) -> dict[str, Any]:
        return {}


class MQNFeaturizer:
    is_binary = False
    name = "mqn"
    n_features = 42

    def transform(self, mols: Sequence[Any]) -> np.ndarray:
        """Featurize a batch of molecules.

        :param mols: the molecules. A ``None`` entry becomes an all-zero row.
        :return: the descriptor matrix, one row per molecule.
        """
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


def functional_group_names() -> tuple[str, ...]:
    """List RDKit's ``fr_*`` functional-group counter names.

    :return: the names from :mod:`rdkit.Chem.Fragments`, sorted.
    """
    from rdkit.Chem import Fragments

    return tuple(
        sorted(
            n for n in dir(Fragments)
            if n.startswith("fr_") and callable(getattr(Fragments, n))
        )
    )


class FunctionalGroupFeaturizer:
    """Counts of every RDKit ``fr_*`` functional group (85 in current RDKit releases), in sorted
    name order. The column set follows the installed RDKit version."""

    is_binary = False
    name = "functional_groups"

    def __init__(self) -> None:
        self.feature_names = functional_group_names()
        self.n_features = len(self.feature_names)

    def transform(self, mols: Sequence[Any]) -> np.ndarray:
        """Featurize a batch of molecules.

        :param mols: the molecules. A ``None`` entry becomes an all-zero row.
        :return: the descriptor matrix, one row per molecule.
        """
        from rdkit.Chem import Fragments

        fns = [getattr(Fragments, name) for name in self.feature_names]
        rows = []
        for mol in mols:
            if mol is None:
                rows.append(np.zeros(self.n_features, dtype=np.float64))
            else:
                rows.append(np.array([fn(mol) for fn in fns], dtype=np.float64))
        return np.vstack(rows) if rows else np.zeros((0, self.n_features))

    def get_params(self) -> dict[str, Any]:
        return {}


__all__ = [
    "FunctionalGroupFeaturizer",
    "MQNFeaturizer",
    "PhysChemFeaturizer",
    "functional_group_names",
]
