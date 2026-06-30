"""Binary fingerprint featurizers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp


def _stack_rows(rows: list[np.ndarray]) -> sp.csr_matrix:
    return sp.csr_matrix(np.vstack(rows).astype(np.uint8))


class ECFPFeaturizer:
    is_binary = True

    def __init__(self, radius: int = 2, n_bits: int = 2048, chirality: bool = False) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self.chirality = chirality
        self.name = f"ecfp{2 * radius}"
        self.n_features = n_bits

    def transform(self, mols: Sequence[Any]) -> sp.csr_matrix:
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits,
            includeChirality=self.chirality,
            countSimulation=False,
        )
        rows = []
        for mol in mols:
            if mol is None:
                rows.append(np.zeros(self.n_bits, dtype=np.uint8))
            else:
                rows.append(gen.GetFingerprintAsNumPy(mol))
        return _stack_rows(rows)

    def get_params(self) -> dict[str, Any]:
        return {"radius": self.radius, "n_bits": self.n_bits, "chirality": self.chirality}


class FCFPFeaturizer:
    is_binary = True

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self.name = f"fcfp{2 * radius}"
        self.n_features = n_bits

    def transform(self, mols: Sequence[Any]) -> sp.csr_matrix:
        from rdkit.Chem import rdFingerprintGenerator

        inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits, atomInvariantsGenerator=inv_gen,
        )
        rows = []
        for mol in mols:
            if mol is None:
                rows.append(np.zeros(self.n_bits, dtype=np.uint8))
            else:
                rows.append(gen.GetFingerprintAsNumPy(mol))
        return _stack_rows(rows)

    def get_params(self) -> dict[str, Any]:
        return {"radius": self.radius, "n_bits": self.n_bits}


class MACCSFeaturizer:
    is_binary = True
    name = "maccs"
    n_features = 167

    def transform(self, mols: Sequence[Any]) -> sp.csr_matrix:
        from rdkit.Chem import rdMolDescriptors

        rows = []
        for mol in mols:
            row = np.zeros(167, dtype=np.uint8)
            if mol is not None:
                fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                on_bits = list(fp.GetOnBits())
                row[on_bits] = 1
            rows.append(row)
        return _stack_rows(rows)

    def get_params(self) -> dict[str, Any]:
        return {}


class RDKitFPFeaturizer:
    is_binary = True
    name = "rdkitfp"

    def __init__(self, min_path: int = 1, max_path: int = 7, n_bits: int = 2048) -> None:
        self.min_path = min_path
        self.max_path = max_path
        self.n_bits = n_bits
        self.n_features = n_bits

    def transform(self, mols: Sequence[Any]) -> sp.csr_matrix:
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=self.min_path, maxPath=self.max_path, fpSize=self.n_bits,
        )
        rows = []
        for mol in mols:
            if mol is None:
                rows.append(np.zeros(self.n_bits, dtype=np.uint8))
            else:
                rows.append(gen.GetFingerprintAsNumPy(mol))
        return _stack_rows(rows)

    def get_params(self) -> dict[str, Any]:
        return {"min_path": self.min_path, "max_path": self.max_path, "n_bits": self.n_bits}


class AvalonFeaturizer:
    is_binary = True
    name = "avalon"

    def __init__(self, n_bits: int = 1024) -> None:
        self.n_bits = n_bits
        self.n_features = n_bits

    def transform(self, mols: Sequence[Any]) -> sp.csr_matrix:
        from rdkit.Avalon import pyAvalonTools

        rows = []
        for mol in mols:
            row = np.zeros(self.n_bits, dtype=np.uint8)
            if mol is not None:
                fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.n_bits)
                row[list(fp.GetOnBits())] = 1
            rows.append(row)
        return _stack_rows(rows)

    def get_params(self) -> dict[str, Any]:
        return {"n_bits": self.n_bits}


class AtomPairFeaturizer:
    is_binary = True
    name = "atompair"

    def __init__(self, n_bits: int = 2048) -> None:
        self.n_bits = n_bits
        self.n_features = n_bits

    def transform(self, mols: Sequence[Any]) -> sp.csr_matrix:
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=self.n_bits, countSimulation=False)
        rows = []
        for mol in mols:
            if mol is None:
                rows.append(np.zeros(self.n_bits, dtype=np.uint8))
            else:
                rows.append(gen.GetFingerprintAsNumPy(mol))
        return _stack_rows(rows)

    def get_params(self) -> dict[str, Any]:
        return {"n_bits": self.n_bits}


class TopTorsionFeaturizer:
    is_binary = True
    name = "toptorsion"

    def __init__(self, n_bits: int = 2048) -> None:
        self.n_bits = n_bits
        self.n_features = n_bits

    def transform(self, mols: Sequence[Any]) -> sp.csr_matrix:
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=self.n_bits)
        rows = []
        for mol in mols:
            if mol is None:
                rows.append(np.zeros(self.n_bits, dtype=np.uint8))
            else:
                rows.append(gen.GetFingerprintAsNumPy(mol))
        return _stack_rows(rows)

    def get_params(self) -> dict[str, Any]:
        return {"n_bits": self.n_bits}


__all__ = [
    "AtomPairFeaturizer",
    "AvalonFeaturizer",
    "ECFPFeaturizer",
    "FCFPFeaturizer",
    "MACCSFeaturizer",
    "RDKitFPFeaturizer",
    "TopTorsionFeaturizer",
]
