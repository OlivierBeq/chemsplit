"""Shared type aliases used throughout chemsplit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import scipy.sparse
    from rdkit import Chem

IndexArray: TypeAlias = npt.NDArray[np.int64]
"""A 1-D, sorted-ascending, duplicate-free int64 array of record indices."""

FeatureMatrix: TypeAlias = "np.ndarray | scipy.sparse.csr_matrix"
"""Either a dense float64 ndarray or a sparse uint8 CSR matrix, shape (n, n_features)."""

MolLike: TypeAlias = "str | Chem.rdchem.Mol"
"""A SMILES string or an already-parsed RDKit molecule."""

SizeSpec: TypeAlias = "float | int | None"
"""Semantics (normative, applies to every ``*_size`` parameter across the library):

- ``float`` in ``(0.0, 1.0)`` — a fraction of ``n``.
- ``int`` >= 1 — an absolute record count.
- ``0.0``, ``0``, or ``None`` — that partition is empty.
- Any other value raises :class:`chemsplit.exceptions.ParameterError`.
"""

Partition: TypeAlias = Literal["train", "valid", "test", "discard"]

__all__ = [
    "FeatureMatrix",
    "IndexArray",
    "MolLike",
    "Partition",
    "SizeSpec",
]
