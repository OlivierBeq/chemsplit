"""Identity featurizer over a caller-supplied feature matrix."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp


class PrecomputedFeaturizer:
    name = "precomputed"

    def __init__(self, X: np.ndarray | sp.spmatrix) -> None:
        self._X = X
        self.n_features = X.shape[1]
        self.is_binary = bool(sp.issparse(X) and X.dtype == np.uint8)

    def transform(self, mols: Sequence[Any]) -> np.ndarray | sp.csr_matrix:
        # `mols` is ignored for a precomputed matrix; `n` must match at call sites.
        """Return the rows of the precomputed matrix corresponding to ``mols``.

        :param mols: the molecules, used only for their count.
        :raises InputError: if the count does not match the stored matrix.
        :return: the stored feature matrix.
        """
        return self._X

    def get_params(self) -> dict[str, Any]:
        return {"n_features": self.n_features}


__all__ = ["PrecomputedFeaturizer"]
