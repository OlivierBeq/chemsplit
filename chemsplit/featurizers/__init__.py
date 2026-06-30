"""Featurizer protocol and alias resolution.

The ``ECFP_k`` <-> ``radius`` mapping is ``radius = k // 2`` — implementations MUST NOT
interpret "ecfp6" as radius=6.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from chemsplit.types import FeatureMatrix

try:
    from chemsplit.exceptions import UnknownFeaturizerError
except ImportError:  # TODO: remove once chemsplit.exceptions lands; parallel effort owns it.
    class UnknownFeaturizerError(ValueError):
        pass

__all__ = ["Featurizer", "get_featurizer"]


@runtime_checkable
class Featurizer(Protocol):
    name: str
    n_features: int
    is_binary: bool

    def transform(self, mols) -> FeatureMatrix:...

    def get_params(self) -> dict[str, Any]:...


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def get_featurizer(spec: "str | Featurizer", **kw: Any) -> Featurizer:
    """Resolve a string alias (case-insensitive) or pass through an existing Featurizer instance.

    Accepted aliases: ``ecfp2``/``ecfp4``/``ecfp6``/``ecfp8`` (radius = k // 2),
    ``morgan2``==``ecfp4``, ``morgan3``==``ecfp6``, ``fcfp2``/``fcfp4``/``fcfp6``/``fcfp8``
    (same radius mapping, feature-invariant Morgan), plus each featurizer's own ``name``:
    ``maccs``, ``rdkitfp``, ``avalon``, ``atompair``, ``toptorsion``, ``physchem``, ``mqn``,
    ``precomputed``.
    """
    if not isinstance(spec, str):
        return spec

    from chemsplit.featurizers.descriptors import MQNFeaturizer, PhysChemFeaturizer
    from chemsplit.featurizers.fingerprints import (
        AtomPairFeaturizer,
        AvalonFeaturizer,
        ECFPFeaturizer,
        FCFPFeaturizer,
        MACCSFeaturizer,
        RDKitFPFeaturizer,
        TopTorsionFeaturizer,
    )
    from chemsplit.featurizers.precomputed import PrecomputedFeaturizer

    key = spec.lower()

    ecfp_like = {
        "ecfp2": 1, "ecfp4": 2, "ecfp6": 3, "ecfp8": 4,
        "morgan2": 2, "morgan3": 3,
    }
    fcfp_like = {"fcfp2": 1, "fcfp4": 2, "fcfp6": 3, "fcfp8": 4}

    if key in ecfp_like:
        return ECFPFeaturizer(radius=ecfp_like[key], **kw)
    if key in fcfp_like:
        return FCFPFeaturizer(radius=fcfp_like[key], **kw)
    if key == "maccs":
        return MACCSFeaturizer(**kw)
    if key == "rdkitfp":
        return RDKitFPFeaturizer(**kw)
    if key == "avalon":
        return AvalonFeaturizer(**kw)
    if key == "atompair":
        return AtomPairFeaturizer(**kw)
    if key == "toptorsion":
        return TopTorsionFeaturizer(**kw)
    if key == "physchem":
        return PhysChemFeaturizer(**kw)
    if key == "mqn":
        return MQNFeaturizer(**kw)
    if key == "precomputed":
        return PrecomputedFeaturizer(**kw)

    known = list(ecfp_like) + list(fcfp_like) + [
        "maccs", "rdkitfp", "avalon", "atompair", "toptorsion", "physchem", "mqn", "precomputed",
    ]
    closest = sorted(known, key=lambda k: _levenshtein(key, k))[:3]
    raise UnknownFeaturizerError(
        f"unknown featurizer {spec!r}; closest known aliases: {closest}"
    )
