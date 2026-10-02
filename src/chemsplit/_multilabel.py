"""Iterative stratification of multi-label data into folds of given sizes.

Implements Sechidis et al.'s iterative stratification (``order=1``) and Szymański &
Kajdanowicz's second-order variant over label pairs (``order=2``).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from chemsplit.exceptions import LabelError

__all__ = ["iterative_stratification", "label_keys"]


def label_keys(Y: np.ndarray, order: int) -> list[tuple[tuple[int, ...], ...]]:
    """Compute each record's stratification keys.

    :param Y: a binary label matrix, with NaN for unlabelled.
    :param order: ``1`` keys on single positive labels, ``2`` on pairs of them, falling back to
        the single label for a record with only one.
    :return: one key list per record.
    """
    keys: list[tuple[tuple[int, ...], ...]] = []
    for row in Y:
        positive = [int(j) for j in np.flatnonzero(np.nan_to_num(row, nan=0.0) == 1.0)]
        if order == 1 or len(positive) < 2:
            keys.append(tuple((j,) for j in positive))
        else:
            keys.append(tuple(combinations(positive, 2)))
    return keys


def _check_binary(Y: np.ndarray, owner: str) -> np.ndarray:
    try:
        Yf = np.asarray(Y, dtype=np.float64)
    except (TypeError, ValueError):
        raise LabelError(
            f"{owner}: multi-label stratification needs a numeric 0/1 label matrix"
        ) from None
    if Yf.ndim != 2:
        raise LabelError(
            f"{owner}: multi-label stratification needs a 2-D label matrix, "
            f"got shape {Yf.shape}"
        )
    observed = Yf[~np.isnan(Yf)]
    if not np.all((observed == 0.0) | (observed == 1.0)):
        raise LabelError(f"{owner}: multi-label stratification needs binary labels (0, 1 or NaN)")
    return Yf


def iterative_stratification(
    Y: np.ndarray,
    fold_sizes: Sequence[int],
    rng: np.random.Generator,
    order: int = 1,
    owner: str = "iterative_stratification",
) -> np.ndarray:
    """Spread every label, or label pair, across folds in proportion to the fold sizes.

    Repeatedly takes the key with the fewest unassigned positive records, ties to the lowest
    key, and gives each of its records in index order to the fold with the largest remaining
    demand for it. Ties go to the fold with the most room, then to a draw from ``rng``. Full
    folds are skipped, so the realised sizes match the request exactly, and records with no
    positive label are placed last by remaining room.

    :param Y: a binary label matrix, with NaN for unlabelled.
    :param fold_sizes: the record count per fold. Must sum to ``len(Y)``.
    :param rng: generator used only to break remaining ties.
    :param order: ``1`` balances single labels, ``2`` balances label pairs.
    :param owner: the caller's name, for error messages.
    :raises LabelError: if ``Y`` is not a 2-D binary matrix.
    :raises ParameterError: if ``fold_sizes`` does not sum to the record count.
    :return: the fold index of each record.
    """
    Yf = _check_binary(Y, owner)
    n = Yf.shape[0]
    if sum(fold_sizes) != n:
        raise ValueError(f"fold sizes sum to {sum(fold_sizes)}, expected {n}")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    record_keys = label_keys(Yf, order)
    members: dict[tuple[int, ...], list[int]] = {}
    for i, ks in enumerate(record_keys):
        for key in ks:
            members.setdefault(key, []).append(i)
    room = np.asarray(fold_sizes, dtype=np.float64)
    shares = room / n
    demand = {key: len(rows) * shares for key, rows in members.items()}
    fold = np.full(n, -1, dtype=np.int64)
    remaining = {key: len(rows) for key, rows in members.items()}

    def place(i: int, score: np.ndarray) -> None:
        open_folds = np.flatnonzero(room > 0)
        best = open_folds[score[open_folds] >= score[open_folds].max()]
        if best.size > 1:
            best = best[room[best] >= room[best].max()]
        m = int(best[0]) if best.size == 1 else int(rng.choice(best))
        fold[i] = m
        room[m] -= 1
        for key in record_keys[i]:
            demand[key][m] -= 1
            remaining[key] -= 1

    while True:
        live = [key for key in sorted(remaining) if remaining[key] > 0]
        if not live:
            break
        key = min(live, key=lambda k: (remaining[k], k))
        for i in members[key]:
            if fold[i] < 0:
                place(i, demand[key])
    for i in np.flatnonzero(fold < 0):
        place(int(i), room.copy())
    return fold
