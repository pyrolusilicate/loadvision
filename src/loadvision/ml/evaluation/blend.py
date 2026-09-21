"""Joining the out-of-fold runs of several models into one ensemble, photo for photo.

The members are joined by their mean or by their median. The metric is MAE, which is
minimised by the median and not by the mean, and on our runs the median wins by 0.10 to
0.14 with the direction the same in every combination of members. The mean stays the
default because every number measured before 21.09 was measured with it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.pipeline.stages.train_folds import OofRow

type Joint = Literal["mean", "median"]

JOINTS: Final[tuple[Joint, ...]] = ("mean", "median")


@dataclass(frozen=True, slots=True, kw_only=True)
class Blend:
    """Averaged out-of-fold predictions of several runs.

    Attributes:
        ids: Photo ids, sorted, the order of every array below.
        folds: Fold every photo was held out in.
        labels: Reference loads.
        raw: The members' predictions joined, before postprocessing.
        how: How they were joined.
    """

    ids: tuple[str, ...]
    folds: NDArray[np.int64]
    labels: NDArray[np.float64]
    raw: NDArray[np.float64]
    how: Joint = "mean"


def blend(runs: Mapping[str, Mapping[str, OofRow]], how: Joint = "mean") -> Blend:
    """Joins the raw predictions of runs measured on the same photos and the same folds.

    Every member predicted each photo with a model that never saw it, so the joined answer did
    not see it either. The folds must match too: the calibration on top is fitted fold by
    fold, and a photo sitting in two folds would leak into its own curve.

    Args:
        runs: Member name to its rows of `oof.csv`.
        how: Join the members by their `mean` or by their `median`.

    Returns:
        The joined predictions.

    Raises:
        ValueError: If there are no runs, the join is unknown, or two runs disagree about
            which photos they cover, the fold of a photo or its label.
    """
    if how not in JOINTS:
        raise ValueError(f"unknown join {how!r}, known: {', '.join(JOINTS)}")
    if not runs:
        raise ValueError("a blend needs at least one run")
    names = list(runs)
    first = runs[names[0]]
    ids = sorted(first)
    for name in names[1:]:
        other = runs[name]
        if set(other) != set(first):
            raise ValueError(f"{name} covers other photos than {names[0]}")
        moved = next((i for i in ids if other[i].fold != first[i].fold), None)
        if moved is not None:
            raise ValueError(f"{name} holds {moved} out in another fold than {names[0]}")
        relabelled = next((i for i in ids if other[i].label != first[i].label), None)
        if relabelled is not None:
            raise ValueError(f"{name} gives {relabelled} another label than {names[0]}")
    return Blend(
        ids=tuple(ids),
        folds=np.array([first[i].fold for i in ids], dtype=np.int64),
        labels=np.array([first[i].label for i in ids], dtype=np.float64),
        raw=join(np.array([[runs[name][i].raw for i in ids] for name in names]), how),
        how=how,
    )


def join(members: NDArray[np.float64], how: Joint = "mean") -> NDArray[np.float64]:
    """Joins one row per member into one row, by mean or by median."""
    return np.median(members, axis=0) if how == "median" else np.mean(members, axis=0)


def subsets(names: Sequence[str]) -> list[tuple[str, ...]]:
    """Every non-empty combination of members, the smallest first.

    Five members give 31 blends; each is scored out of fold in well under a second, so the
    whole search costs nothing next to a single fold of training.
    """
    return [group for size in range(1, len(names) + 1) for group in combinations(names, size)]
