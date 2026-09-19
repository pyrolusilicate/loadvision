"""Averaging the out-of-fold runs of several models into one ensemble, photo for photo."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.pipeline.stages.train_folds import OofRow


@dataclass(frozen=True, slots=True, kw_only=True)
class Blend:
    """Averaged out-of-fold predictions of several runs.

    Attributes:
        ids: Photo ids, sorted, the order of every array below.
        folds: Fold every photo was held out in.
        labels: Reference loads.
        raw: Mean of the members' predictions before postprocessing.
    """

    ids: tuple[str, ...]
    folds: NDArray[np.int64]
    labels: NDArray[np.float64]
    raw: NDArray[np.float64]


def blend(runs: Mapping[str, Mapping[str, OofRow]]) -> Blend:
    """Averages the raw predictions of runs measured on the same photos and the same folds.

    Every member predicted each photo with a model that never saw it, so their mean did not
    see it either. The folds must match too: the calibration on top is fitted fold by fold,
    and a photo sitting in two folds would leak into its own curve.

    Args:
        runs: Member name to its rows of `oof.csv`.

    Returns:
        The averaged predictions.

    Raises:
        ValueError: If there are no runs, or two runs disagree about which photos they cover,
            the fold of a photo or its label.
    """
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
        raw=np.mean([[runs[name][i].raw for i in ids] for name in names], axis=0),
    )


def subsets(names: Sequence[str]) -> list[tuple[str, ...]]:
    """Every non-empty combination of members, the smallest first.

    Five members give 31 blends; each is scored out of fold in well under a second, so the
    whole search costs nothing next to a single fold of training.
    """
    return [group for size in range(1, len(names) + 1) for group in combinations(names, size)]
