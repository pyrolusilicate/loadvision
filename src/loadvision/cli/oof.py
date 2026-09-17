"""What every out-of-fold command starts from: the labelled split, the folds and an ensemble."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from loadvision.cli.dataset import train_samples
from loadvision.cli.errors import CommandError
from loadvision.cli.paths import embeddings_cache
from loadvision.ml.data.manifest import Sample
from loadvision.ml.data.splits import Folds, read_folds
from loadvision.ml.evaluation.ensemble import Member, ensemble_out_of_fold
from loadvision.ml.pipeline.stages import DEFAULT_FOLDS

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Dataset:
    """The labelled split with the canonical folds every comparison rests on.

    Attributes:
        samples: Labelled photos, sorted by image id.
        labels: Reference loads, in the same order.
        numbers: Fold number of every photo.
        folds: The folds file itself, for the record rows.
    """

    samples: list[Sample]
    labels: NDArray[np.float64]
    numbers: list[int]
    folds: Folds


def add_member_option(parser: argparse.ArgumentParser, default: tuple[str, ...]) -> None:
    """Adds `--member`, an ensemble member written as ``encoder:alpha``, repeatable."""
    parser.add_argument(
        "--member",
        action="append",
        help=f"ensemble member as encoder:alpha, repeatable (default: {' '.join(default)})",
    )


def members_of(args: argparse.Namespace, default: tuple[str, ...]) -> list[Member]:
    """Reads the members from the command line, or the default ensemble.

    Raises:
        CommandError: If a member is not written as `encoder:alpha`.
    """
    try:
        return [Member.parse(text) for text in args.member or default]
    except ValueError as error:
        raise CommandError(str(error)) from error


def dataset_of(data_dir: Path) -> Dataset:
    """Reads the labelled split and the canonical folds.

    Raises:
        CommandError: If the dataset is broken.
        FileNotFoundError: If a dataset file or the folds file is missing.
    """
    samples = train_samples(data_dir)
    folds = read_folds(DEFAULT_FOLDS)
    return Dataset(
        samples=samples,
        labels=np.array([sample.label for sample in samples], dtype=np.float64),
        numbers=folds.fold_indices(samples),
        folds=folds,
    )


def ensemble_predictions(
    data: Dataset, members: list[Member], *, device: str, runs: Path
) -> NDArray[np.float64]:
    """Fits every member out of fold, logs its error and returns the ensemble predictions.

    Raises:
        KeyError: If an encoder is not in the registry.
    """
    result = ensemble_out_of_fold(
        data.samples,
        data.labels,
        data.numbers,
        members,
        device=device,
        cache_dir=embeddings_cache(runs),
    )
    for part in result.members:
        _logger.info("%s: oof mae %.3f", part.member, part.mae)
    return result.predictions
