"""What every out-of-fold command starts from: the labelled split, the folds and an ensemble."""

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from loadvision.cli.dataset import train_samples
from loadvision.cli.errors import CommandError
from loadvision.cli.paths import embeddings_cache
from loadvision.imaging.quality import aspect
from loadvision.ml.data.manifest import Sample
from loadvision.ml.data.splits import Folds, read_folds
from loadvision.ml.evaluation.ensemble import Member, ensemble_out_of_fold
from loadvision.ml.pipeline.stages import DEFAULT_FOLDS
from loadvision.ml.postprocess import ASPECT, Facts
from loadvision.ml.training.embedding_cache import read_image

_logger = logging.getLogger(__name__)


def photo_facts(samples: Sequence[Sample]) -> Facts:
    """Reads what conditional postprocessing steps look at, in the order of `samples`.

    The pipeline takes these from the `manifest.csv` of its run; a sweep has no run, so it
    decodes the photos itself. Both go through the same decoder and the same definition of
    proportions, so a chain measured here behaves the same way inside the package.

    Args:
        samples: Photos, in prediction order.

    Returns:
        Fact name to one number per photo.

    Raises:
        LoadvisionError: If a photo cannot be decoded.
    """
    return {ASPECT: np.array([aspect(read_image(s.path)) for s in samples], dtype=np.float64)}


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
