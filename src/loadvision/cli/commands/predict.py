"""CLI commands for predictions and submission files."""

import argparse
import logging
import time
from pathlib import Path
from typing import Final

from loadvision.cli.dataset import split_samples
from loadvision.cli.errors import CommandError
from loadvision.cli.options import add_data_option, add_device_option
from loadvision.cli.paths import SUBMISSIONS_DIR, runs_dir
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage
from loadvision.ml.data.manifest import Sample
from loadvision.ml.data.submission import write_submission
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.model_package import ModelPackage
from loadvision.ml.training.embedding_cache import read_image
from loadvision.settings import load_settings

DEFAULT_BATCH: Final = 8
_REPORT_EVERY: Final = 50

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `predict` group and its commands.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser("predict", help="predictions of a model package")
    commands = parser.add_subparsers(title="commands", required=True)

    submission = commands.add_parser(
        "submission",
        help="predict a dataset split with a model package",
        description=(
            "Predicts every photo of a split with the package the service would answer with, "
            "and writes the answer file for the platform. The package decides everything: "
            "encoders, heads and postprocessing come from its manifest."
        ),
    )
    submission.add_argument(
        "--version", help="package version in the models folder (default: model.version)"
    )
    submission.add_argument(
        "--package", type=Path, help="package folder, instead of a version in the models folder"
    )
    submission.add_argument(
        "--split", choices=("test", "train"), default="test", help="split to predict"
    )
    submission.add_argument("--out", type=Path, help="answer file (default: <runs>/submissions/)")
    submission.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
        help="photos per model call (default: %(default)s)",
    )
    add_device_option(submission, default="cpu")
    add_data_option(submission)
    submission.set_defaults(run=predict_submission)


def predict_submission(args: argparse.Namespace) -> int:
    """Predicts a split with a model package and writes the answer file.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the version is invalid, the batch size is not positive or the dataset
            is broken.
        ModelPackageError: If the package is missing, invalid or not supported.
        LoadvisionError: If a photo cannot be decoded or the model fails on it.
    """
    if args.batch_size < 1:
        raise CommandError(f"batch size must be at least 1, got {args.batch_size}")
    package = ModelPackage.load(_package_dir(args))
    samples = split_samples(args.data, args.split)
    _logger.info(
        "package %s, %d photos of %s on %s",
        package.version.value,
        len(samples),
        args.split,
        args.device,
    )

    predictor = PackagePredictor(package, args.device)
    started = time.perf_counter()
    loads: list[float] = []
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        loads += [prediction.load_pct.value for prediction in predictor.predict(_photos(batch))]
        if len(loads) % _REPORT_EVERY < args.batch_size:
            _logger.info("predicted %d of %d", len(loads), len(samples))
    seconds = time.perf_counter() - started

    path = _out_path(args, package.version)
    write_submission(path, [sample.image_id for sample in samples], loads)
    _logger.info(
        "%d photos in %.1f s, %.0f ms per photo; min %.1f, mean %.1f, max %.1f",
        len(loads),
        seconds,
        1000 * seconds / len(loads),
        min(loads),
        sum(loads) / len(loads),
        max(loads),
    )
    _logger.info("written %s", path)
    return 0


def _package_dir(args: argparse.Namespace) -> Path:
    if args.package is not None:
        return args.package
    settings = load_settings()
    tag = settings.model.version if args.version is None else args.version
    try:
        version = ModelVersionId(tag)
    except ValueError as error:
        raise CommandError(f"invalid version {tag!r}: {error}") from error
    return settings.storage.models_dir / version.value


def _out_path(args: argparse.Namespace, version: ModelVersionId) -> Path:
    if args.out is not None:
        return args.out
    return runs_dir() / SUBMISSIONS_DIR / f"{version.value}-{args.split}.csv"


def _photos(batch: list[Sample]) -> list[PreparedImage]:
    return [read_image(sample.path) for sample in batch]
