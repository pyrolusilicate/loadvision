"""CLI commands for model packages."""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from loadvision.cli.errors import CommandError
from loadvision.cli.provenance import dataset_hash, git_commit
from loadvision.domain.model_version import ModelVersionId
from loadvision.ml.packaging.baseline_package import build_baseline_package
from loadvision.settings import load_settings

DEFAULT_BASELINE_DIR: Final = Path("baseline")
DEFAULT_DATA_DIR: Final = Path("data")

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `model` group and its commands.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser("model", help="model packages")
    commands = parser.add_subparsers(title="commands", required=True)

    build = commands.add_parser(
        "build-baseline",
        help="package the organizers' baseline for the service",
        description=(
            "Packages the organizers' baseline into the models folder from the settings, so the "
            "service has a model to answer with."
        ),
    )
    build.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="folder with the organizers' model.npz (default: %(default)s)",
    )
    build.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="dataset root with train/ and test/, recorded by hash (default: %(default)s)",
    )
    build.add_argument("--version", help="package version (default: model.version from settings)")
    build.set_defaults(run=build_baseline)


def build_baseline(args: argparse.Namespace) -> int:
    """Builds the baseline package into `<storage.models_dir>/<version>`.

    Everything is checked before the folder is created, so a failure leaves no package folder.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the version tag is invalid, the package folder exists, the git commit
            cannot be read or the dataset is inconsistent.
        FileNotFoundError: If the baseline weights or a dataset file are missing.
    """
    settings = load_settings()
    tag = settings.model.version if args.version is None else args.version
    try:
        version = ModelVersionId(tag)
    except ValueError as error:
        raise CommandError(f"invalid version {tag!r}: {error}") from error
    out_dir = settings.storage.models_dir / version.value
    # A version is never rebuilt in place: the database and past assessments refer to it.
    if out_dir.exists():
        raise CommandError(f"{out_dir} already exists; pass another --version")
    commit = git_commit()
    data_hash = dataset_hash(args.data)
    package = build_baseline_package(
        args.baseline_dir,
        out_dir,
        version=version.value,
        git_commit=commit,
        data_hash=data_hash,
        created_at=datetime.now(UTC),
    )
    _logger.info("built model package %s in %s", package.version.value, package.root)
    return 0
