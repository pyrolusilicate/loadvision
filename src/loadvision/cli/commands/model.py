"""CLI commands for model packages."""

import argparse
import logging
from pathlib import Path
from typing import Final

from loadvision.application.model_versions import ActivateModel, RegisterModel, RollbackModel
from loadvision.cli.database import unit_of_work_factory
from loadvision.cli.errors import CommandError
from loadvision.cli.options import add_data_option
from loadvision.cli.provenance import dataset_hash, git_commit
from loadvision.cli.values import short_utc, version_of
from loadvision.domain.timestamps import utc_now
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.baseline_package import build_baseline_package
from loadvision.ml.packaging.model_package import ModelPackage
from loadvision.ml.packaging.model_store import place_package
from loadvision.settings import load_settings

DEFAULT_BASELINE_DIR: Final = Path("baseline")

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
    add_data_option(build)
    build.add_argument("--version", help="package version (default: model.version from settings)")
    build.set_defaults(run=build_baseline)

    register = commands.add_parser(
        "register",
        help="put a package into the models folder and the database",
        description=(
            "Checks the package, copies it into the models folder and registers its version as "
            "a candidate. Registering does not change what the service answers with: that is "
            "what `activate` is for."
        ),
    )
    register.add_argument("package", type=Path, help="package folder, such as a run's package/")
    register.add_argument(
        "--skip-load-check",
        action="store_true",
        help="do not build the predictor, for a machine without the ML dependencies",
    )
    register.set_defaults(run=register_package)

    activate = commands.add_parser(
        "activate",
        help="make a registered version the one the service answers with",
        description=(
            "The previously active version retires, so `rollback` can bring it back. The "
            "service picks the new version up when it starts."
        ),
    )
    activate.add_argument("version", help="registered version")
    activate.set_defaults(run=activate_version)

    rollback = commands.add_parser(
        "rollback",
        help="bring back the version that answered before the current one",
        description="Activates the version that retired most recently.",
    )
    rollback.set_defaults(run=rollback_version)

    listing = commands.add_parser(
        "list", help="registered versions, newest first", description="Versions in the database."
    )
    listing.set_defaults(run=list_versions)


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
    version = version_of(tag)
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
        created_at=utc_now(),
    )
    _logger.info("built model package %s in %s", package.version.value, package.root)
    return 0


def register_package(args: argparse.Namespace) -> int:
    """Registers a package: checks it, copies it into the models folder, writes the version.

    The files are in place before the row appears, so a registered version always has a
    package the service can load.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the models folder already holds this version or it is registered.
        ModelPackageError: If the package is missing, invalid or not supported.
        OSError: If the package cannot be copied.
        RuntimeError: If `DATABASE_URL` is not set.
    """
    package = ModelPackage.load(args.package)
    version = package.version
    if not args.skip_load_check:
        # Fails here rather than on the first request of a client after activation.
        PackagePredictor(package)
    try:
        placed = place_package(args.package, load_settings().storage.models_dir)
    except FileExistsError as error:
        raise CommandError(str(error)) from error
    target = placed.root
    with unit_of_work_factory() as uow_factory:
        try:
            registered = RegisterModel(uow_factory=uow_factory, clock=utc_now)(version)
        except ValueError as error:
            raise CommandError(str(error)) from error
    _logger.info("registered %s as %s in %s", registered.version.value, registered.status, target)
    _logger.info("activate it with: loadvision model activate %s", registered.version.value)
    return 0


def activate_version(args: argparse.Namespace) -> int:
    """Makes a registered version active and retires the previous one.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the version tag is invalid.
        LoadvisionError: `MODEL_VERSION_NOT_FOUND` if the version is not registered;
            `MODEL_VERSION_STATUS_CONFLICT` if it was rejected.
        RuntimeError: If `DATABASE_URL` is not set.
    """
    version = version_of(args.version)
    with unit_of_work_factory() as uow_factory:
        active = ActivateModel(uow_factory=uow_factory, clock=utc_now)(version)
    _logger.info("active version is %s", active.version.value)
    _logger.info("restart the service to load it")
    return 0


def rollback_version(_args: argparse.Namespace) -> int:
    """Activates the version that retired most recently.

    Returns:
        Exit code 0.

    Raises:
        LoadvisionError: `MODEL_VERSION_NOT_FOUND` if nothing has been retired yet;
            `MODEL_VERSION_STATUS_CONFLICT` if that version cannot be activated again.
        RuntimeError: If `DATABASE_URL` is not set.
    """
    with unit_of_work_factory() as uow_factory:
        activate = ActivateModel(uow_factory=uow_factory, clock=utc_now)
        active = RollbackModel(uow_factory=uow_factory, activate=activate)()
    _logger.info("rolled back to %s", active.version.value)
    _logger.info("restart the service to load it")
    return 0


def list_versions(_args: argparse.Namespace) -> int:
    """Prints the registered versions, newest first.

    Returns:
        Exit code 0.

    Raises:
        RuntimeError: If `DATABASE_URL` is not set.
    """
    with unit_of_work_factory() as uow_factory, uow_factory() as uow:
        versions = uow.model_versions.list_all()
    if not versions:
        _logger.info("no versions registered yet")
        return 0
    _logger.info("%-24s%-12s%-22s%s", "version", "status", "registered", "activated")
    for version in versions:
        _logger.info(
            "%-24s%-12s%-22s%s",
            version.version.value,
            version.status.value,
            short_utc(version.created_at),
            "-" if version.activated_at is None else short_utc(version.activated_at),
        )
    return 0
