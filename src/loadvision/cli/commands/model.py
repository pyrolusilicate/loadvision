"""CLI commands for model packages."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Final

from loadvision.application.model_versions import ActivateModel, RegisterModel, RollbackModel
from loadvision.cli.database import unit_of_work_factory
from loadvision.cli.errors import CommandError
from loadvision.cli.options import add_data_option
from loadvision.cli.provenance import dataset_hash, git_commit
from loadvision.cli.values import version_of
from loadvision.domain.timestamps import format_utc_seconds, utc_now
from loadvision.ml.data.manifest import load_split
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.baseline_package import build_baseline_package
from loadvision.ml.packaging.checkpoints import fetch_checkpoints
from loadvision.ml.packaging.model_package import ModelPackage
from loadvision.ml.packaging.model_store import place_package
from loadvision.settings import load_settings

DEFAULT_BASELINE_DIR: Final = Path("baseline")

# Settings that change how a run goes but not what it learns: loader processes, and whether
# block activations were recomputed to fit into a smaller graphics card.
INCIDENTAL: Final = frozenset({"workers", "checkpointing"})

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

    finetuned = commands.add_parser(
        "build-finetuned",
        help="package a model fine-tuned by `finetune`",
        description=(
            "Packages a `finetune` run: the model of a --full run, or the five fold models of "
            "an out-of-fold run averaged in equal shares, the way its submission was made. "
            "Packages the model with the calibration and the metrics "
            "of the out-of-fold run of the same settings. The package is written into the full "
            "run's folder; `register` puts it into the models folder."
        ),
    )
    finetuned.add_argument("run_dir", type=Path, help="folder of the finetune run to package")
    finetuned.add_argument(
        "--oof-run",
        type=Path,
        help="out-of-fold run the calibration and the metrics come from; the run itself when "
        "it is the one being packaged",
    )
    add_data_option(finetuned)
    finetuned.add_argument("--version", required=True, help="package version, such as v2")
    finetuned.add_argument("--device", default="cpu", help="torch device (default: %(default)s)")
    finetuned.set_defaults(run=build_finetuned)

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

    fetch = commands.add_parser(
        "fetch",
        help="download the encoder weights a package lists but does not hold",
        description=(
            "Downloads every encoder checkpoint the manifest lists from Hugging Face and keeps it "
            "only if its hash matches the manifest. A package travels without its gigabyte of "
            "public weights; this makes it whole again."
        ),
    )
    fetch.add_argument("package", type=Path, help="package folder")
    fetch.set_defaults(run=fetch_weights)

    activate = commands.add_parser(
        "activate",
        help="make a registered version the one the service answers with",
        description=(
            "The previously active version retires, so `rollback` can bring it back. The "
            "service switches to the new version within model.check_seconds."
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


def build_finetuned(args: argparse.Namespace) -> int:
    """Builds the package of a fine-tuned run into `<run>/package`.

    A run on every photo is packaged as one member; an out-of-fold run as its fold models
    averaged in equal shares, which is the model its submission was made with.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the version tag is invalid, the package folder exists, the runs do
            not match each other or the dataset.
    """
    # Imported here: building needs torch, the other model commands do not.
    from loadvision.ml.packaging.finetuned_package import (
        build_finetuned_package,
        build_folds_package,
    )
    from loadvision.ml.pipeline.stages import DEFAULT_CHAIN
    from loadvision.ml.pipeline.stages.build_package import LICENSES_DIR
    from loadvision.ml.pipeline.stages.train_folds import OOF_FILE, read_oof

    version = version_of(args.version)
    out_dir = args.run_dir / "package"
    if out_dir.exists():
        raise CommandError(f"{out_dir} already exists")
    run = _run_settings(args.run_dir)
    oof_run_dir = args.oof_run or args.run_dir
    oof_run = _run_settings(oof_run_dir)
    if oof_run["full"]:
        raise CommandError(f"{oof_run_dir} is a --full run; the calibration needs folds")
    if _learned(run) != _learned(oof_run):
        raise CommandError("the two runs were fine-tuned with different settings")
    build = build_finetuned_package if run["full"] else build_folds_package
    try:
        package = build(
            args.run_dir,
            out_dir,
            encoder=run["settings"]["encoder"],
            train=load_split(args.data, "train"),
            oof=read_oof(oof_run_dir / OOF_FILE),
            oof_file=oof_run_dir / OOF_FILE,
            chain=[(name, dict(params)) for name, params in DEFAULT_CHAIN],
            folds_file=Path(oof_run["folds"]),
            licenses_dir=LICENSES_DIR,
            version=version.value,
            git_commit=git_commit(),
            data_hash=dataset_hash(args.data),
            created_at=utc_now(),
            device=args.device,
            depth=bool(run["settings"].get("depth")),
            far_view=bool(run["settings"].get("far_view")),
        )
    except ValueError as error:
        raise CommandError(str(error)) from error
    _logger.info(
        "built model package %s of %d member(s) in %s",
        package.version.value,
        len(package.manifest.members),
        package.root,
    )
    return 0


def _run_settings(folder: Path) -> dict[str, Any]:
    path = folder / "settings.json"
    if not path.is_file():
        raise CommandError(f"{path} is missing; {folder} is not a finetune run")
    content: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return content


def _learned(run: dict[str, Any]) -> dict[str, Any]:
    """Settings that change what a run learns, unlike how fast or on what it ran."""
    return {key: value for key, value in run["settings"].items() if key not in INCIDENTAL}


def fetch_weights(args: argparse.Namespace) -> int:
    """Downloads the checkpoints a package misses, then verifies the whole package.

    Returns:
        Exit code 0.

    Raises:
        ModelPackageError: If the manifest is invalid, a downloaded checkpoint has another hash
            than the manifest records, or the package still does not verify.
        OSError: If a file cannot be written.
    """
    fetched = fetch_checkpoints(args.package)
    for name in fetched:
        _logger.info("fetched %s", name)
    package = ModelPackage.load(args.package)
    _logger.info(
        "package %s is complete: every file matches its hash (%d downloaded)",
        package.version.value,
        len(fetched),
    )
    return 0


def register_package(args: argparse.Namespace) -> int:
    """Registers a package: checks it, copies it into the models folder, writes the version.

    The files are in place before the row appears, so a registered version always has a
    package the service can load.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the models folder already holds this version, it is registered, or
            `DATABASE_URL` is not set.
        ModelPackageError: If the package is missing, invalid or not supported.
        OSError: If the package cannot be copied.
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
        CommandError: If the version tag is invalid or `DATABASE_URL` is not set.
        LoadvisionError: `MODEL_VERSION_NOT_FOUND` if the version is not registered;
            `MODEL_VERSION_STATUS_CONFLICT` if it was rejected.
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
        CommandError: If `DATABASE_URL` is not set.
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
        CommandError: If `DATABASE_URL` is not set.
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
            format_utc_seconds(version.created_at),
            "-" if version.activated_at is None else format_utc_seconds(version.activated_at),
        )
    return 0
