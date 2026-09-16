"""Builds the v0 model package: the organizers' baseline in our package format."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Final

from loadvision.ml.packaging.manifest import Manifest, Member, Preprocessing, Runtime
from loadvision.ml.packaging.model_package import ModelPackage, hash_package_files, write_package

BASELINE_HEAD_FILE: Final = "m1/head.npz"
BASELINE_BACKBONE: Final = "baseline_features"
BASELINE_PREPROCESSING: Final = "baseline_v0"

_BASELINE_WEIGHTS: Final = "model.npz"


def build_baseline_package(
    baseline_dir: Path,
    out_dir: Path,
    *,
    version: str,
    git_commit: str,
    data_hash: str,
    created_at: datetime,
) -> ModelPackage:
    """Writes a package that predicts exactly what the baseline predicts.

    The package exists so the service has a real model on day one: the loader, the ports and
    the API can be tested before our own model is trained. Its quality is the baseline's,
    about 19 points of MAE, and it is the floor every later version has to beat.

    Args:
        baseline_dir: Folder with the organizers' `model.npz`.
        out_dir: Folder to write the package into; it must not exist yet, so an earlier
            package is never overwritten silently.
        version: Package version in the `ModelVersionId` format.
        git_commit: Commit the package is built from, 40 hex characters.
        data_hash: SHA-256 of the dataset manifest the baseline was fitted on.
        created_at: Build time, timezone-aware.

    Returns:
        The written package, loaded back and verified.

    Raises:
        FileNotFoundError: If the baseline weights are missing.
        FileExistsError: If `out_dir` already exists.
        ValidationError: If the manifest fields do not pass validation.
        ModelPackageError: If the written package does not load.
    """
    weights = baseline_dir / _BASELINE_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"baseline weights not found: {weights}")
    head_path = out_dir / BASELINE_HEAD_FILE
    head_path.parent.mkdir(parents=True)
    shutil.copyfile(weights, head_path)

    manifest = Manifest(
        schema_version=1,
        version=version,
        created_at=created_at,
        git_commit=git_commit,
        data_hash=data_hash,
        members=(
            Member(
                backbone=BASELINE_BACKBONE,
                runtime=Runtime.NUMPY,
                head=BASELINE_HEAD_FILE,
                weight=1.0,
                preprocessing=Preprocessing(name=BASELINE_PREPROCESSING),
            ),
        ),
        files=hash_package_files(out_dir),
    )
    return write_package(out_dir, manifest)
