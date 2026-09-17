"""Valid model package manifests and folders for tests; override only what a test is about."""

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from loadvision.ml.packaging.baseline_package import build_baseline_package
from loadvision.ml.packaging.manifest import Manifest
from loadvision.ml.packaging.model_package import ModelPackage, hash_package_files, write_package

HEAD_BYTES = b"ridge head"
BASELINE_FEATURES = 233
_BUILT_AT = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def build_numpy_package(
    root: Path, *, load: float, version: str = "baseline-v0", features: int = BASELINE_FEATURES
) -> Path:
    """Writes a package of baseline features whose head predicts `load` for every photo.

    Args:
        root: Folder to work in; the package lands in `root/package`.
        load: What the head predicts.
        version: Package version.
        features: Size the head expects; another number makes a package the service refuses.

    Returns:
        The package folder.
    """
    baseline = root / "baseline"
    baseline.mkdir(parents=True)
    np.savez(
        baseline / "model.npz",
        mean=np.zeros(features),
        scale=np.ones(features),
        weights=np.zeros(features),
        intercept=np.asarray(load),
    )
    package = root / "package"
    build_baseline_package(
        baseline,
        package,
        version=version,
        git_commit="0" * 40,
        data_hash="d" * 64,
        created_at=_BUILT_AT,
    )
    return package


def member_data(**overrides: Any) -> dict[str, Any]:
    """Returns a valid member as JSON data: numpy features with a head file."""
    data: dict[str, Any] = {
        "backbone": "baseline_features",
        "runtime": "numpy",
        "head": "head.npz",
        "weight": 1.0,
        "preprocessing": {"name": "baseline_v0", "params": {}},
    }
    data.update(overrides)
    return data


def manifest_data(**overrides: Any) -> dict[str, Any]:
    """Returns a valid manifest as JSON data, shaped like package v0: one member, no CV."""
    data: dict[str, Any] = {
        "schema_version": 1,
        "version": "baseline-v0",
        "created_at": "2026-09-16T12:00:00Z",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "data_hash": "d" * 64,
        "members": [member_data()],
        "files": {"head.npz": hashlib.sha256(HEAD_BYTES).hexdigest()},
    }
    data.update(overrides)
    return data


def build_package(
    root: Path, files: Mapping[str, bytes] | None = None, **overrides: Any
) -> ModelPackage:
    """Writes files into `root` and a manifest whose `files` lists them, then loads the package.

    Args:
        root: Package folder; created if missing.
        files: File contents by package path; by default only the head of `member_data`.
        **overrides: Manifest fields to replace; `files` is always computed from the folder.
    """
    for name, content in ({"head.npz": HEAD_BYTES} if files is None else files).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    data = manifest_data(**overrides)
    data["files"] = hash_package_files(root)
    return write_package(root, Manifest.model_validate(data))
