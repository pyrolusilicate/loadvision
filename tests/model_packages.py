"""Valid model package manifests and folders for tests; override only what a test is about."""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loadvision.ml.packaging.manifest import Manifest
from loadvision.ml.packaging.model_package import ModelPackage, hash_package_files, write_package

HEAD_BYTES = b"ridge head"


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
