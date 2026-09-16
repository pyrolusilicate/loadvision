from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from loadvision.ml.packaging.baseline_package import (
    BASELINE_HEAD_FILE,
    build_baseline_package,
)
from loadvision.ml.packaging.model_package import ModelPackage

T0 = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
DATA_HASH = "d" * 64


def write_baseline(baseline_dir: Path, features: int = 233) -> Path:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    weights = baseline_dir / "model.npz"
    np.savez(
        weights,
        mean=np.zeros(features),
        scale=np.ones(features),
        weights=np.zeros(features),
        intercept=np.asarray(55.0),
    )
    return weights


def build(tmp_path: Path) -> ModelPackage:
    write_baseline(tmp_path / "baseline")
    return build_baseline_package(
        tmp_path / "baseline",
        tmp_path / "package",
        version="baseline-v0",
        git_commit=COMMIT,
        data_hash=DATA_HASH,
        created_at=T0,
    )


@pytest.mark.ml
def test_package_describes_the_baseline_as_one_numpy_member(tmp_path: Path) -> None:
    package = build(tmp_path)
    [member] = package.manifest.members
    assert (member.runtime.value, member.head, member.weight) == ("numpy", BASELINE_HEAD_FILE, 1.0)
    assert member.preprocessing.name == "baseline_v0"


@pytest.mark.ml
def test_package_has_no_cv_metrics_because_it_was_not_evaluated_on_our_folds(
    tmp_path: Path,
) -> None:
    package = build(tmp_path)
    assert (package.manifest.cv, package.manifest.folds) == (None, None)


@pytest.mark.ml
def test_written_package_loads_back_with_the_head_in_place(tmp_path: Path) -> None:
    build(tmp_path)
    package = ModelPackage.load(tmp_path / "package")
    assert package.path(BASELINE_HEAD_FILE).is_file()


@pytest.mark.ml
def test_building_over_an_existing_folder_is_refused(tmp_path: Path) -> None:
    build(tmp_path)
    with pytest.raises(FileExistsError):
        build_baseline_package(
            tmp_path / "baseline",
            tmp_path / "package",
            version="baseline-v1",
            git_commit=COMMIT,
            data_hash=DATA_HASH,
            created_at=T0,
        )


@pytest.mark.ml
def test_missing_baseline_weights_are_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"model\.npz"):
        build_baseline_package(
            tmp_path / "empty",
            tmp_path / "package",
            version="baseline-v0",
            git_commit=COMMIT,
            data_hash=DATA_HASH,
            created_at=T0,
        )
