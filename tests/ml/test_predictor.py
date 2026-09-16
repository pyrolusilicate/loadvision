from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.ml.inference.predictor import PackagePredictor, RidgeHead
from loadvision.ml.packaging.model_package import ModelPackageError
from tests.model_packages import build_package, member_data

T0 = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
FEATURES = 233


def write_head(path: Path, *, intercept: float, weight: float = 0.0) -> bytes:
    """Writes a head that predicts `intercept` plus `weight` times the first feature."""
    weights = np.zeros(FEATURES)
    weights[0] = weight
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=np.zeros(FEATURES),
        scale=np.ones(FEATURES),
        weights=weights,
        intercept=np.asarray(intercept),
    )
    return path.read_bytes()


def make_image(value: int = 0) -> PreparedImage:
    pixels: NDArray[np.uint8] = np.full((8, 6, 3), value, dtype=np.uint8)
    return PreparedImage(b"photo", MediaType.JPEG, pixels)


def build_ensemble(root: Path, intercepts: list[float], weights: list[float], **overrides: object):
    files: dict[str, bytes] = {}
    members = []
    for index, (intercept, weight) in enumerate(zip(intercepts, weights, strict=True)):
        name = f"m{index}/head.npz"
        files[name] = write_head(root / name, intercept=intercept)
        members.append(member_data(head=name, weight=weight))
    return build_package(root, files=files, members=members, **overrides)


@pytest.mark.ml
def test_predictor_reports_the_package_version(tmp_path: Path) -> None:
    package = build_ensemble(tmp_path, [50.0], [1.0], version="baseline-v0")
    assert PackagePredictor(package).version == ModelVersionId("baseline-v0")


@pytest.mark.ml
def test_one_member_predicts_its_own_value(tmp_path: Path) -> None:
    predictor = PackagePredictor(build_ensemble(tmp_path, [42.0], [1.0]))
    [prediction] = predictor.predict([make_image()])
    assert prediction.load_pct.value == pytest.approx(42.0)
    assert prediction.uncertainty is None


@pytest.mark.ml
def test_ensemble_averages_members_by_weight_and_reports_their_spread(tmp_path: Path) -> None:
    predictor = PackagePredictor(build_ensemble(tmp_path, [40.0, 60.0], [0.25, 0.75]))
    [prediction] = predictor.predict([make_image()])
    assert prediction.load_pct.value == pytest.approx(55.0)
    assert prediction.uncertainty == pytest.approx(20.0)


@pytest.mark.ml
def test_predictions_keep_the_order_of_the_photos(tmp_path: Path) -> None:
    predictor = PackagePredictor(build_ensemble(tmp_path, [50.0], [1.0]))
    predictions = predictor.predict([make_image(0), make_image(255), make_image(128)])
    assert len(predictions) == 3


@pytest.mark.ml
@pytest.mark.parametrize("intercept", [-30.0, 130.0])
def test_predictions_stay_within_the_range(tmp_path: Path, intercept: float) -> None:
    predictor = PackagePredictor(build_ensemble(tmp_path, [intercept], [1.0]))
    [prediction] = predictor.predict([make_image()])
    assert 0.0 <= prediction.load_pct.value <= 100.0


@pytest.mark.ml
def test_postprocess_rounds_and_snaps(tmp_path: Path) -> None:
    package = build_ensemble(tmp_path, [97.0], [1.0], postprocess={"round_to": 5, "snap_100": 95})
    [prediction] = PackagePredictor(package).predict([make_image()])
    assert prediction.load_pct.value == pytest.approx(100.0)


@pytest.mark.ml
def test_an_unsupported_runtime_fails_when_the_predictor_is_built(tmp_path: Path) -> None:
    package = build_package(
        tmp_path,
        files={"head.npz": write_head(tmp_path / "head.npz", intercept=1.0), "b.onnx": b"onnx"},
        members=[member_data(runtime="onnx", weights="b.onnx")],
    )
    with pytest.raises(ModelPackageError, match="onnx"):
        PackagePredictor(package)


@pytest.mark.ml
def test_a_head_that_expects_other_features_is_reported(tmp_path: Path) -> None:
    head = tmp_path / "head.npz"
    head.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        head, mean=np.zeros(5), scale=np.ones(5), weights=np.zeros(5), intercept=np.asarray(1.0)
    )
    package = build_package(tmp_path, files={"head.npz": head.read_bytes()})
    with pytest.raises(ModelPackageError, match="233"):
        PackagePredictor(package).predict([make_image()])


@pytest.mark.ml
@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({"mean": np.zeros(3), "scale": np.ones(3), "weights": np.zeros(3)}, "intercept"),
        (
            {
                "mean": np.zeros(3),
                "scale": np.zeros(3),
                "weights": np.zeros(3),
                "intercept": np.asarray(1.0),
            },
            "scale must be positive",
        ),
        (
            {
                "mean": np.zeros(3),
                "scale": np.ones(4),
                "weights": np.zeros(3),
                "intercept": np.asarray(1.0),
            },
            "one shape",
        ),
    ],
)
def test_a_broken_head_file_is_reported(
    tmp_path: Path, arrays: dict[str, Any], message: str
) -> None:
    head = tmp_path / "head.npz"
    np.savez(head, **arrays)
    with pytest.raises(ModelPackageError, match=message):
        RidgeHead.load(head)
