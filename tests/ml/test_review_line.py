"""The line "send this photo to a person": picked by one rule and kept in the package."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.inference.review_line import Measured, line_for, record_line, write_line
from loadvision.ml.packaging.model_package import ModelPackage, ModelPackageError
from tests.builders import make_image
from tests.ml.test_unusual_scene import NEVER, built
from tests.model_packages import build_package
from tests.tiny_encoder import use_tiny_encoder


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    use_tiny_encoder(monkeypatch)


def folder_of(root: Path, colours: list[tuple[int, int, int]]) -> Path:
    """Writes one plain photo per colour into a new folder."""
    root.mkdir(parents=True)
    for index, colour in enumerate(colours):
        Image.new("RGB", (96, 128), colour).save(root / f"{index}.png")
    return root


def measured(real: list[float], foreign: list[float]) -> Measured:
    return Measured(real=np.asarray(real), foreign=np.asarray(foreign))


@pytest.mark.ml
def test_the_line_stands_just_under_the_nearest_foreign_frame() -> None:
    line = line_for(measured([-0.10, -0.05, 0.0, 0.01], [0.047, 0.30, 0.90]))

    assert line == pytest.approx(0.04)


@pytest.mark.ml
def test_the_line_retreats_when_catching_everything_costs_too_many_real_photos() -> None:
    """Every real photo past the nearest foreign frame: the budget wins and the frame slips."""
    real = np.linspace(0.0, 0.5, 100).tolist()

    line = line_for(measured(real, [0.10]), budget=0.01)

    assert line > 0.10


@pytest.mark.ml
def test_without_foreign_frames_the_line_keeps_a_margin_over_the_real_photos() -> None:
    line = line_for(measured([0.0] * 100, []))

    assert line == pytest.approx(0.03)


@pytest.mark.ml
def test_no_real_photo_leaves_nothing_to_price_the_line_by() -> None:
    with pytest.raises(ValueError, match="no real photo"):
        line_for(measured([], [0.5]))


@pytest.mark.ml
def test_a_picked_line_travels_with_the_package_and_reaches_every_prediction(
    tmp_path: Path,
) -> None:
    built(tmp_path / "package", NEVER)
    real = folder_of(tmp_path / "real", [(120, 110, 100), (130, 120, 110), (125, 118, 104)])
    foreign = folder_of(tmp_path / "foreign", [(255, 0, 0), (0, 0, 255)])

    line = record_line(tmp_path / "package", real, foreign)

    assert line is not None
    package = ModelPackage.load(tmp_path / "package")
    assert package.manifest.ood is not None
    assert package.manifest.ood.review_share == line
    [prediction] = PackagePredictor(package).predict([make_image()])
    assert prediction.ood_review_share == line


@pytest.mark.ml
def test_a_package_that_measures_no_distances_has_no_line_to_keep(tmp_path: Path) -> None:
    build_package(tmp_path / "plain")

    with pytest.raises(ModelPackageError, match="measures no distances"):
        write_line(tmp_path / "plain", 0.04)


@pytest.mark.ml
def test_a_line_outside_the_band_is_refused_by_the_manifest(tmp_path: Path) -> None:
    built(tmp_path / "package", NEVER)

    with pytest.raises(ValidationError):
        write_line(tmp_path / "package", 1.5)
