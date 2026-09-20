"""A photo farther from the training photos than the package allows is marked for a check."""

from pathlib import Path

import numpy as np
import pytest

from loadvision.application.ports import Prediction
from loadvision.domain.assessment import ReviewReason
from loadvision.ml.inference.ood import OodReference
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.models import backbones
from tests.builders import make_image
from tests.model_packages import build_package, member_data
from tests.tiny_encoder import BACKBONE, WEIGHTS, use_tiny_encoder, write_checkpoint

HEAD = "head.npz"
REFERENCE = "ood/reference.npz"
TRAINING_PHOTOS = 4
# The score is a cosine distance clipped to 2, so no photo is ever above this threshold.
NEVER = 2.0


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    use_tiny_encoder(monkeypatch)


def predict(root: Path, threshold: float) -> Prediction:
    """Predicts one photo with a package of the tiny encoder and the given OOD threshold."""
    write_checkpoint(root / WEIGHTS)
    probe = backbones.load_backbone(BACKBONE, "cpu", root / WEIGHTS)
    np.savez(
        root / HEAD,
        mean=np.zeros(probe.dim),
        scale=np.ones(probe.dim),
        weights=np.zeros(probe.dim),
        intercept=np.asarray(42.0),
    )
    OodReference.build(BACKBONE, np.eye(TRAINING_PHOTOS, probe.dim)).save(root / REFERENCE)
    member = member_data(
        backbone=BACKBONE,
        runtime="torch",
        weights=WEIGHTS,
        preprocessing={"name": probe.preprocessing, "params": dict(probe.params)},
    )
    package = build_package(
        root,
        {name: (root / name).read_bytes() for name in (HEAD, WEIGHTS, REFERENCE)},
        members=[member],
        ood={"member": 0, "k": 1, "threshold": threshold, "reference": REFERENCE},
    )
    [prediction] = PackagePredictor(package).predict([make_image()])
    return prediction


def measured_score(tmp_path: Path) -> float:
    """The OOD score of the photo, from a package that never marks it."""
    prediction = predict(tmp_path / "measure", NEVER)
    assert prediction.review_reasons == frozenset()
    assert prediction.ood_score is not None
    assert prediction.ood_score > 0
    return prediction.ood_score


@pytest.mark.ml
def test_a_photo_above_the_threshold_is_an_unusual_scene(tmp_path: Path) -> None:
    score = measured_score(tmp_path)

    prediction = predict(tmp_path / "package", score / 2)

    assert prediction.review_reasons == frozenset({ReviewReason.UNUSUAL_SCENE})
    assert prediction.ood_score == score
    assert prediction.load_pct.value == 42.0


@pytest.mark.ml
def test_a_photo_right_at_the_threshold_is_not_marked(tmp_path: Path) -> None:
    score = measured_score(tmp_path)

    # The threshold is where the training photos end, and they are ordinary by definition.
    prediction = predict(tmp_path / "package", score)

    assert prediction.review_reasons == frozenset()
