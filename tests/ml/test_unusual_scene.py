"""A photo farther from the training photos than the package allows is marked for a check."""

import json
from pathlib import Path

import numpy as np
import pytest

from loadvision.application.ports import Prediction
from loadvision.domain.assessment import ReviewReason
from loadvision.ml.inference.ood import OodReference
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.inference.unusable_frames import frames_fingerprint
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


def predict(root: Path, threshold: float, cache: Path | None = None) -> Prediction:
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
    [prediction] = PackagePredictor(package, scale_cache=cache).predict([make_image()])
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


@pytest.mark.ml
def test_the_package_measures_where_empty_frames_land_on_its_own_scale(tmp_path: Path) -> None:
    """The two ends of the band travel with every prediction, so the caller needs no constant."""
    prediction = predict(tmp_path / "band", NEVER)

    assert prediction.ood_threshold == NEVER
    assert prediction.ood_unusable is not None
    # A grey wall, a dark frame, a blown-out frame and noise: none of them is a truck body, so
    # every one of them must sit further away than a photo of one.
    assert prediction.ood_unusable > 0
    assert prediction.ood_score is not None


@pytest.mark.ml
def test_the_measured_scale_is_kept_and_read_back(tmp_path: Path) -> None:
    """Measuring costs one prediction per frame; a restart must not pay for it again."""
    cache = tmp_path / "scale" / "v1.scale.json"

    first = predict(tmp_path / "keep", NEVER, cache=cache)

    stored = json.loads(cache.read_text(encoding="utf-8"))
    assert stored["unusable"] == first.ood_unusable
    assert stored["frames"] == frames_fingerprint()


@pytest.mark.ml
def test_a_scale_measured_with_other_frames_is_not_reused(tmp_path: Path) -> None:
    """The stored number is only worth keeping while the frames behind it are the same."""
    cache = tmp_path / "v1.scale.json"
    cache.write_text(
        json.dumps({"version": "v1", "frames": "not-our-frames", "unusable": 0.5}),
        encoding="utf-8",
    )

    prediction = predict(tmp_path / "stale", NEVER, cache=cache)

    assert prediction.ood_unusable != 0.5
    assert json.loads(cache.read_text(encoding="utf-8"))["frames"] == frames_fingerprint()


@pytest.mark.ml
def test_an_unreadable_cache_is_measured_again(tmp_path: Path) -> None:
    cache = tmp_path / "broken.scale.json"
    cache.write_text("{ this is not json", encoding="utf-8")

    assert predict(tmp_path / "broken", NEVER, cache=cache).ood_unusable is not None
