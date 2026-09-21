"""A photo farther from the training photos than the package allows is marked for a check."""

import json
from pathlib import Path

import numpy as np
import pytest

from loadvision.application.ports import Prediction
from loadvision.domain.assessment import ReviewReason
from loadvision.imaging.prepared_image import PreparedImage
from loadvision.ml.inference.ood import OodReference
from loadvision.ml.inference.predictor import PackagePredictor, _build_id
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
    predictor, image = built(root, threshold, cache)
    [prediction] = predictor.predict([image])
    return prediction


def built(
    root: Path, threshold: float, cache: Path | None = None
) -> tuple[PackagePredictor, PreparedImage]:
    """Builds a package of the tiny encoder and returns its predictor with one photo to try."""
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
    return PackagePredictor(package, scale_cache=cache), make_image()


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
def test_the_score_alone_is_the_score_of_a_whole_prediction(tmp_path: Path) -> None:
    """Tools walk folders of photos with the short path; it must not answer differently."""
    predictor, image = built(tmp_path / "alone", NEVER)

    [prediction] = predictor.predict([image])

    assert prediction.ood_score is not None
    assert predictor.ood_score(image) == prediction.ood_score


@pytest.mark.ml
def test_the_band_is_the_two_ends_the_package_measures(tmp_path: Path) -> None:
    predictor, image = built(tmp_path / "band-ends", NEVER)

    [prediction] = predictor.predict([image])

    assert predictor.band == (prediction.ood_threshold, prediction.ood_unusable)


@pytest.mark.ml
def test_the_measured_scale_is_kept(tmp_path: Path) -> None:
    """Measuring costs one prediction per frame; a restart must not pay for it again."""
    cache = tmp_path / "scale" / "v1.scale.json"

    first = predict(tmp_path / "keep", NEVER, cache=cache)

    stored = json.loads(cache.read_text(encoding="utf-8"))
    assert stored["unusable"] == first.ood_unusable
    assert stored["frames"] == frames_fingerprint()
    assert stored["build"] == _build_id("cpu")


@pytest.mark.ml
def test_a_kept_scale_is_read_back(tmp_path: Path) -> None:
    """The number the service uses comes from the file, not from a second measurement."""
    cache = tmp_path / "v1.scale.json"
    predict(tmp_path / "keep", NEVER, cache=cache)
    stored = json.loads(cache.read_text(encoding="utf-8"))
    # A number no measurement produces: the frames land where they land, never a metre further.
    stored["unusable"] += 1.0
    cache.write_text(json.dumps(stored), encoding="utf-8")

    assert predict(tmp_path / "again", NEVER, cache=cache).ood_unusable == stored["unusable"]


@pytest.mark.ml
def test_a_scale_measured_with_other_frames_is_not_reused(tmp_path: Path) -> None:
    """The stored number is only worth keeping while the frames behind it are the same."""
    cache = tmp_path / "v1.scale.json"
    cache.write_text(
        json.dumps(
            {
                "version": "v1",
                "frames": "not-our-frames",
                "build": _build_id("cpu"),
                "unusable": 0.5,
            }
        ),
        encoding="utf-8",
    )

    prediction = predict(tmp_path / "stale", NEVER, cache=cache)

    assert prediction.ood_unusable != 0.5
    assert json.loads(cache.read_text(encoding="utf-8"))["frames"] == frames_fingerprint()


@pytest.mark.ml
def test_a_scale_measured_on_another_build_is_not_reused(tmp_path: Path) -> None:
    """The same frames through the same weights land elsewhere on another torch build."""
    cache = tmp_path / "v1.scale.json"
    cache.write_text(
        json.dumps(
            {
                "version": "v1",
                "frames": frames_fingerprint(),
                "build": "cuda torch-2.4.0+cu121 Linux-x86_64",
                "unusable": 0.5,
            }
        ),
        encoding="utf-8",
    )

    prediction = predict(tmp_path / "elsewhere", NEVER, cache=cache)

    assert prediction.ood_unusable != 0.5
    assert json.loads(cache.read_text(encoding="utf-8"))["build"] == _build_id("cpu")


@pytest.mark.ml
def test_the_build_tells_the_devices_apart() -> None:
    """A package measured on the processor says nothing about the same package on the card."""
    assert _build_id("cpu") != _build_id("cuda")


@pytest.mark.ml
def test_an_unreadable_cache_is_measured_again(tmp_path: Path) -> None:
    cache = tmp_path / "broken.scale.json"
    cache.write_text("{ this is not json", encoding="utf-8")

    assert predict(tmp_path / "broken", NEVER, cache=cache).ood_unusable is not None
