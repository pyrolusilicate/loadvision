from pathlib import Path

import numpy as np
import pytest

from loadvision.ml.inference.ood import OodReference


def training(photos: int = 200, width: int = 32) -> np.ndarray:
    """Embeddings of one cluster, as photos of one kind of scene would give."""
    rng = np.random.default_rng(0)
    return rng.normal(scale=0.2, size=(photos, width)) + np.array([5.0] + [0.0] * (width - 1))


@pytest.mark.ml
def test_a_training_photo_scores_zero() -> None:
    vectors = training()
    reference = OodReference.build("toy", vectors)

    assert reference.score(vectors[0], 1) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.ml
def test_an_unseen_scene_scores_higher_than_a_familiar_one() -> None:
    vectors = training()
    reference = OodReference.build("toy", vectors)

    familiar = reference.score(vectors[5], 10)
    unseen = reference.score(-vectors[5], 10)

    assert unseen > familiar


@pytest.mark.ml
def test_the_score_never_goes_below_zero() -> None:
    reference = OodReference.build("toy", training())

    assert reference.score(training()[0], 10) >= 0.0


@pytest.mark.ml
def test_round_trip_through_a_file(tmp_path: Path) -> None:
    vectors = training()
    reference = OodReference.build("toy", vectors)
    path = tmp_path / "reference.npz"

    reference.save(path)
    restored = OodReference.load(path, "toy")

    assert restored.backbone == "toy"
    assert restored.score(vectors[0], 10) == pytest.approx(reference.score(vectors[0], 10))


@pytest.mark.ml
def test_a_reference_built_with_another_encoder_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "reference.npz"
    OodReference.build("dinov3_vitb16", training()).save(path)

    # The manifest points at a member by index, so a reordered manifest could pair the
    # reference with the wrong encoder. The file says whose it is and the mismatch is caught.
    with pytest.raises(ValueError, match="built with 'dinov3_vitb16', paired with 'siglip2'"):
        OodReference.load(path, "siglip2")


@pytest.mark.ml
def test_an_embedding_of_another_width_is_refused() -> None:
    reference = OodReference.build("toy", training(width=32))

    with pytest.raises(ValueError, match="embedding is 16 wide, reference is 32"):
        reference.score(np.zeros(16), 10)


@pytest.mark.ml
def test_a_file_without_the_vectors_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "reference.npz"
    np.savez(path, something_else=np.zeros(3))

    with pytest.raises(ValueError, match="missing vectors"):
        OodReference.load(path, "toy")


@pytest.mark.ml
def test_an_empty_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one embedding"):
        OodReference.build("toy", np.zeros((0, 8)))


@pytest.mark.ml
def test_neighbours_must_be_positive() -> None:
    reference = OodReference.build("toy", training())

    with pytest.raises(ValueError, match="must be positive"):
        reference.score(training()[0], 0)


@pytest.mark.ml
def test_asking_for_more_neighbours_than_there_are_still_works() -> None:
    vectors = training(photos=5)
    reference = OodReference.build("toy", vectors)

    assert reference.score(vectors[0], 100) >= 0.0
