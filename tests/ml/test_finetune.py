from pathlib import Path
from typing import cast

import numpy as np
import pytest
import timm
import torch
from safetensors.torch import load_file

from loadvision.imaging.crops import far_end
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.preprocessing import preprocess
from loadvision.ml.data.manifest import load_split
from loadvision.ml.inference.predictor import RidgeHead
from loadvision.ml.models.backbones import Backbone
from loadvision.ml.models.depth import DEPTH_DIM, GRID_COLS, GRID_ROWS, mirror_grids
from loadvision.ml.registry import BackboneSpec
from loadvision.ml.training.embedding_cache import read_image
from loadvision.ml.training.finetune import (
    FinetuneSettings,
    fit,
    out_of_fold,
    predict,
    save_model,
)
from tests.ml.test_stages import PARAMS, make_dataset, toy_loader

FOLDS = 3


def quick(**overrides: object) -> FinetuneSettings:
    """Settings that train in seconds on the toy encoder."""
    values: dict[str, object] = {
        "encoder": "toy",
        "epochs": 30,
        "batch_size": 16,
        "lr": 1e-2,
        "head_lr_scale": 1.0,
        "workers": 0,
    }
    values.update(overrides)
    return FinetuneSettings.from_mapping(values)


@pytest.fixture
def photos(tmp_path: Path) -> tuple[list, np.ndarray, list[int]]:
    """Photos whose load is readable from how red they are, split into three folds."""
    samples = load_split(make_dataset(tmp_path / "data"), "train")
    labels = np.array([sample.label for sample in samples], dtype=np.float64)
    return samples, labels, [index % FOLDS for index in range(len(samples))]


def tiny_vit() -> Backbone:
    """A real timm ViT, small and seeded, so the layer-wise decay path runs for real."""
    torch.manual_seed(0)
    spec = BackboneSpec(
        timm_name="vit_tiny_patch16_224",
        patch=16,
        height=32,
        width=32,
        license="Apache-2.0",
        origin="test",
    )
    module = timm.create_model(spec.timm_name, pretrained=False, num_classes=0, img_size=(32, 32))
    return Backbone(
        name="tiny",
        module=module,
        preprocessing=spec.preprocessing,
        params={**PARAMS, "height": 32, "width": 32},
        dim=cast(int, module.num_features),
        spec=spec,
    )


@pytest.mark.ml
def test_every_photo_is_predicted_once_by_a_model_that_never_saw_it(photos) -> None:
    samples, labels, folds = photos

    predicted = out_of_fold(samples, labels, folds, quick(epochs=2), device="cpu", load=toy_loader)

    assert predicted.shape == labels.shape
    assert np.isfinite(predicted).all()
    assert predicted.min() >= 0.0
    assert predicted.max() <= 100.0


@pytest.mark.ml
def test_the_model_learns_the_load_from_the_photo(photos) -> None:
    samples, labels, folds = photos

    predicted = out_of_fold(samples, labels, folds, quick(), device="cpu", load=toy_loader)

    constant = float(np.abs(labels - np.median(labels)).mean())
    assert float(np.abs(labels - predicted).mean()) < 0.6 * constant


@pytest.mark.ml
def test_every_fold_starts_from_fresh_weights(photos) -> None:
    samples, labels, folds = photos
    loaded: list[str] = []

    def counting(name: str, device: str) -> Backbone:
        loaded.append(name)
        return toy_loader(name, device)

    out_of_fold(samples, labels, folds, quick(epochs=1), device="cpu", load=counting)

    # One pretrained copy per fold: reusing a fine-tuned encoder would let it see validation photos.
    assert len(loaded) == FOLDS


@pytest.mark.ml
def test_a_single_fold_leaves_the_others_unpredicted(photos) -> None:
    samples, labels, folds = photos

    predicted = out_of_fold(
        samples, labels, folds, quick(epochs=1), device="cpu", only=1, load=toy_loader
    )

    held = np.array(folds) == 1
    assert np.isfinite(predicted[held]).all()
    assert np.isnan(predicted[~held]).all()


@pytest.mark.ml
def test_the_same_settings_give_the_same_predictions(photos) -> None:
    samples, labels, folds = photos

    first = out_of_fold(samples, labels, folds, quick(epochs=3), device="cpu", load=toy_loader)
    again = out_of_fold(samples, labels, folds, quick(epochs=3), device="cpu", load=toy_loader)

    np.testing.assert_array_equal(first, again)


@pytest.mark.ml
def test_early_blocks_move_less_than_late_ones(photos) -> None:
    samples, labels, _ = photos
    reference = {key: value.clone() for key, value in tiny_vit().module.state_dict().items()}
    settings = quick(encoder="tiny", epochs=2, lr=1e-3, layer_decay=0.5)

    model = fit(samples, labels, settings, device="cpu", load=lambda _name, _device: tiny_vit())

    def moved(prefix: str) -> float:
        now = model.encoder.state_dict()
        keys = [key for key in reference if key.startswith(prefix)]
        change = sum(float((now[key] - reference[key]).norm()) for key in keys)
        return change / sum(float(reference[key].norm()) for key in keys)

    # Layer-wise decay: the first block learns at a small fraction of the last one's rate.
    assert moved("blocks.0.") < moved("blocks.11.")


@pytest.mark.ml
@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"epochs": 3}, "encoder"),
        ({"encoder": "toy", "epocs": 3}, "unknown settings: epocs"),
        ({"encoder": "toy", "layer_decay": 1.5}, "layer_decay"),
        ({"encoder": "toy", "lr": 0}, "lr must be positive"),
    ],
)
def test_malformed_settings_are_refused(raw: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FinetuneSettings.from_mapping(raw)


@pytest.mark.ml
def test_the_saved_head_answers_what_the_model_does(photos, tmp_path: Path) -> None:
    samples, labels, _ = photos
    model = fit(samples, labels, quick(epochs=5), device="cpu", load=toy_loader)

    _, head_file = save_model(model, tmp_path / "model")

    # The service reads the head as a ridge head on the encoder's vector: it must agree with the
    # model to the last digit that matters, or the service and the submission drift apart.
    backbone = toy_loader("toy", "cpu")
    head = RidgeHead.load(head_file)
    with torch.inference_mode():
        served = [
            head.apply(
                backbone.encode(
                    torch.from_numpy(
                        preprocess(backbone.preprocessing, read_image(s.path), backbone.params)
                    )[None].float()
                ).numpy()[0]
            )
            for s in samples
        ]
    answered = predict(model, samples, device="cpu", batch_size=16)
    np.testing.assert_allclose(served, answered, atol=1e-3)


@pytest.mark.ml
def test_the_saved_encoder_loads_back_into_a_fresh_model(photos, tmp_path: Path) -> None:
    samples, labels, _ = photos
    settings = quick(encoder="tiny", epochs=1, lr=1e-3)
    model = fit(samples, labels, settings, device="cpu", load=lambda _name, _device: tiny_vit())

    encoder_file, _ = save_model(model, tmp_path / "model")

    fresh = tiny_vit()
    fresh.module.load_state_dict(load_file(str(encoder_file)))
    fresh.module.eval()
    batch = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        np.testing.assert_allclose(
            fresh.encode(batch).numpy(), model.encoder(batch).numpy(), atol=1e-5
        )


def label_depth(samples, _device: str) -> np.ndarray:
    """A depth grid that carries the load, so the test sees whether depth reaches the answer."""
    loads = {sample.path: sample.label for sample in samples}
    return np.stack([np.full(DEPTH_DIM, loads[s.path] / 100, np.float32) for s in samples])


@pytest.mark.ml
def test_the_far_view_and_depth_feed_the_head(photos, monkeypatch: pytest.MonkeyPatch) -> None:
    samples, labels, _ = photos
    monkeypatch.setattr("loadvision.ml.training.finetune.depth_grids", label_depth)

    model = fit(
        samples, labels, quick(epochs=5, far_view=True, depth=True), device="cpu", load=toy_loader
    )

    # The encoder vector of the whole frame, the one of the crop, and the depth grid.
    assert model.head.in_features == 2 * toy_loader("toy", "cpu").dim + DEPTH_DIM
    with_depth = predict(model, samples, device="cpu", batch_size=16)
    monkeypatch.setattr(
        "loadvision.ml.training.finetune.depth_grids",
        lambda s, _d: np.zeros((len(s), DEPTH_DIM), np.float32),
    )
    without_depth = predict(model, samples, device="cpu", batch_size=16)
    assert not np.allclose(with_depth, without_depth)


@pytest.mark.ml
def test_the_far_view_model_learns_the_load(photos) -> None:
    samples, labels, folds = photos

    predicted = out_of_fold(
        samples, labels, folds, quick(far_view=True), device="cpu", load=toy_loader
    )

    constant = float(np.abs(labels - np.median(labels)).mean())
    assert float(np.abs(labels - predicted).mean()) < 0.6 * constant


@pytest.mark.unit
def test_the_far_end_crop_keeps_the_proportions_and_the_middle() -> None:
    pixels = np.zeros((400, 300, 3), np.uint8)
    pixels[200, 150] = 255
    image = PreparedImage(b"x", MediaType.PNG, pixels)

    crop = far_end(image)

    assert crop.pixels.shape == (200, 150, 3)
    assert crop.pixels.max() == 255


@pytest.mark.unit
def test_a_mirrored_depth_grid_has_its_columns_reversed() -> None:
    grid = torch.arange(DEPTH_DIM, dtype=torch.float32).reshape(1, -1)

    mirrored = mirror_grids(grid).reshape(GRID_ROWS, GRID_COLS)

    assert mirrored[0].tolist() == list(reversed(grid.reshape(GRID_ROWS, GRID_COLS)[0].tolist()))
