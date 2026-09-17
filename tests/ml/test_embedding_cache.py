from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loadvision.ml.data.manifest import Sample, load_split
from loadvision.ml.models.backbones import Backbone
from loadvision.ml.registry import BackboneSpec
from loadvision.ml.training.embedding_cache import embed_samples, read_image

DIM = 4
SPEC = BackboneSpec(timm_name="fake", patch=8, height=32, width=24)
PARAMS = {
    "height": 32,
    "width": 24,
    "mean_r": 0.5,
    "mean_g": 0.5,
    "mean_b": 0.5,
    "std_r": 0.5,
    "std_g": 0.5,
    "std_b": 0.5,
}


class CountingEncoder(torch.nn.Module):
    """Stand-in for a real encoder: returns one number per channel and counts the photos seen."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = 0

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.seen += batch.shape[0]
        means = batch.mean(dim=(2, 3))
        return torch.cat([means, means[:, :1]], dim=1)


def fake_backbone() -> tuple[Backbone, CountingEncoder]:
    encoder = CountingEncoder()
    return Backbone(name="fake", module=encoder, params=PARAMS, dim=DIM, spec=SPEC), encoder


def make_photo(directory: Path, image_id: str, color: tuple[int, int, int]) -> Sample:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{image_id}.jpg"
    Image.new("RGB", (40, 30), color).save(path)
    prepared = read_image(path)
    return Sample(image_id=image_id, path=path, sha256=prepared.sha256, label=None, group_id=None)


@pytest.mark.ml
def test_returns_one_vector_per_sample_in_order(tmp_path: Path) -> None:
    backbone, encoder = fake_backbone()
    samples = [
        make_photo(tmp_path / "img", "a", (255, 0, 0)),
        make_photo(tmp_path / "img", "b", (0, 255, 0)),
    ]

    vectors = embed_samples(samples, backbone, cache_dir=tmp_path / "cache", device="cpu")

    assert vectors.shape == (2, DIM)
    assert encoder.seen == 2
    # The first photo is red and the second green, so the channel means order them apart.
    assert vectors[0][0] > vectors[1][0]
    assert vectors[1][1] > vectors[0][1]


@pytest.mark.ml
def test_second_call_reads_the_cache_instead_of_the_encoder(tmp_path: Path) -> None:
    backbone, encoder = fake_backbone()
    samples = [make_photo(tmp_path / "img", "a", (255, 0, 0))]
    cache_dir = tmp_path / "cache"

    first = embed_samples(samples, backbone, cache_dir=cache_dir, device="cpu")
    second = embed_samples(samples, backbone, cache_dir=cache_dir, device="cpu")

    assert encoder.seen == 1
    assert np.array_equal(first, second)


@pytest.mark.ml
def test_cache_is_keyed_by_photo_hash_not_by_image_id(tmp_path: Path) -> None:
    backbone, encoder = fake_backbone()
    original = make_photo(tmp_path / "img", "a", (255, 0, 0))
    renamed = make_photo(tmp_path / "img", "copy_of_a", (255, 0, 0))
    cache_dir = tmp_path / "cache"

    embed_samples([original], backbone, cache_dir=cache_dir, device="cpu")
    embed_samples([renamed], backbone, cache_dir=cache_dir, device="cpu")

    assert original.sha256 == renamed.sha256
    assert encoder.seen == 1


@pytest.mark.ml
def test_only_missing_photos_go_through_the_encoder(tmp_path: Path) -> None:
    backbone, encoder = fake_backbone()
    first = make_photo(tmp_path / "img", "a", (255, 0, 0))
    second = make_photo(tmp_path / "img", "b", (0, 255, 0))
    cache_dir = tmp_path / "cache"

    embed_samples([first], backbone, cache_dir=cache_dir, device="cpu")
    embed_samples([first, second], backbone, cache_dir=cache_dir, device="cpu")

    assert encoder.seen == 2


@pytest.mark.ml
def test_cache_of_another_backbone_size_is_rejected(tmp_path: Path) -> None:
    backbone, _ = fake_backbone()
    samples = [make_photo(tmp_path / "img", "a", (255, 0, 0))]
    cache_dir = tmp_path / "cache"
    embed_samples(samples, backbone, cache_dir=cache_dir, device="cpu")

    wider = Backbone(name="fake", module=backbone.module, params=PARAMS, dim=DIM + 1, spec=SPEC)
    with pytest.raises(ValueError, match="does not match the backbone"):
        embed_samples(samples, wider, cache_dir=cache_dir, device="cpu")


@pytest.mark.ml
def test_cache_built_with_other_preprocessing_is_rejected(tmp_path: Path) -> None:
    backbone, encoder = fake_backbone()
    samples = [make_photo(tmp_path / "img", "a", (255, 0, 0))]
    cache_dir = tmp_path / "cache"
    embed_samples(samples, backbone, cache_dir=cache_dir, device="cpu")

    rescaled = Backbone(
        name="fake", module=encoder, params={**PARAMS, "mean_r": 0.25}, dim=DIM, spec=SPEC
    )
    with pytest.raises(ValueError, match="other preprocessing"):
        embed_samples(samples, rescaled, cache_dir=cache_dir, device="cpu")


@pytest.mark.ml
def test_unsupported_image_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "photo.gif"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(path)

    with pytest.raises(ValueError, match="unsupported image format"):
        read_image(path)


@pytest.mark.data
@pytest.mark.skipif(not Path("data/train").is_dir(), reason="dataset is not present")
def test_real_photo_decodes_to_rgb_pixels() -> None:
    sample = load_split(Path("data"), "train")[0]

    prepared = read_image(sample.path)

    assert prepared.sha256 == sample.sha256
    assert prepared.pixels.shape[2] == 3
    assert prepared.height > 0 and prepared.width > 0
