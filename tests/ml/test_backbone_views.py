"""Averaging over views: the part that must not blend the left of a body with its right."""

import numpy as np
import pytest
import torch

from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.views import get
from loadvision.ml.models.backbones import Backbone
from loadvision.ml.registry import BackboneSpec

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
ROWS, COLS, WIDTH = 4, 3, 2


class ColumnEncoder(torch.nn.Module):
    """Toy encoder whose token depends on where the token sits, so a mirror is visible."""

    num_prefix_tokens = 0

    def forward_features(self, batch: torch.Tensor) -> torch.Tensor:
        photos = batch.shape[0]
        rows, cols = 32 // 8, 24 // 8
        column = torch.arange(cols, dtype=torch.float32).repeat(rows)
        brightness = batch.mean(dim=(1, 2, 3))[:, None]
        tokens = torch.stack([column.repeat(photos, 1), column.repeat(photos, 1)], dim=2)
        return tokens + brightness[:, :, None]

    def forward_head(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)


def backbone(pooling: str, views: tuple[str, ...]) -> Backbone:
    spec = BackboneSpec(
        timm_name="fake",
        patch=8,
        height=32,
        width=24,
        license="Toy-1.0",
        origin="a test",
        pooling="grid" if pooling == "grid" else "pooled",
        grid_rows=ROWS if pooling == "grid" else 1,
        grid_cols=COLS if pooling == "grid" else 1,
        views=views,
    )
    dim = WIDTH * (ROWS * COLS if pooling == "grid" else 1)
    return Backbone(
        name="toy",
        module=ColumnEncoder(),
        preprocessing=spec.preprocessing,
        params=PARAMS,
        dim=dim,
        spec=spec,
    )


def photo(value: int = 128) -> PreparedImage:
    pixels = np.full((30, 40, 3), value, dtype=np.uint8)
    pixels[:, :20] = 255
    return PreparedImage(b"photo", MediaType.JPEG, pixels)


@pytest.mark.ml
def test_unmirror_reverses_the_columns_of_a_grid() -> None:
    encoder = backbone("grid", ("identity",))
    # One photo, one channel per cell, values numbered so the order is readable.
    vectors = torch.arange(WIDTH * ROWS * COLS, dtype=torch.float32)[None]

    restored = encoder.unmirror(vectors)

    expected = vectors.reshape(1, WIDTH, ROWS, COLS).flip(-1).reshape(1, -1)
    assert torch.equal(restored, expected)
    # Column 0 and column 2 of every row swapped; column 1 stayed.
    assert restored[0, 0].item() == vectors[0, 2].item()
    assert restored[0, 1].item() == vectors[0, 1].item()


@pytest.mark.ml
def test_unmirror_leaves_a_pooled_encoder_alone() -> None:
    encoder = backbone("pooled", ("identity",))
    vectors = torch.arange(WIDTH, dtype=torch.float32)[None]

    assert torch.equal(encoder.unmirror(vectors), vectors)


@pytest.mark.ml
def test_a_single_identity_view_changes_nothing() -> None:
    encoder = backbone("grid", ("identity",))
    images = [photo()]

    embedded = encoder.embed(images, "cpu")

    direct = encoder.encode(encoder.prepare(images, get("identity")))
    assert torch.allclose(embedded, direct)


@pytest.mark.ml
def test_two_views_are_averaged_after_the_columns_are_put_back() -> None:
    encoder = backbone("grid", ("identity", "mirror"))
    images = [photo()]

    embedded = encoder.embed(images, "cpu")

    straight = encoder.encode(encoder.prepare(images, get("identity")))
    mirrored = encoder.unmirror(encoder.encode(encoder.prepare(images, get("mirror"))))
    assert torch.allclose(embedded, (straight + mirrored) / 2)
    # Without putting the columns back the answer would differ: the toy encoder is asymmetric.
    raw_mirror = encoder.encode(encoder.prepare(images, get("mirror")))
    assert not torch.allclose(embedded, (straight + raw_mirror) / 2)


@pytest.mark.ml
def test_the_registry_refuses_an_unknown_view() -> None:
    with pytest.raises(ValueError, match="unknown views"):
        BackboneSpec(
            timm_name="fake",
            patch=8,
            height=32,
            width=24,
            license="Toy-1.0",
            origin="a test",
            views=("identity", "zoom"),
        )


@pytest.mark.ml
def test_the_cache_fingerprint_changes_when_a_view_is_added() -> None:
    from loadvision.ml.training.embedding_cache import preprocessing_fingerprint

    plain = preprocessing_fingerprint("resize_pad", PARAMS, views=("identity",))
    with_mirror = preprocessing_fingerprint("resize_pad", PARAMS, views=("identity", "mirror"))

    # Otherwise vectors computed with one view list would quietly mix with another's.
    assert plain != with_mirror
