import io
from typing import cast

import numpy as np
import pytest
import timm
import torch
from PIL import Image

from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.ml.inference.explain import Explainer
from loadvision.ml.inference.heatmap import render
from loadvision.ml.inference.predictor import RidgeHead
from loadvision.ml.models.backbones import Backbone
from loadvision.ml.registry import BackboneSpec
from tests.ml.test_stages import PARAMS

ROWS, COLS = 4, 3


def small_vit() -> Backbone:
    """A real timm ViT with registers, like DINOv2, on a 4x3 patch grid."""
    torch.manual_seed(0)
    spec = BackboneSpec(
        timm_name="vit_tiny_patch16_224", patch=16, height=64, width=48, license="x", origin="x"
    )
    module = timm.create_model(
        spec.timm_name, pretrained=False, num_classes=0, img_size=(64, 48), reg_tokens=4
    ).eval()
    return Backbone(
        name="small",
        module=module,
        preprocessing=spec.preprocessing,
        params={**PARAMS, "height": 64, "width": 48},
        dim=cast(int, module.num_features),
        spec=spec,
    )


def head(backbone: Backbone, sign: float = 1.0) -> RidgeHead:
    weights = np.random.default_rng(0).normal(size=backbone.dim) * sign
    return RidgeHead(
        mean=np.zeros(backbone.dim), scale=np.ones(backbone.dim), weights=weights, intercept=50.0
    )


def photo(width: int = 60, height: int = 80) -> PreparedImage:
    pixels = np.random.default_rng(1).integers(0, 255, (height, width, 3), dtype=np.uint8)
    return PreparedImage(b"x", MediaType.PNG, pixels)


@pytest.mark.ml
def test_attention_is_always_there_and_contribution_only_when_asked() -> None:
    backbone = small_vit()
    explainer = Explainer(backbone, head(backbone), "cpu")

    fast = explainer.explain(photo(), contribution=False)
    full = explainer.explain(photo())

    assert fast.contribution is None
    assert fast.attention.shape == (ROWS, COLS)
    assert fast.attention.min() >= 0.0 and fast.attention.max() == pytest.approx(1.0)
    assert full.contribution is not None
    assert full.contribution.shape == (ROWS, COLS)
    assert -1.0 <= full.contribution.min() < 0.0 < full.contribution.max() <= 1.0
    # Asking for contribution changes nothing about where the model looks.
    np.testing.assert_allclose(fast.attention, full.attention, atol=1e-5)


@pytest.mark.ml
def test_a_head_that_reads_the_features_backwards_flips_every_contribution() -> None:
    backbone = small_vit()

    plain = Explainer(backbone, head(backbone), "cpu").explain(photo()).contribution
    flipped = Explainer(backbone, head(backbone, sign=-1.0), "cpu").explain(photo()).contribution

    assert plain is not None and flipped is not None
    np.testing.assert_allclose(flipped, -plain, atol=1e-5)


@pytest.mark.ml
@pytest.mark.parametrize("size", [(60, 80), (160, 60)])
def test_the_map_is_drawn_over_the_photo_at_its_own_size(size: tuple[int, int]) -> None:
    backbone = small_vit()
    image = photo(*size)
    explanation = Explainer(backbone, head(backbone), "cpu").explain(image)

    for kind in ("attention", "contribution"):
        drawn = Image.open(io.BytesIO(render(image, explanation, kind)))
        # A wide photo sits in the tall box with bars above and below; the bars are cut off.
        assert drawn.size == size


@pytest.mark.ml
def test_drawing_a_contribution_that_was_not_computed_is_refused() -> None:
    backbone = small_vit()
    image = photo()
    explanation = Explainer(backbone, head(backbone), "cpu").explain(image, contribution=False)

    with pytest.raises(ValueError, match="not computed"):
        render(image, explanation, "contribution")


@pytest.mark.ml
def test_a_drawn_map_has_no_holes() -> None:
    backbone = small_vit()
    image = photo()
    explanation = Explainer(backbone, head(backbone), "cpu").explain(image)

    for kind in ("attention", "contribution"):
        drawn = np.asarray(Image.open(io.BytesIO(render(image, explanation, kind))))
        # Stretching the grid overshoots below zero; a power of a negative number is NaN, and
        # NaN turned into a pixel is a black hole in the picture.
        assert (drawn.sum(-1) > 0).all()
