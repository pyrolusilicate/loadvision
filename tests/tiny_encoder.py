"""The smallest ViT in timm as a registry backbone, for tests that run the torch path offline."""

from pathlib import Path
from typing import Any

import pytest
import timm
import timm.models._builder as timm_builder
import torch
from safetensors.torch import save_file

from loadvision.ml.models import backbones
from loadvision.ml.registry import BackboneSpec

# The smallest ViT in timm, loaded the same way as the checkpoints of our ensemble.
TIMM_NAME = "test_vit.r160_in1k"
SPEC = BackboneSpec(
    timm_name=TIMM_NAME, patch=16, height=32, width=32, license="Toy-1.0", origin="a test"
)
BACKBONE = "tiny"
WEIGHTS = "weights/tiny.safetensors"


def use_tiny_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes any download fail and puts the tiny encoder into the registry."""

    def no_download(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("the encoder was downloaded")

    monkeypatch.setattr(timm_builder, "load_state_dict_from_hf", no_download)
    monkeypatch.setattr(timm_builder, "load_state_dict_from_url", no_download)
    monkeypatch.setattr(backbones, "BACKBONES", {BACKBONE: SPEC})


def write_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Saves random weights of the tiny encoder the way a package stores a checkpoint."""
    torch.manual_seed(7)
    source = timm.create_model(TIMM_NAME, pretrained=False, num_classes=0, img_size=(32, 32))
    state = {name: tensor.contiguous() for name, tensor in source.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path))
    return state
