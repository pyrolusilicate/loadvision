"""The service runs the encoder weights its model package carries, never a fresh download."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import timm
import timm.models._builder as timm_builder
import torch
from safetensors.torch import save_file

from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.models import backbones
from loadvision.ml.registry import BackboneSpec
from tests.builders import make_image
from tests.model_packages import build_package, member_data

# The smallest ViT in timm, loaded the same way as the checkpoints of our ensemble.
TIMM_NAME = "test_vit.r160_in1k"
SPEC = BackboneSpec(
    timm_name=TIMM_NAME, patch=16, height=32, width=32, license="Toy-1.0", origin="a test"
)
WEIGHTS = "weights/tiny.safetensors"


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes any download fail and puts the tiny encoder into the registry."""

    def no_download(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("the encoder was downloaded")

    monkeypatch.setattr(timm_builder, "load_state_dict_from_hf", no_download)
    monkeypatch.setattr(timm_builder, "load_state_dict_from_url", no_download)
    monkeypatch.setattr(backbones, "BACKBONES", {"tiny": SPEC})


def write_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Saves random weights of the tiny encoder the way a package stores a checkpoint."""
    torch.manual_seed(7)
    source = timm.create_model(TIMM_NAME, pretrained=False, num_classes=0, img_size=(32, 32))
    state = {name: tensor.contiguous() for name, tensor in source.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path))
    return state


@pytest.mark.ml
def test_an_encoder_runs_the_weights_of_the_given_file(tmp_path: Path) -> None:
    saved = write_checkpoint(tmp_path / "tiny.safetensors")

    loaded = backbones.load_backbone("tiny", "cpu", tmp_path / "tiny.safetensors")

    state = loaded.module.state_dict()
    assert saved.keys() == state.keys()
    for name, tensor in saved.items():
        assert torch.equal(state[name], tensor), name


@pytest.mark.ml
def test_a_package_predictor_reads_the_encoder_from_its_own_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "package"
    write_checkpoint(root / WEIGHTS)
    probe = backbones.load_backbone("tiny", "cpu", root / WEIGHTS)
    head = tmp_path / "head.npz"
    np.savez(
        head,
        mean=np.zeros(probe.dim),
        scale=np.ones(probe.dim),
        weights=np.zeros(probe.dim),
        intercept=np.asarray(42.0),
    )
    member = member_data(
        backbone="tiny",
        runtime="torch",
        weights=WEIGHTS,
        preprocessing={"name": probe.preprocessing, "params": dict(probe.params)},
    )
    package = build_package(
        root,
        {"head.npz": head.read_bytes(), WEIGHTS: (root / WEIGHTS).read_bytes()},
        members=[member],
    )
    asked: list[Path | None] = []
    real = backbones.load_backbone

    def spy(name: str, device: str, weights: Path | None = None) -> backbones.Backbone:
        asked.append(weights)
        return real(name, device, weights)

    monkeypatch.setattr(backbones, "load_backbone", spy)

    [prediction] = PackagePredictor(package).predict([make_image()])

    assert asked == [root / WEIGHTS]
    assert prediction.load_pct.value == 42.0
