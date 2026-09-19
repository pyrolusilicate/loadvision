"""The service runs the encoder weights its model package carries, never a fresh download."""

from pathlib import Path

import numpy as np
import pytest
import torch

from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.models import backbones
from tests.builders import make_image
from tests.model_packages import build_package, member_data
from tests.tiny_encoder import BACKBONE, WEIGHTS, use_tiny_encoder, write_checkpoint


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    use_tiny_encoder(monkeypatch)


@pytest.mark.ml
def test_an_encoder_runs_the_weights_of_the_given_file(tmp_path: Path) -> None:
    saved = write_checkpoint(tmp_path / "tiny.safetensors")

    loaded = backbones.load_backbone(BACKBONE, "cpu", tmp_path / "tiny.safetensors")

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
    probe = backbones.load_backbone(BACKBONE, "cpu", root / WEIGHTS)
    head = tmp_path / "head.npz"
    np.savez(
        head,
        mean=np.zeros(probe.dim),
        scale=np.ones(probe.dim),
        weights=np.zeros(probe.dim),
        intercept=np.asarray(42.0),
    )
    member = member_data(
        backbone=BACKBONE,
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
