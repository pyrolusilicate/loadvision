from pathlib import Path
from typing import Any

import numpy as np
import pytest
import timm
import timm.models._builder as timm_builder
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file

from loadvision.imaging.crops import FAR_END, WHOLE
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.models import backbones, depth
from loadvision.ml.models.depth import DEPTH_DIM, GRID_COLS, GRID_ROWS
from loadvision.ml.registry import BackboneSpec
from tests.model_packages import build_package, member_data

TIMM_NAME = "test_vit.r160_in1k"
WEIGHTS = "weights/tiny.safetensors"
DEPTH_DIR = "depth"
SPEC = BackboneSpec(
    timm_name=TIMM_NAME, patch=16, height=32, width=32, license="Toy-1.0", origin="a test"
)
# What the fake depth model answers, and the weight the head gives its first cell.
GRID_VALUE = 0.5
CELL_WEIGHT = 40.0
BASE = 50.0
# How much the next fold's head answers above the first one.
FOLD_STEP = 20.0
# How much of the first embedding feature the head reads.
EMBEDDING_WEIGHT = 1.0


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Puts the tiny encoder into the registry and makes any download fail."""

    def no_download(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("the encoder was downloaded")

    monkeypatch.setattr(timm_builder, "load_state_dict_from_hf", no_download)
    monkeypatch.setattr(timm_builder, "load_state_dict_from_url", no_download)
    monkeypatch.setattr(backbones, "BACKBONES", {"tiny": SPEC})


@pytest.fixture
def fake_depth(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """A depth model that answers a constant grid, so no checkpoint is downloaded."""
    asked: list[Path] = []

    class Fake:
        @classmethod
        def load(cls, source: str | Path = "", device: str = "cpu") -> "Fake":
            asked.append(Path(source))
            return cls()

        def grid(self, _image: PreparedImage) -> NDArray[np.float32]:
            return np.full(DEPTH_DIM, GRID_VALUE, dtype=np.float32)

    monkeypatch.setattr(depth, "DepthEncoder", Fake)
    return asked


def make_image(height: int = 8, width: int = 6) -> PreparedImage:
    pixels: NDArray[np.uint8] = np.zeros((height, width, 3), dtype=np.uint8)
    return PreparedImage(b"photo", MediaType.JPEG, pixels)


def _write_weights(path: Path, seed: int) -> Path:
    """Saves the tiny encoder with its own random weights."""
    torch.manual_seed(seed)
    module = timm.create_model(TIMM_NAME, pretrained=False, num_classes=0, img_size=(32, 32))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: t.contiguous() for name, t in module.state_dict().items()}, str(path))
    return path


def build(
    root: Path, *, with_depth: bool, inputs: tuple[str, ...] = (WHOLE,), folds: int = 1
) -> Any:
    """A package of one tiny torch member whose head reads the embedding and the depth grid."""
    weights_path = _write_weights(root / WEIGHTS, seed=7)
    probe = backbones.load_backbone("tiny", "cpu", weights_path)
    width = probe.dim * len(inputs) + (DEPTH_DIM if with_depth else 0)
    head_weights = np.zeros(width)
    # The first feature of the embedding is read, so the answer depends on which weights ran.
    head_weights[0] = EMBEDDING_WEIGHT
    if with_depth:
        # Only the first cell of the depth grid moves the answer, so the test can predict it.
        head_weights[probe.dim * len(inputs)] = CELL_WEIGHT
    files = {WEIGHTS: weights_path.read_bytes()}
    members = []
    for fold in range(folds):
        # Every fold of a cross-validation has its own weights and its own head; here the
        # heads differ by what they answer, so a test sees which of them actually ran.
        weights = WEIGHTS if fold == 0 else f"weights/fold_{fold}.safetensors"
        if fold:
            # Другие веса того же энкодера: так видно, что каждый член считается своими.
            files[weights] = _write_weights(root / weights, seed=7 + fold).read_bytes()
        head = f"m{fold}/head.npz"
        path = root / head
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=np.zeros(width),
            scale=np.ones(width),
            weights=head_weights,
            intercept=np.asarray(BASE + FOLD_STEP * fold),
        )
        files[head] = path.read_bytes()
        members.append(
            member_data(
                backbone="tiny",
                runtime="torch",
                weights=weights,
                head=head,
                weight=1 / folds,
                preprocessing={"name": probe.preprocessing, "params": dict(probe.params)},
                inputs=list(inputs),
            )
        )
    extra: dict[str, Any] = {}
    if with_depth:
        files[f"{DEPTH_DIR}/config.json"] = b"{}"
        extra["depth"] = {"model": DEPTH_DIR, "rows": GRID_ROWS, "cols": GRID_COLS}
    return build_package(root, files, members=members, **extra)


def _expected(package: Any, grid_value: float) -> float:
    """What the head answers: its intercept, the embedding it reads and the depth cell."""
    member = package.manifest.members[0]
    probe = backbones.load_backbone("tiny", "cpu", package.path(member.weights))
    vector = probe.embed([make_image()], "cpu").numpy()[0]
    return BASE + EMBEDDING_WEIGHT * float(vector[0]) + CELL_WEIGHT * grid_value


@pytest.mark.ml
def test_a_member_of_a_depth_package_is_fed_the_grid_after_its_embedding(
    tmp_path: Path, fake_depth: list[Path]
) -> None:
    package = build(tmp_path / "package", with_depth=True)

    [prediction] = PackagePredictor(package).predict([make_image()])

    # The head reads one cell of the grid; the encoder part of its weights is zero.
    assert prediction.load_pct.value == pytest.approx(_expected(package, GRID_VALUE))
    assert fake_depth == [tmp_path / "package" / DEPTH_DIR]


@pytest.mark.ml
def test_a_package_without_a_depth_block_never_loads_the_depth_model(
    tmp_path: Path, fake_depth: list[Path]
) -> None:
    package = build(tmp_path / "package", with_depth=False)

    [prediction] = PackagePredictor(package).predict([make_image()])

    assert prediction.load_pct.value == pytest.approx(_expected(package, 0.0))
    assert fake_depth == []


@pytest.mark.ml
def test_a_member_shown_the_far_end_gets_a_vector_for_every_input(
    tmp_path: Path, fake_depth: list[Path]
) -> None:
    package = build(tmp_path / "package", with_depth=True, inputs=(WHOLE, FAR_END))

    # Built first: the predictor checks itself on a blank photo, and those runs are not the
    # ones under test.
    predictor = PackagePredictor(package)
    seen: list[tuple[int, int]] = []
    real = backbones.Backbone.prepare

    def spy(self: backbones.Backbone, images: Any, view: Any) -> Any:
        seen.append((images[0].height, images[0].width))
        return real(self, images, view)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backbones.Backbone, "prepare", spy)
        [prediction] = predictor.predict([make_image(80, 60)])

    # The whole frame and the crop of its far end, each once; the head still reads the grid.
    assert sorted(seen) == [(40, 30), (80, 60)]
    assert prediction.load_pct.value is not None


@pytest.mark.ml
def test_members_of_one_checkpoint_with_different_weights_each_run_their_own(
    tmp_path: Path, fake_depth: list[Path]
) -> None:
    """Five models of a cross-validation name one architecture and five sets of weights."""
    root = tmp_path / "package"
    package = build(root, with_depth=False, folds=2)

    [prediction] = PackagePredictor(package).predict([make_image()])

    # Every member ran its own weights, so the two answers differ; one encoder for both
    # would leave them apart by exactly the step between their heads.
    assert prediction.uncertainty is not None
    assert abs(prediction.uncertainty - FOLD_STEP) > 1e-6
