from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pytest
import timm
import timm.models._builder as timm_builder
import torch
from numpy.typing import NDArray
from safetensors.numpy import save_file as save_numpy
from safetensors.torch import save_file

from loadvision.imaging.crops import FAR_END, WHOLE
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.ml.inference.predictor import BinsHead, PackagePredictor
from loadvision.ml.models import backbones, depth
from loadvision.ml.models.depth import DEPTH_DIM, GRID_COLS, GRID_ROWS
from loadvision.ml.packaging.model_package import ModelPackageError
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


@pytest.mark.ml
def test_the_bins_head_answers_the_mean_of_its_own_distribution() -> None:
    weights = np.zeros((21, 4), dtype=np.float32)
    bias = np.full(21, -50.0, dtype=np.float32)
    bias[10] = 0.0  # вся масса в середине сетки, то есть ровно 50%
    with TemporaryDirectory() as folder:
        path = Path(folder) / "head.safetensors"
        save_numpy({"layer.weight": weights, "layer.bias": bias}, str(path))
        head = BinsHead.load(path)

    assert head.apply(np.zeros(4)) == pytest.approx(50.0, abs=0.01)


@pytest.mark.ml
def test_the_bins_head_never_leaves_the_possible_range() -> None:
    rng = np.random.default_rng(0)
    weights = rng.normal(scale=20.0, size=(21, 4)).astype(np.float32)
    with TemporaryDirectory() as folder:
        path = Path(folder) / "head.safetensors"
        save_numpy(
            {"layer.weight": weights, "layer.bias": np.zeros(21, dtype=np.float32)}, str(path)
        )
        head = BinsHead.load(path)

    answers = [head.apply(rng.normal(size=4) * 10) for _ in range(50)]

    assert all(0.0 <= value <= 100.0 for value in answers)


@pytest.mark.ml
def test_a_head_file_without_the_layer_is_refused() -> None:
    with TemporaryDirectory() as folder:
        path = Path(folder) / "head.safetensors"
        save_numpy({"something.else": np.zeros(3, dtype=np.float32)}, str(path))

        with pytest.raises(ModelPackageError, match=r"layer\.weight"):
            BinsHead.load(path)


def bins_head(folder: str, centres: NDArray[np.float32] | None, bins: int = 21) -> Path:
    """Writes a head of `bins` bins, with the grid it was fitted on when one is given."""
    path = Path(folder) / "head.safetensors"
    stored: dict[str, NDArray[Any]] = {
        "layer.weight": np.zeros((bins, 4), dtype=np.float32),
        "layer.bias": np.zeros(bins, dtype=np.float32),
    }
    if centres is not None:
        stored["centres"] = centres
    save_numpy(stored, str(path))
    return path


@pytest.mark.ml
def test_the_grid_the_training_wrote_is_read_as_shares() -> None:
    """Training keeps the centres as shares of a full body; the service answers in percent."""
    with TemporaryDirectory() as folder:
        path = bins_head(folder, np.linspace(0.0, 1.0, 21, dtype=np.float32))

        head = BinsHead.load(path)

    assert head.centres[0] == 0.0
    assert head.centres[-1] == 100.0


@pytest.mark.ml
def test_a_head_fitted_on_uneven_bins_is_refused() -> None:
    """Such a head loads and answers all the same, only wrong, so it must not load at all."""
    packed = np.linspace(0.0, 1.0, 21, dtype=np.float32) ** 2
    with TemporaryDirectory() as folder:
        path = bins_head(folder, packed)

        with pytest.raises(ModelPackageError, match="equal steps"):
            BinsHead.load(path)


@pytest.mark.ml
def test_a_head_whose_bins_stop_short_of_a_full_body_is_refused() -> None:
    with TemporaryDirectory() as folder:
        path = bins_head(folder, np.linspace(0.0, 0.8, 21, dtype=np.float32))

        with pytest.raises(ModelPackageError, match="equal steps"):
            BinsHead.load(path)


@pytest.mark.ml
def test_a_head_carrying_the_wrong_number_of_centres_is_refused() -> None:
    with TemporaryDirectory() as folder:
        path = bins_head(folder, np.linspace(0.0, 1.0, 20, dtype=np.float32))

        with pytest.raises(ModelPackageError, match="20 bin centres for 21 bins"):
            BinsHead.load(path)


@pytest.mark.ml
def test_a_head_without_a_grid_of_its_own_is_read_as_before() -> None:
    """Every head built before 21.09 carries no centres, and those packages still load."""
    with TemporaryDirectory() as folder:
        head = BinsHead.load(bins_head(folder, None))

    assert head.centres.size == 21


@pytest.mark.ml
def test_the_bins_head_refuses_a_vector_of_the_wrong_length() -> None:
    with TemporaryDirectory() as folder:
        path = Path(folder) / "head.safetensors"
        save_numpy(
            {
                "layer.weight": np.zeros((21, 4), dtype=np.float32),
                "layer.bias": np.zeros(21, dtype=np.float32),
            },
            str(path),
        )
        head = BinsHead.load(path)

    with pytest.raises(ModelPackageError, match="expects 4 features"):
        head.apply(np.zeros(9))
