"""Retraining that keeps the encoders of a package and fits its heads again."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from loadvision.ml.data.manifest import load_split
from loadvision.ml.models.depth import GRID_COLS, GRID_ROWS
from loadvision.ml.packaging.model_package import ModelPackage, ModelPackageError
from loadvision.ml.training import refit
from loadvision.ml.training.refit import refit_heads
from tests.ml.test_stages import make_dataset, make_folds
from tests.model_packages import build_package, member_data

# `make_folds` of the stage tests builds three; the package must carry one member per fold.
FOLDS = 3
# A head of this many numbers, so a fitted ridge has something to stand on without being slow.
FEATURES = 12


def per_fold_members(root: Path) -> ModelPackage:
    """A package shaped like the served one: one member per fold, each with its own weights."""
    return build_package(
        root,
        {f"m{index}/head.npz": b"head" for index in range(FOLDS)}
        | {f"weights/m{index}.safetensors": f"weights {index}".encode() for index in range(FOLDS)},
        members=[
            member_data(
                backbone="dinov2_vitb14",
                runtime="torch",
                head=f"m{index}/head.npz",
                weights=f"weights/m{index}.safetensors",
                weight=1.0 / FOLDS,
            )
            for index in range(FOLDS)
        ],
    )


def plain_features() -> Any:
    """Returns a `_features` that needs no encoder: every column simply carries the label."""

    def features(
        _active: ModelPackage,
        _member: Any,
        index: int,
        samples: Sequence[Any],
        _cache_dir: Path,
        _device: str,
        _depth: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(index)
        labels = np.array([sample.label for sample in samples], dtype=np.float64)
        return rng.normal(size=(len(samples), FEATURES)) + labels[:, None]

    return features


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModelPackage, Path, Path]:
    """A package, a dataset whose photos all have a fold, and the folds file.

    The work happens inside `tmp_path`, because the manifest stores the folds file as a
    relative path, exactly as the service and the worker pass it.
    """
    monkeypatch.chdir(tmp_path)
    data = make_dataset(Path("data"))
    folds = make_folds(data, Path("folds.json"))
    return per_fold_members(Path("active")), data, folds


@pytest.mark.ml
def test_a_package_with_fewer_members_than_folds_is_refused(
    tmp_path: Path, prepared: tuple[ModelPackage, Path, Path]
) -> None:
    """One member cannot cover five folds: some fold would be scored by an encoder that saw it."""
    _package, data, folds = prepared
    single = build_package(Path("single"), members=[member_data()])

    with pytest.raises(ModelPackageError, match="one member per fold"):
        refit_heads(
            active=single,
            dataset=data,
            folds_file=folds,
            cache_dir=Path("cache"),
            out_dir=Path("out"),
            version="candidate",
        )


@pytest.mark.ml
def test_the_held_out_fold_of_every_member_is_read_off_the_numbers() -> None:
    """The manifest names no fold, so the mapping is measured; here member i owns fold i."""
    labels = np.repeat([10.0, 20.0, 30.0], 4)
    numbers = np.repeat([0, 1, 2], 4)
    computed = {
        member: {
            fold: labels[numbers == fold] + (50.0 if fold == member else 0.0) for fold in range(3)
        }
        for member in range(3)
    }

    raw = refit._out_of_fold(computed, labels, numbers, 3)

    # Every photo was scored by the member that is worst on its fold, so every value is off by
    # the 50 that marks it; nothing was scored by a member that had already seen it.
    assert np.allclose(raw, labels + 50.0)


@pytest.mark.ml
def test_members_that_do_not_mark_one_fold_each_are_refused() -> None:
    """Two members worst on the same fold means the mapping is a guess, and a guess is a lie."""
    labels = np.repeat([10.0, 20.0, 30.0], 4)
    numbers = np.repeat([0, 1, 2], 4)
    computed = {
        member: {fold: labels[numbers == fold] + (50.0 if fold == 0 else 0.0) for fold in range(3)}
        for member in range(3)
    }

    with pytest.raises(ModelPackageError, match="which fold each member"):
        refit._out_of_fold(computed, labels, numbers, 3)


@pytest.mark.ml
def test_the_candidate_keeps_everything_but_the_heads(
    tmp_path: Path, prepared: tuple[ModelPackage, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weights, the licences and the reference travel unchanged; only the heads are new."""
    active, data, folds = prepared
    monkeypatch.setattr(refit, "_features", plain_features())
    monkeypatch.setattr(refit, "_out_of_fold", lambda computed, labels, numbers, n: labels.copy())

    root = refit_heads(
        active=active,
        dataset=data,
        folds_file=folds,
        cache_dir=Path("cache"),
        out_dir=Path("candidate"),
        version="candidate",
    )

    candidate = ModelPackage.load(root)
    assert candidate.version.value == "candidate"
    for index in range(FOLDS):
        weights = f"weights/m{index}.safetensors"
        assert (root / weights).read_bytes() == (active.root / weights).read_bytes()
        assert (root / f"m{index}/head.npz").read_bytes() != b"head"
        # The weights are shared with the active package to save two gigabytes a run, so a head
        # written through a link would overwrite the model the service is answering with.
        assert (active.root / f"m{index}/head.npz").read_bytes() == b"head"


@pytest.mark.ml
def test_the_candidate_carries_the_predictions_the_gate_reads(
    tmp_path: Path, prepared: tuple[ModelPackage, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `metrics/oof.csv` the gate has nothing to compare and refuses every candidate."""
    active, data, folds = prepared
    monkeypatch.setattr(refit, "_features", plain_features())
    monkeypatch.setattr(refit, "_out_of_fold", lambda computed, labels, numbers, n: labels.copy())

    root = refit_heads(
        active=active,
        dataset=data,
        folds_file=folds,
        cache_dir=Path("cache"),
        out_dir=Path("candidate"),
        version="candidate",
    )

    rows = (root / refit.PACKAGE_OOF).read_text(encoding="utf-8").splitlines()
    photos = len(load_split(data, "train"))

    assert rows[0] == "image_id,fold,label,raw,predicted"
    assert len(rows) == photos + 1


@pytest.mark.ml
def test_a_filesystem_that_refuses_a_link_still_gets_the_weights(
    prepared: tuple[ModelPackage, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two mounts under Docker cannot be linked across; the candidate must still be complete."""
    active, data, folds = prepared
    monkeypatch.setattr(refit, "_features", plain_features())
    monkeypatch.setattr(refit, "_out_of_fold", lambda computed, labels, numbers, n: labels.copy())

    def refuse(_source: object, _target: object) -> None:
        raise OSError("Invalid cross-device link")

    monkeypatch.setattr(refit.os, "link", refuse)

    root = refit_heads(
        active=active,
        dataset=data,
        folds_file=folds,
        cache_dir=Path("cache"),
        out_dir=Path("candidate"),
        version="candidate",
    )

    for index in range(FOLDS):
        weights = f"weights/m{index}.safetensors"
        assert (root / weights).read_bytes() == (active.root / weights).read_bytes()


def with_depth(root: Path, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> ModelPackage:
    """A package that carries a depth checkpoint of its own, as the served one does."""
    return build_package(
        root,
        {f"m{index}/head.npz": b"head" for index in range(FOLDS)}
        | {f"weights/m{index}.safetensors": f"weights {index}".encode() for index in range(FOLDS)}
        | {"depth/model.safetensors": b"depth weights"},
        members=[
            member_data(
                backbone="dinov2_vitb14",
                runtime="torch",
                head=f"m{index}/head.npz",
                weights=f"weights/m{index}.safetensors",
                weight=1.0 / FOLDS,
            )
            for index in range(FOLDS)
        ],
        depth={"model": "depth", "rows": rows, "cols": cols},
    )


@pytest.mark.ml
def test_the_depth_checkpoint_comes_from_the_package(
    prepared: tuple[ModelPackage, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker of the GPU machine has no way out to the hub, and needs none: it is in there."""
    _package, data, folds = prepared
    active = with_depth(Path("with-depth"))
    asked: list[object] = []

    def grids(samples: Sequence[Any], _device: str, source: object = None) -> Any:
        asked.append(source)
        return np.zeros((len(samples), GRID_ROWS * GRID_COLS), dtype=np.float32)

    monkeypatch.setattr(refit, "depth_grids", grids)
    monkeypatch.setattr(refit, "_features", plain_features())
    monkeypatch.setattr(refit, "_out_of_fold", lambda computed, labels, numbers, n: labels.copy())

    refit_heads(
        active=active,
        dataset=data,
        folds_file=folds,
        cache_dir=Path("cache"),
        out_dir=Path("candidate"),
        version="candidate",
    )

    assert asked == [active.root / "depth"]


@pytest.mark.ml
def test_a_depth_grid_of_another_shape_is_refused(
    prepared: tuple[ModelPackage, Path, Path],
) -> None:
    """Heads fitted on one width and fed another would fail late, in the service."""
    _package, data, folds = prepared
    active = with_depth(Path("odd-depth"), rows=GRID_ROWS + 1)

    with pytest.raises(ModelPackageError, match="depth grid"):
        refit_heads(
            active=active,
            dataset=data,
            folds_file=folds,
            cache_dir=Path("cache"),
            out_dir=Path("candidate"),
            version="candidate",
        )
