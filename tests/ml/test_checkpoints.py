"""A package travels without its public encoder weights and gets exactly them back."""

from pathlib import Path

import pytest

from loadvision.ml.packaging.checkpoints import fetch_checkpoints, place_checkpoint
from loadvision.ml.packaging.model_package import ModelPackage, ModelPackageError
from tests.model_packages import (
    DINOV2,
    HEAD_BYTES,
    WEIGHTS,
    Hub,
    build_numpy_package,
    build_package,
    member_data,
    path_of,
    travelling_package,
)


@pytest.mark.ml
def test_missing_checkpoints_are_downloaded_once_each_and_the_package_verifies(
    tmp_path: Path,
) -> None:
    root = travelling_package(tmp_path / "package")
    hub = Hub(tmp_path, WEIGHTS)

    fetched = fetch_checkpoints(root, hub)

    assert sorted(fetched) == sorted(path_of(name) for name in WEIGHTS)
    assert sorted(hub.asked) == sorted(WEIGHTS)
    assert ModelPackage.load(root).manifest.version == "baseline-v0"


@pytest.mark.ml
def test_nothing_is_downloaded_when_every_checkpoint_is_in_place(tmp_path: Path) -> None:
    root = travelling_package(tmp_path / "package")
    fetch_checkpoints(root, Hub(tmp_path, WEIGHTS))
    hub = Hub(tmp_path, WEIGHTS)

    assert fetch_checkpoints(root, hub) == []
    assert hub.asked == []


@pytest.mark.ml
def test_a_damaged_checkpoint_is_downloaded_again(tmp_path: Path) -> None:
    root = travelling_package(tmp_path / "package")
    fetch_checkpoints(root, Hub(tmp_path, WEIGHTS))
    (root / path_of(DINOV2)).write_bytes(b"half a download")

    fetched = fetch_checkpoints(root, Hub(tmp_path, WEIGHTS))

    assert fetched == [path_of(DINOV2)]
    assert (root / path_of(DINOV2)).read_bytes() == WEIGHTS[DINOV2]


@pytest.mark.ml
def test_a_checkpoint_changed_on_the_hub_is_refused_and_not_kept(tmp_path: Path) -> None:
    root = travelling_package(tmp_path / "package")
    hub = Hub(tmp_path, WEIGHTS | {DINOV2: b"retrained upstream"})

    with pytest.raises(ModelPackageError, match=f"timm/{DINOV2} now has sha256"):
        fetch_checkpoints(root, hub)

    assert not (root / path_of(DINOV2)).exists()


@pytest.mark.ml
def test_a_member_of_an_unknown_encoder_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "package"
    member = member_data(backbone="retired_encoder", runtime="torch", weights="weights/x.bin")
    build_package(root, {"head.npz": HEAD_BYTES, "weights/x.bin": b"x"}, members=[member])

    with pytest.raises(ModelPackageError, match="unknown backbone 'retired_encoder'"):
        fetch_checkpoints(root, Hub(tmp_path, {}))


@pytest.mark.ml
def test_building_a_package_copies_a_checkpoint_once(tmp_path: Path) -> None:
    hub = Hub(tmp_path, WEIGHTS)

    first = place_checkpoint(tmp_path / "package", DINOV2, hub)
    again = place_checkpoint(tmp_path / "package", DINOV2, hub)

    assert first == again == tmp_path / "package" / path_of(DINOV2)
    assert first.read_bytes() == WEIGHTS[DINOV2]
    assert hub.asked == [DINOV2]


@pytest.mark.ml
def test_a_package_of_numpy_members_needs_no_checkpoint(tmp_path: Path) -> None:
    root = build_numpy_package(tmp_path, load=55.0)
    hub = Hub(tmp_path, {})

    assert fetch_checkpoints(root, hub) == []
    assert hub.asked == []
