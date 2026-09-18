import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import pytest

from loadvision.application.ports import Prediction
from loadvision.cli.main import main
from loadvision.domain.assessment import LoadPercent
from tests import image_files
from tests.model_packages import build_numpy_package
from tests.settings_files import use_settings

LOAD = 55.0
VERSION = "baseline-v0"
TEST_PHOTOS = ("img_a", "img_b", "img_c")


@dataclass(frozen=True)
class Workspace:
    data_dir: Path
    models_dir: Path
    runs_dir: Path

    def predict(self, *extra: str) -> int:
        return main(["predict", "submission", "--data", str(self.data_dir), *extra])


def write_photos(split_dir: Path, image_ids: tuple[str, ...], header: str) -> None:
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    for image_id in image_ids:
        (images_dir / f"{image_id}.jpg").write_bytes(image_files.jpeg())
    rows = [header] + [
        image_id if header == "image_id" else f"{image_id},50.0" for image_id in image_ids
    ]
    (split_dir / f"{split_dir.name}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """A package that predicts 55 for every photo and a dataset of real JPEG files."""
    use_settings(
        tmp_path,
        monkeypatch,
        storage={"images_dir": str(tmp_path / "images"), "models_dir": str(tmp_path / "models")},
        runs={"dir": str(tmp_path / "runs")},
        model={"version": VERSION},
    )
    (tmp_path / "models").mkdir()
    build_numpy_package(tmp_path / "served", load=LOAD, version=VERSION).rename(
        tmp_path / "models" / VERSION
    )
    data_dir = tmp_path / "data"
    write_photos(data_dir / "test", TEST_PHOTOS, "image_id")
    write_photos(data_dir / "train", ("img_t",), "image_id,load_pct")
    (data_dir / "train" / "train_groups.csv").write_text(
        "image_id,group_id\nimg_t,grp_1\n", encoding="utf-8"
    )
    return Workspace(data_dir=data_dir, models_dir=tmp_path / "models", runs_dir=tmp_path / "runs")


def rows_of(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.mark.unit
def test_test_split_is_predicted_into_the_answer_file(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    assert workspace.predict() == 0

    path = workspace.runs_dir / "submissions" / f"{VERSION}-test.csv"
    rows = rows_of(path)
    assert [row["image_id"] for row in rows] == list(TEST_PHOTOS)
    assert {row["load_pct"] for row in rows} == {"55.0"}
    assert path.read_text(encoding="utf-8").splitlines()[0] == "image_id,load_pct"
    assert f"package {VERSION}, {len(TEST_PHOTOS)} photos of test" in caplog.text
    assert "ms per photo" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("batch", ["1", "2", "8"])
def test_every_photo_is_predicted_whatever_the_batch(workspace: Workspace, batch: str) -> None:
    assert workspace.predict("--batch-size", batch) == 0
    rows = rows_of(workspace.runs_dir / "submissions" / f"{VERSION}-test.csv")
    assert len(rows) == len(TEST_PHOTOS)


@pytest.mark.unit
def test_train_split_and_a_chosen_file(workspace: Workspace, tmp_path: Path) -> None:
    out = tmp_path / "answer.csv"

    assert workspace.predict("--split", "train", "--out", str(out)) == 0

    assert [row["image_id"] for row in rows_of(out)] == ["img_t"]


@pytest.mark.unit
def test_package_folder_wins_over_the_version_from_the_settings(
    workspace: Workspace, tmp_path: Path
) -> None:
    other = build_numpy_package(tmp_path / "other", load=20.0, version="other-v1")
    out = tmp_path / "answer.csv"

    assert workspace.predict("--package", str(other), "--out", str(out)) == 0

    assert {row["load_pct"] for row in rows_of(out)} == {"20.0"}


@pytest.mark.unit
def test_the_device_reaches_the_package(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    devices: list[str] = []

    class Recording:
        def __init__(self, package: object, device: str = "cpu") -> None:
            devices.append(device)

        def predict(self, images: list[object]) -> list[object]:
            return [Prediction(LoadPercent(LOAD)) for _ in images]

    monkeypatch.setattr("loadvision.cli.commands.predict.PackagePredictor", Recording)

    assert workspace.predict("--device", "mps") == 0

    assert devices == ["mps"]


@pytest.mark.unit
def test_missing_package_is_reported(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    assert workspace.predict("--version", "absent-v9") == 1
    assert "manifest.json" in caplog.text


@pytest.mark.unit
def test_batch_size_below_one_is_refused(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    assert workspace.predict("--batch-size", "0") == 1
    assert "batch size" in caplog.text


@pytest.mark.unit
def test_unreadable_photo_is_reported(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    (workspace.data_dir / "test" / "images" / "img_b.jpg").write_bytes(b"not a photo")

    assert workspace.predict() == 1

    assert "UNSUPPORTED_FORMAT" in caplog.text
    assert not (workspace.runs_dir / "submissions").exists()
