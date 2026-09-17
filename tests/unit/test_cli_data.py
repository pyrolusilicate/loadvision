import json
import logging
from pathlib import Path

import pytest

from loadvision.cli.main import main
from loadvision.ml.data.splits import read_folds
from tests.ml.test_dataset_manifest import write_split

PHOTOS_PER_LABEL = 20
LABELS = (0.0, 25.0, 50.0, 75.0, 100.0)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A dataset with photos spread over labels and groups, so folds can be balanced."""
    rows = []
    groups = []
    for label in LABELS:
        for number in range(PHOTOS_PER_LABEL):
            image_id = f"img_{int(label):03d}_{number:02d}"
            rows.append((image_id, f"{label:g}"))
            groups.append((image_id, f"grp_{int(label):03d}_{number // 2:02d}"))
    root = tmp_path / "data"
    write_split(root, "train", rows, groups=groups)
    return root


def run_folds(out: Path, data_dir: Path, *extra: str) -> int:
    return main(["data", "folds", "--out", str(out), "--data", str(data_dir), *extra])


@pytest.mark.unit
def test_folds_are_written_and_reported(
    tmp_path: Path, data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    out = tmp_path / "folds.json"

    assert run_folds(out, data_dir) == 0

    folds = read_folds(out)
    assert len(folds.images) == PHOTOS_PER_LABEL * len(LABELS)
    assert f"written {out}" in caplog.text
    assert "fold_size_gap" in caplog.text


@pytest.mark.unit
def test_rebuilding_with_the_same_seed_keeps_every_photo_in_its_fold(
    tmp_path: Path, data_dir: Path
) -> None:
    out = tmp_path / "folds.json"
    assert run_folds(out, data_dir) == 0
    before = read_folds(out)

    assert run_folds(out, data_dir) == 0

    after = read_folds(out)
    assert {image: after.fold_of(image) for image in after.images} == {
        image: before.fold_of(image) for image in before.images
    }


@pytest.mark.unit
def test_rebuilding_that_would_move_photos_is_refused(
    tmp_path: Path, data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "folds.json"
    assert run_folds(out, data_dir) == 0
    before = out.read_text(encoding="utf-8")

    assert run_folds(out, data_dir, "--seed", "7") == 1

    assert "would move" in caplog.text
    assert out.read_text(encoding="utf-8") == before


@pytest.mark.unit
def test_check_reports_a_folds_file_that_misses_photos(
    tmp_path: Path, data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "folds.json"
    assert run_folds(out, data_dir) == 0
    stored = json.loads(out.read_text(encoding="utf-8"))
    dropped = sorted(stored["images"])[0]
    del stored["images"][dropped]
    out.write_text(json.dumps(stored), encoding="utf-8")

    assert run_folds(out, data_dir, "--check") == 1

    assert "folds.json" in caplog.text
    assert dropped in caplog.text


@pytest.mark.unit
def test_check_without_a_folds_file_is_reported(
    tmp_path: Path, data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert run_folds(tmp_path / "absent.json", data_dir, "--check") == 1
    assert "absent.json" in caplog.text


@pytest.mark.unit
def test_broken_dataset_is_reported(
    tmp_path: Path, data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (data_dir / "train" / "images" / "img_extra.jpg").write_bytes(b"extra")

    assert run_folds(tmp_path / "folds.json", data_dir) == 1

    assert "missing from the csv" in caplog.text
