import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from loadvision.cli.main import main
from loadvision.cli.oof import Dataset
from loadvision.cli.paths import (
    ERROR_REPORT_FILE,
    OOF_HEADS_FILE,
    OOF_POSTPROCESS_FILE,
    OOF_PREDICTIONS_FILE,
)
from loadvision.ml.data.manifest import Sample
from loadvision.ml.data.splits import Folds
from loadvision.ml.evaluation.postprocess_search import CHAINS
from tests.settings_files import use_settings

PHOTOS = 200
FOLD_COUNT = 5


def dataset() -> Dataset:
    """Twenty labelled photos in ten scenes, two photos each, spread over five folds."""
    # Up to 90, so adding five in the fake ensemble never runs into the clip at 100.
    labels = np.array([float((index * 13) % 91) for index in range(PHOTOS)])
    samples = [
        Sample(
            image_id=f"img_{index:04d}",
            path=Path(f"img_{index:04d}.jpg"),
            sha256="a" * 64,
            label=float(labels[index]),
            group_id=f"grp_{index // 2}",
        )
        for index in range(PHOTOS)
    ]
    folds = Folds(
        schema_version=1,
        created_at="2026-09-17T12:00:00Z",
        seed=0,
        n_folds=FOLD_COUNT,
        bin_edges=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        grouping="group_id",
        dedup={},
        scene_groups={f"grp_{scene}": scene % FOLD_COUNT for scene in range(PHOTOS // 2)},
        images={sample.image_id: sample.group_id or "" for sample in samples},
    )
    return Dataset(
        samples=samples,
        labels=labels,
        numbers=folds.fold_indices(samples),
        folds=folds,
    )


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runs = tmp_path / "runs"
    use_settings(tmp_path, monkeypatch, runs={"dir": str(runs)})
    data = dataset()
    monkeypatch.setattr("loadvision.cli.commands.eval.dataset_of", lambda _data_dir: data)
    monkeypatch.setattr(
        "loadvision.cli.commands.eval.ensemble_predictions",
        lambda given, members, *, device, runs: given.labels + 5.0,
    )
    monkeypatch.setattr(
        "loadvision.cli.commands.eval.brightness_of",
        lambda samples: np.full(len(samples), 120.0),
    )
    return runs


def rows_of(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.unit
def test_every_candidate_chain_is_measured_and_recorded(
    runs_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    assert main(["eval", "postprocess"]) == 0

    rows = rows_of(runs_dir / OOF_POSTPROCESS_FILE)
    assert [row["chain"] for row in rows] == list(CHAINS)
    assert all("mae" in row and "gain_vs_clip" in row for row in rows)
    # Every prediction is five points too high: clipping alone changes nothing, so the chain
    # that only clips gains nothing either.
    [plain] = [row for row in rows if row["chain"] == "none"]
    assert plain["gain_vs_clip"] == pytest.approx(0.0)
    assert plain["mae"] == pytest.approx(5.0)
    assert rows[0]["members"] == [
        "dinov3_vitb16:200",
        "pe_spatial_vitb16:50",
        "dinov3_vitb16_grid4x3:3000",
        "pe_spatial_vitb16_grid4x3:1000",
    ]
    assert "median16" in caplog.text


@pytest.mark.unit
def test_records_are_appended_not_replaced(runs_dir: Path) -> None:
    assert main(["eval", "postprocess"]) == 0
    assert main(["eval", "postprocess", "--member", "siglip2_vitb16:300"]) == 0

    rows = rows_of(runs_dir / OOF_POSTPROCESS_FILE)
    assert len(rows) == 2 * len(CHAINS)
    assert rows[-1]["members"] == ["siglip2_vitb16:300"]


@pytest.mark.unit
def test_malformed_member_is_refused(runs_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert main(["eval", "postprocess", "--member", "siglip2_vitb16"]) == 1

    assert "encoder:alpha" in caplog.text
    assert not (runs_dir / OOF_POSTPROCESS_FILE).exists()


@pytest.mark.unit
def test_error_report_and_predictions_are_written(
    runs_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    assert main(["eval", "errors"]) == 0

    with (runs_dir / OOF_PREDICTIONS_FILE).open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == PHOTOS
    assert rows[0]["image_id"] == "img_0000"
    assert float(rows[0]["error"]) == pytest.approx(5.0)
    page = (runs_dir / ERROR_REPORT_FILE).read_text(encoding="utf-8")
    assert "Разбор ошибок" in page
    assert "ensemble oof mae 5.000" in caplog.text


@pytest.mark.unit
def test_heads_are_measured_on_every_asked_for_encoder(
    runs_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr("loadvision.cli.commands.eval.load_backbone", lambda name, device: name)
    monkeypatch.setattr(
        "loadvision.cli.commands.eval.embed_samples",
        # Plus one: a photo with load 0 would otherwise get a zero vector, which no encoder
        # produces and which the kNN head cannot normalize.
        lambda samples, backbone, *, cache_dir, device: np.array(
            [[float(sample.label) + 1.0] for sample in samples]
        ),
    )

    assert main(["eval", "heads", "siglip2_vitb16", "--head", "knn", "--k", "3"]) == 0

    rows = rows_of(runs_dir / OOF_HEADS_FILE)
    assert [(row["encoder"], row["head"]) for row in rows] == [("siglip2_vitb16", "knn_k3")]
    assert rows[0]["photos"] == PHOTOS
    assert rows[0]["folds_created_at"] == "2026-09-17T12:00:00Z"
    assert "within10" in caplog.text
