import argparse
import csv
import json
from pathlib import Path

import pytest

from loadvision.cli.commands import finetune
from loadvision.cli.errors import CommandError
from tests.ml.test_stages import make_dataset, make_folds, toy_loader

PHOTOS = 84
# The calibration splits predictions into 16 bins and needs 5 photos in a bin. With three
# folds a fit sees two thirds of the photos: 84 leave about 3.5 per bin, 240 leave 10.
PHOTOS_TOO_FEW_TO_CALIBRATE = 84
PHOTOS_ENOUGH_TO_CALIBRATE = 240


def run(*argv: str) -> int:
    """Parses and runs the command alone, the way the top-level CLI dispatches it."""
    parser = argparse.ArgumentParser()
    finetune.add_commands(parser.add_subparsers())
    args = parser.parse_args(["finetune", *argv])
    return args.run(args)


def make_workspace(root: Path, photos: int) -> dict[str, Path]:
    data = make_dataset(root / "data", train_photos=photos)
    folds = make_folds(data, root / "folds.json")
    config = root / "toy.json"
    config.write_text(
        json.dumps({"encoder": "toy", "epochs": 15, "lr": 0.01, "head_lr_scale": 1, "workers": 0}),
        encoding="utf-8",
    )
    return {"data": data, "folds": folds, "config": config, "runs": root / "runs"}


@pytest.fixture(autouse=True)
def toy_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loadvision.ml.training.finetune.load_backbone", toy_loader)


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    return make_workspace(tmp_path, PHOTOS)


def arguments(paths: dict[str, Path], *extra: str) -> list[str]:
    return [
        str(paths["config"]),
        "--data",
        str(paths["data"]),
        "--folds",
        str(paths["folds"]),
        "--runs-dir",
        str(paths["runs"]),
        "--device",
        "cpu",
        *extra,
    ]


def run_folder(runs: Path) -> Path:
    [folder] = (runs / "finetune").iterdir()
    return folder


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.mark.unit
def test_a_full_run_writes_calibrated_predictions_in_the_pipeline_format(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path, PHOTOS_ENOUGH_TO_CALIBRATE)

    assert run(*arguments(workspace)) == 0

    folder = run_folder(workspace["runs"])
    rows = read_rows(folder / "oof.csv")
    # The same columns the pipeline writes, so a fine-tuned run compares with a frozen one.
    assert list(rows[0]) == ["image_id", "fold", "label", "raw", "predicted"]
    assert len(rows) == PHOTOS_ENOUGH_TO_CALIBRATE
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert {"mae", "mae_raw", "within_10"} <= set(metrics)
    assert metrics["calibrated"] is True
    assert any(row["raw"] != row["predicted"] for row in rows)
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    assert settings["settings"]["encoder"] == "toy"


@pytest.mark.unit
def test_a_single_fold_writes_only_that_fold_and_skips_calibration(
    workspace: dict[str, Path],
) -> None:
    assert run(*arguments(workspace, "--fold", "0")) == 0

    rows = read_rows(run_folder(workspace["runs"]) / "oof.csv")
    assert {row["fold"] for row in rows} == {"0"}
    # The chain is fitted on the other folds, and they were not run.
    assert all(row["raw"] == row["predicted"] for row in rows)


@pytest.mark.unit
def test_a_reference_run_is_compared_on_the_same_photos(
    workspace: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    run(*arguments(workspace))
    reference = run_folder(workspace["runs"]) / "oof.csv"
    caplog.set_level("INFO")

    assert run(*arguments(workspace, "--reference", str(reference))) == 0

    assert "reference  on 84 shared photos" in caplog.text


@pytest.mark.unit
def test_a_config_with_an_unknown_setting_is_refused(workspace: dict[str, Path]) -> None:
    workspace["config"].write_text(json.dumps({"encoder": "toy", "lr_typo": 1}), encoding="utf-8")

    with pytest.raises(CommandError, match="unknown settings: lr_typo"):
        run(*arguments(workspace))


@pytest.mark.unit
def test_a_calibration_that_cannot_be_fitted_keeps_the_raw_predictions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    workspace = make_workspace(tmp_path, PHOTOS_TOO_FEW_TO_CALIBRATE)

    assert run(*arguments(workspace)) == 0

    folder = run_folder(workspace["runs"])
    rows = read_rows(folder / "oof.csv")
    # The training already paid for survives: raw predictions, flagged as uncalibrated.
    assert len(rows) == PHOTOS_TOO_FEW_TO_CALIBRATE
    assert all(row["raw"] == row["predicted"] for row in rows)
    assert json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["calibrated"] is False
    assert "calibration skipped" in caplog.text


@pytest.mark.unit
def test_full_saves_the_model_and_answers_every_test_photo(workspace: dict[str, Path]) -> None:
    assert run(*arguments(workspace, "--full")) == 0

    folder = run_folder(workspace["runs"])
    assert (folder / "encoder.safetensors").is_file()
    assert (folder / "head.npz").is_file()
    # Raw test predictions at full precision, for averaging members into an ensemble.
    assert len(read_rows(folder / "test_raw.csv")) == 12
    rows = read_rows(folder / "submission.csv")
    assert list(rows[0]) == ["image_id", "load_pct"]
    assert len(rows) == 12
    assert all(0.0 <= float(row["load_pct"]) <= 100.0 for row in rows)


@pytest.mark.unit
def test_full_calibrates_the_test_with_an_out_of_fold_run(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path, PHOTOS_ENOUGH_TO_CALIBRATE)
    run(*arguments(workspace))
    oof = run_folder(workspace["runs"]) / "oof.csv"
    workspace["runs"] = tmp_path / "full_runs"

    assert run(*arguments(workspace, "--full", "--calibrate-from", str(oof))) == 0

    rows = read_rows(run_folder(workspace["runs"]) / "submission.csv")
    # The shipped chain ends in rounding to 5, so a calibrated answer sits on that grid.
    assert all(float(row["load_pct"]) % 5 == 0 for row in rows)


@pytest.mark.unit
def test_calibrating_from_a_partial_run_is_refused(workspace: dict[str, Path]) -> None:
    run(*arguments(workspace, "--fold", "0"))
    partial = run_folder(workspace["runs"]) / "oof.csv"

    with pytest.raises(CommandError, match="every training photo"):
        run(*arguments(workspace, "--full", "--calibrate-from", str(partial)))


@pytest.mark.unit
def test_full_with_a_single_fold_is_refused(workspace: dict[str, Path]) -> None:
    with pytest.raises(CommandError, match="--fold makes no sense"):
        run(*arguments(workspace, "--full", "--fold", "0"))


@pytest.mark.unit
def test_every_fold_keeps_its_model_and_answers_the_test(workspace: dict[str, Path]) -> None:
    assert run(*arguments(workspace)) == 0

    folder = run_folder(workspace["runs"])
    # The fold models are kept, so a run never has to be repeated to get them back.
    for fold in range(3):
        assert (folder / "folds" / f"fold_{fold}" / "encoder.safetensors").is_file()
        assert (folder / "folds" / f"fold_{fold}" / "head.npz").is_file()
    per_fold = read_rows(folder / "test_folds.csv")
    assert len(per_fold) == 12
    assert list(per_fold[0]) == ["image_id", "fold_0", "fold_1", "fold_2"]
    # The run's own test answer is the mean of its fold models, ready to submit or blend.
    mean = read_rows(folder / "test_raw.csv")
    for row, fold_row in zip(mean, per_fold, strict=True):
        folds = [float(fold_row[f"fold_{fold}"]) for fold in range(3)]
        assert float(row["raw"]) == pytest.approx(sum(folds) / 3, abs=1e-3)


@pytest.mark.unit
def test_no_weights_keeps_the_test_answers_but_not_the_models(workspace: dict[str, Path]) -> None:
    assert run(*arguments(workspace, "--no-weights")) == 0

    folder = run_folder(workspace["runs"])
    assert not (folder / "folds").exists()
    assert len(read_rows(folder / "test_raw.csv")) == 12
