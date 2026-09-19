import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from loadvision.cli.commands import blend
from loadvision.cli.errors import CommandError
from loadvision.ml.data.manifest import load_split
from loadvision.ml.data.splits import read_folds
from loadvision.ml.pipeline.stages.train_folds import write_oof
from tests.ml.test_stages import make_dataset, make_folds

PHOTOS = 240
TEST_PHOTOS = 12


def run(*argv: str) -> int:
    parser = argparse.ArgumentParser()
    blend.add_commands(parser.add_subparsers())
    args = parser.parse_args(["blend", *argv])
    return args.run(args)


def settings_of(encoder: str, epochs: int, workers: int = 8) -> dict[str, dict[str, object]]:
    return {"settings": {"encoder": encoder, "epochs": epochs, "lr": 5e-5, "workers": workers}}


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    data = make_dataset(tmp_path / "data", train_photos=PHOTOS, test_photos=TEST_PHOTOS)
    return {"data": data, "folds": make_folds(data, tmp_path / "folds.json"), "root": tmp_path}


def oof_run(
    workspace: dict[str, Path], encoder: str, seed: int, epochs: int = 10, test: bool = False
) -> Path:
    """An out-of-fold run whose model is the truth plus its own independent noise."""
    samples = load_split(workspace["data"], "train")
    folds = read_folds(workspace["folds"]).fold_indices(samples)
    labels = np.array([sample.label for sample in samples], dtype=np.float64)
    raw = np.clip(labels + np.random.default_rng(seed).normal(0.0, 8.0, labels.size), 0, 100)
    folder = workspace["root"] / f"oof_{encoder}"
    folder.mkdir()
    write_oof(folder / "oof.csv", samples, folds, labels, raw, raw)
    (folder / "settings.json").write_text(json.dumps(settings_of(encoder, epochs)))
    if test:
        write_test_raw(workspace, folder, seed + 100)
    return folder


def write_test_raw(workspace: dict[str, Path], folder: Path, seed: int) -> None:
    test = load_split(workspace["data"], "test")
    values = np.random.default_rng(seed).uniform(0, 100, len(test))
    lines = ["image_id,raw"] + [f"{s.image_id},{v:.4f}" for s, v in zip(test, values, strict=True)]
    (folder / "test_raw.csv").write_text("\n".join(lines) + "\n")


def full_run(workspace: dict[str, Path], encoder: str, seed: int, epochs: int = 10) -> Path:
    test = load_split(workspace["data"], "test")
    folder = workspace["root"] / f"full_{encoder}_e{epochs}"
    folder.mkdir()
    values = np.random.default_rng(seed).uniform(0, 100, len(test))
    lines = ["image_id,raw"] + [f"{s.image_id},{v:.4f}" for s, v in zip(test, values, strict=True)]
    (folder / "test_raw.csv").write_text("\n".join(lines) + "\n")
    # Another number of loader workers is incidental and must not stop the pairing.
    (folder / "settings.json").write_text(json.dumps(settings_of(encoder, epochs, workers=2)))
    return folder


def common(workspace: dict[str, Path], *extra: str) -> list[str]:
    return [
        "--data",
        str(workspace["data"]),
        "--folds",
        str(workspace["folds"]),
        "--runs-dir",
        str(workspace["root"] / "runs"),
        *extra,
    ]


def blend_folder(workspace: dict[str, Path]) -> Path:
    [folder] = (workspace["root"] / "runs" / "blends").iterdir()
    return folder


def mae_of(path: Path, column: str) -> float:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return float(np.mean([abs(float(r["label"]) - float(r[column])) for r in rows]))


@pytest.mark.unit
def test_averaging_models_with_independent_errors_beats_each_of_them(
    workspace: dict[str, Path],
) -> None:
    first, second = oof_run(workspace, "a", seed=1), oof_run(workspace, "b", seed=2)

    assert run(str(first), str(second), *common(workspace)) == 0

    folder = blend_folder(workspace)
    ensemble = mae_of(folder / "oof.csv", "raw")
    assert ensemble < min(mae_of(first / "oof.csv", "raw"), mae_of(second / "oof.csv", "raw"))
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["mae_raw"] == pytest.approx(ensemble, abs=1e-3)
    assert set(json.loads((folder / "members.json").read_text())) == {"a/e10", "b/e10"}


@pytest.mark.unit
def test_search_scores_every_combination(
    workspace: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    runs = [str(oof_run(workspace, name, seed)) for seed, name in enumerate("abc")]
    caplog.set_level("INFO")

    assert run(*runs, *common(workspace, "--search")) == 0

    assert "best of 7 combinations" in caplog.text


@pytest.mark.unit
def test_submit_averages_the_members_and_calibrates_the_test(workspace: dict[str, Path]) -> None:
    runs = [str(oof_run(workspace, "a", 1)), str(oof_run(workspace, "b", 2))]
    full = [str(full_run(workspace, "a", 3)), str(full_run(workspace, "b", 4))]

    assert run(*runs, *common(workspace, "--submit", *full)) == 0

    with (blend_folder(workspace) / "submission.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == TEST_PHOTOS
    # The shipped chain ends in rounding to 5.
    assert all(float(row["load_pct"]) % 5 == 0 for row in rows)


@pytest.mark.unit
def test_a_full_run_of_another_member_is_refused(workspace: dict[str, Path]) -> None:
    runs = [str(oof_run(workspace, "a", 1)), str(oof_run(workspace, "b", 2))]
    # Fifteen epochs is another model than the ten the out-of-fold run measured.
    full = [str(full_run(workspace, "a", 3)), str(full_run(workspace, "b", 4, epochs=15))]

    with pytest.raises(CommandError, match="one --full run for every member"):
        run(*runs, *common(workspace, "--submit", *full))


@pytest.mark.unit
def test_the_same_member_twice_is_refused(workspace: dict[str, Path]) -> None:
    first = oof_run(workspace, "a", 1)

    with pytest.raises(CommandError, match="pass each member once"):
        run(str(first), str(first), *common(workspace))


@pytest.mark.unit
def test_submit_alone_uses_the_fold_models_test_answers(workspace: dict[str, Path]) -> None:
    runs = [str(oof_run(workspace, "a", 1, test=True)), str(oof_run(workspace, "b", 2, test=True))]

    assert run(*runs, *common(workspace, "--submit")) == 0

    with (blend_folder(workspace) / "submission.csv").open(newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == TEST_PHOTOS


@pytest.mark.unit
def test_submit_alone_explains_a_run_without_test_answers(workspace: dict[str, Path]) -> None:
    runs = [str(oof_run(workspace, "a", 1)), str(oof_run(workspace, "b", 2))]

    with pytest.raises(CommandError, match="predates keeping test predictions"):
        run(*runs, *common(workspace, "--submit"))


@pytest.mark.unit
def test_runs_of_one_encoder_with_other_inputs_blend(workspace: dict[str, Path]) -> None:
    plain = oof_run(workspace, "a", 1)
    far = oof_run(workspace, "a_far", 2)
    settings = settings_of("a", 10)
    settings["settings"]["far_view"] = True
    # Same encoder and epochs; only the crop of the far end tells the two members apart.
    (far / "settings.json").write_text(json.dumps(settings))

    assert run(str(plain), str(far), *common(workspace)) == 0
