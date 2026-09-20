"""CLI command that fine-tunes an encoder and measures it out of fold."""

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from loadvision.cli.errors import CommandError
from loadvision.cli.oof import Dataset, calibrated, dataset_of, photo_facts
from loadvision.cli.options import add_data_option, add_device_option
from loadvision.cli.paths import FINETUNE_DIR, runs_dir
from loadvision.cli.provenance import describe_commit
from loadvision.cli.standalone import run_alone
from loadvision.ml.data.manifest import Sample, load_split
from loadvision.ml.data.submission import SUBMISSION_FILE, write_submission
from loadvision.ml.evaluation.metrics import mae, summary
from loadvision.ml.pipeline.stage import new_run_id
from loadvision.ml.pipeline.stages import DEFAULT_CHAIN, DEFAULT_FOLDS
from loadvision.ml.pipeline.stages.train_folds import OOF_FILE, read_oof, write_oof
from loadvision.ml.training.finetune import (
    FinetuneSettings,
    LoadRegressor,
    fit,
    out_of_fold,
    predict,
    save_model,
)
from loadvision.ml.training.postprocess_fit import fit_chain

SETTINGS_FILE: Final = "settings.json"
METRICS_FILE: Final = "metrics.json"
# Uncalibrated test predictions at full precision: an ensemble averages these across members
# and calibrates once, so the rounded submission is not enough to build it from.
TEST_RAW_FILE: Final = "test_raw.csv"
# Every fold model's own test predictions; `test_raw.csv` of an out-of-fold run is their mean.
TEST_FOLDS_FILE: Final = "test_folds.csv"
# Weights of every fold model, `folds/fold_<n>/`, in the files a model package carries.
FOLDS_DIR: Final = "folds"

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `finetune` command.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser(
        "finetune",
        help="fine-tune an encoder and measure it out of fold",
        description=(
            "Fine-tunes a fresh copy of the encoder on every fold and predicts the fold it never "
            "saw, then applies the shipped calibration chain fitted the same way. The run folder "
            "holds oof.csv in the pipeline's format, so the result compares with a frozen run "
            "photo for photo. Every finished fold is saved at once, so a stopped machine keeps it."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help="fine-tuning settings in JSON, such as configs/finetune/<name>.json",
    )
    add_device_option(parser, default="cuda")
    add_data_option(parser)
    parser.add_argument(
        "--folds",
        type=Path,
        default=DEFAULT_FOLDS,
        help="folds file (default: %(default)s); the seed files show how much the split moves it",
    )
    parser.add_argument(
        "--fold", type=int, help="run this fold only, for a quick check before the full run"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="oof.csv of another run, such as the frozen pipeline, compared on the same photos",
    )
    parser.add_argument(
        "--runs-dir", type=Path, help="folder of run folders (default: runs.dir from settings)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="train on every labelled photo, predict the test split and save the model in the "
        "files a model package carries",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="do not keep the weights of the fold models (about 350 MB each for a ViT-B)",
    )
    parser.add_argument(
        "--calibrate-from",
        type=Path,
        help="oof.csv of an out-of-fold run with the same settings; with --full its calibration "
        "chain corrects the test predictions",
    )
    parser.set_defaults(run=finetune)


def finetune(args: argparse.Namespace) -> int:
    """Fine-tunes out of fold, calibrates, writes the run folder and logs the metrics.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the config is not valid JSON or does not describe valid settings.
        FileNotFoundError: If the dataset or the folds file is missing.
    """
    settings = _settings(args.config)
    if args.full and args.fold is not None:
        raise CommandError("--full trains on every photo; --fold makes no sense with it")
    data = dataset_of(args.data, args.folds)
    run_dir = runs_dir(args.runs_dir) / FINETUNE_DIR / new_run_id(datetime.now(UTC))
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / SETTINGS_FILE,
        {
            "settings": asdict(settings),
            "folds": str(args.folds),
            "fold": args.fold,
            "full": args.full,
            "weights_kept": not args.no_weights,
            "calibrate_from": None if args.calibrate_from is None else str(args.calibrate_from),
            "device": args.device,
            "commit": describe_commit(),
        },
    )
    _logger.info("run %s: %s on %s", run_dir.name, settings.encoder, args.device)
    if args.full:
        return _full(args, settings, data, run_dir)

    test = load_split(args.data, "test")
    fold_tests: dict[int, NDArray[np.float64]] = {}

    def save(fold: int, model: LoadRegressor, predictions: NDArray[np.float64]) -> None:
        # While the fold model is still in memory: keep it, and let it answer the test photos.
        # Their mean over the folds is a submission of its own, so no run is ever repeated to
        # get test predictions or weights back.
        _write_rows(run_dir, data, predictions, predictions)
        if not args.no_weights:
            save_model(model, run_dir / FOLDS_DIR / f"fold_{fold}")
        fold_tests[fold] = predict(
            model,
            test,
            device=args.device,
            batch_size=settings.batch_size,
            workers=settings.workers,
        )
        _write_test_folds(run_dir, test, fold_tests)

    raw = out_of_fold(
        data.samples,
        data.labels,
        data.numbers,
        settings,
        device=args.device,
        only=args.fold,
        on_fold=save,
    )
    done = ~np.isnan(raw)
    predicted = calibrated(data, raw) if done.all() else None
    _write_rows(run_dir, data, raw, raw if predicted is None else predicted)

    numbers = np.asarray(data.numbers)
    final = raw if predicted is None else predicted
    metrics: dict[str, float | bool] = {
        "mae_raw": mae(data.labels[done], raw[done]),
        "calibrated": predicted is not None,
        **summary(data.labels[done], final[done], numbers[done].tolist()),
    }
    _write_json(run_dir / METRICS_FILE, metrics)
    _logger.info(
        "photos %d   MAE before calibration %.3f   after %.3f   within10 %.3f",
        int(done.sum()),
        metrics["mae_raw"],
        metrics["mae"],
        metrics["within_10"],
    )
    if args.reference is not None:
        _compare(args.reference, run_dir / OOF_FILE)
    _logger.info("%s", run_dir)
    return 0


def _full(
    args: argparse.Namespace, settings: FinetuneSettings, data: Dataset, run_dir: Path
) -> int:
    """Trains on every labelled photo, saves the model and writes the platform's answer file."""
    model = fit(data.samples, data.labels, settings, device=args.device)
    encoder, head = save_model(model, run_dir)
    test = load_split(args.data, "test")
    raw = predict(
        model, test, device=args.device, batch_size=settings.batch_size, workers=settings.workers
    )
    _write_test_raw(run_dir / TEST_RAW_FILE, test, raw)
    loads = (
        raw
        if args.calibrate_from is None
        else _calibrate_test(args.calibrate_from, data, test, raw)
    )
    write_submission(run_dir / SUBMISSION_FILE, [sample.image_id for sample in test], loads)
    _logger.info(
        "test %d photos   min %.1f   mean %.1f   max %.1f   calibrated %s",
        len(test),
        float(loads.min()),
        float(loads.mean()),
        float(loads.max()),
        args.calibrate_from is not None,
    )
    _logger.info("model: %s, %s", encoder.name, head.name)
    _logger.info("%s", run_dir)
    return 0


def _write_test_folds(
    run_dir: Path, test: Sequence[Sample], fold_tests: dict[int, NDArray[np.float64]]
) -> None:
    folds = sorted(fold_tests)
    with (run_dir / TEST_FOLDS_FILE).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", *(f"fold_{fold}" for fold in folds)))
        writer.writerows(
            (sample.image_id, *(f"{fold_tests[fold][index]:.4f}" for fold in folds))
            for index, sample in enumerate(test)
        )
    _write_test_raw(
        run_dir / TEST_RAW_FILE, test, np.mean([fold_tests[fold] for fold in folds], axis=0)
    )


def _write_test_raw(path: Path, test: Sequence[Sample], raw: NDArray[np.float64]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", "raw"))
        writer.writerows(
            (sample.image_id, f"{value:.4f}") for sample, value in zip(test, raw, strict=True)
        )


def _calibrate_test(
    oof: Path, data: Dataset, test: Sequence[Sample], raw: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Fits the shipped chain on an out-of-fold run and applies it to the test predictions.

    The chain must learn from predictions of models that never saw the photos, which is what
    an out-of-fold run holds; fitting it on the full model's own training answers would teach
    it the model's memory instead of its errors.

    Raises:
        CommandError: If the run does not cover exactly the training photos of this dataset.
    """
    rows = read_oof(oof)
    by_id = {sample.image_id: sample for sample in data.samples}
    if set(rows) != set(by_id):
        raise CommandError(f"{oof} must hold an out-of-fold prediction for every training photo")
    ids = sorted(rows)
    chain = [(name, dict(params)) for name, params in DEFAULT_CHAIN]
    fitted = fit_chain(
        chain,
        np.array([rows[image_id].raw for image_id in ids]),
        np.array([rows[image_id].label for image_id in ids]),
        photo_facts([by_id[image_id] for image_id in ids]),
    )
    return fitted(raw, photo_facts(test))


def _settings(path: Path) -> FinetuneSettings:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CommandError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise CommandError(f"{path} must hold a JSON object")
    try:
        return FinetuneSettings.from_mapping(raw)
    except (TypeError, ValueError) as error:
        raise CommandError(f"{path}: {error}") from error


def _write_rows(
    run_dir: Path, data: Dataset, raw: NDArray[np.float64], predicted: NDArray[np.float64]
) -> None:
    done = ~np.isnan(raw)
    write_oof(
        run_dir / OOF_FILE,
        [sample for sample, kept in zip(data.samples, done, strict=True) if kept],
        [fold for fold, kept in zip(data.numbers, done, strict=True) if kept],
        data.labels[done],
        raw[done],
        predicted[done],
    )


def _compare(reference: Path, ours: Path) -> None:
    """Logs both runs' MAE on the photos they share, before and after postprocessing."""
    theirs, mine = read_oof(reference), read_oof(ours)
    shared = sorted(set(theirs) & set(mine))
    if not shared:
        raise CommandError(f"{reference} shares no photo with this run")
    for name, rows in (("reference", theirs), ("this run", mine)):
        labels = np.array([rows[image_id].label for image_id in shared])
        raw = np.array([rows[image_id].raw for image_id in shared])
        predicted = np.array([rows[image_id].predicted for image_id in shared])
        _logger.info(
            "%-10s on %d shared photos: before calibration %.3f   after %.3f",
            name,
            len(shared),
            mae(labels, raw),
            mae(labels, predicted),
        )


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """Runs `finetune` on its own: `python -m loadvision.cli.commands.finetune ...`."""
    return run_alone(add_commands, "finetune", argv)


if __name__ == "__main__":
    raise SystemExit(main())
