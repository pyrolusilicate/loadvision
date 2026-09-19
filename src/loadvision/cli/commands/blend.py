"""CLI command that averages fine-tuned runs into an ensemble and writes its answer file."""

import argparse
import csv
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from loadvision.cli.commands.finetune import SETTINGS_FILE, TEST_RAW_FILE
from loadvision.cli.errors import CommandError
from loadvision.cli.oof import Dataset, calibrated, dataset_of, photo_facts
from loadvision.cli.options import add_data_option
from loadvision.cli.paths import BLENDS_DIR, runs_dir
from loadvision.cli.standalone import run_alone
from loadvision.ml.data.manifest import load_split
from loadvision.ml.data.submission import SUBMISSION_FILE, write_submission
from loadvision.ml.evaluation.blend import Blend, blend, subsets
from loadvision.ml.evaluation.metrics import mae, paired_difference, summary
from loadvision.ml.pipeline.stage import new_run_id
from loadvision.ml.pipeline.stages import DEFAULT_CHAIN, DEFAULT_FOLDS
from loadvision.ml.pipeline.stages.train_folds import OOF_FILE, OofRow, read_oof, write_oof
from loadvision.ml.postprocess import Facts
from loadvision.ml.training.postprocess_fit import fit_chain

METRICS_FILE: Final = "metrics.json"
MEMBERS_FILE: Final = "members.json"
TOP_SHOWN: Final = 10

# Settings that change how fast a run goes but not what it learns; a --full run and an
# out-of-fold run of the same member may differ in them.
_INCIDENTAL: Final = frozenset({"workers"})

# Inputs beyond the photo itself; two runs of one encoder differ by them.
_EXTRA_INPUTS: Final = ("far_view", "depth")

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `blend` command.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser(
        "blend",
        help="average fine-tuned runs into an ensemble and measure it out of fold",
        description=(
            "Averages the out-of-fold predictions of several `finetune` runs photo by photo, "
            "calibrates the mean with the shipped chain fitted fold by fold, and compares it with "
            "every member alone. With --submit it averages the test predictions of the members' "
            "--full runs and writes the ensemble's submission.csv."
        ),
    )
    parser.add_argument("runs", nargs="+", type=Path, help="out-of-fold run folders of finetune")
    add_data_option(parser)
    parser.add_argument(
        "--folds", type=Path, default=DEFAULT_FOLDS, help="folds the runs were measured on"
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="score every combination of the runs and list the best ones",
    )
    parser.add_argument(
        "--reference", type=Path, help="oof.csv to compare the ensemble with, photo for photo"
    )
    parser.add_argument(
        "--submit",
        nargs="*",
        type=Path,
        metavar="FULL_RUN",
        help="write the ensemble's submission.csv; alone it averages the fold models' test "
        "predictions each run keeps, or pass --full run folders of the same members",
    )
    parser.add_argument(
        "--runs-dir", type=Path, help="folder of run folders (default: runs.dir from settings)"
    )
    parser.set_defaults(run=run_blend)


def run_blend(args: argparse.Namespace) -> int:
    """Scores the ensemble of the given runs, or searches the best one, and writes its files.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If a run is missing or the runs do not fit together.
    """
    data = dataset_of(args.data, args.folds)
    runs = {_member_name(folder): _read_run(folder) for folder in args.runs}
    if len(runs) != len(args.runs):
        raise CommandError("two runs have the same encoder and settings; pass each member once")
    facts = photo_facts(data.samples)
    if args.search:
        _search(data, runs, facts)
        return 0

    ensemble = _blend(data, runs)
    final = _final(data, ensemble.raw, facts)
    _report_members(data, runs, facts, final)
    if args.reference is not None:
        _compare_reference(args.reference, ensemble, final)

    folder = runs_dir(args.runs_dir) / BLENDS_DIR / new_run_id(datetime.now(UTC))
    folder.mkdir(parents=True)
    write_oof(folder / OOF_FILE, data.samples, data.numbers, data.labels, ensemble.raw, final)
    metrics = {
        "mae_raw": mae(data.labels, ensemble.raw),
        **summary(data.labels, final, data.numbers),
    }
    _write_json(folder / METRICS_FILE, metrics)
    _write_json(
        folder / MEMBERS_FILE, {name: str(path) for name, path in zip(runs, args.runs, strict=True)}
    )
    if args.submit is not None:
        _submit(args, ensemble, facts, folder)
    _logger.info("%s", folder)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Runs `blend` on its own: `python -m loadvision.cli.commands.blend ...`."""
    return run_alone(add_commands, "blend", argv)


def _read_run(folder: Path) -> Mapping[str, OofRow]:
    path = folder / OOF_FILE
    if not path.is_file():
        raise CommandError(f"{path} is missing; pass the out-of-fold run of finetune")
    return read_oof(path)


def _member_name(folder: Path) -> str:
    settings = _settings(folder)
    extras = "".join(f"+{key}" for key in _EXTRA_INPUTS if settings.get(key))
    return f"{settings['encoder']}/e{settings['epochs']}{extras}"


def _settings(folder: Path) -> dict[str, Any]:
    path = folder / SETTINGS_FILE
    if not path.is_file():
        raise CommandError(f"{path} is missing; {folder} is not a finetune run")
    content: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return content["settings"]


def _blend(data: Dataset, runs: Mapping[str, Mapping[str, OofRow]]) -> Blend:
    try:
        ensemble = blend(runs)
    except ValueError as error:
        raise CommandError(str(error)) from error
    if list(ensemble.ids) != [sample.image_id for sample in data.samples]:
        raise CommandError("the runs do not cover exactly the training photos of this dataset")
    if ensemble.folds.tolist() != list(data.numbers):
        raise CommandError("the runs were measured on other folds than --folds")
    return ensemble


def _final(data: Dataset, raw: NDArray[np.float64], facts: Facts) -> NDArray[np.float64]:
    predicted = calibrated(data, raw, facts)
    return raw if predicted is None else predicted


def _report_members(
    data: Dataset,
    runs: Mapping[str, Mapping[str, OofRow]],
    facts: Facts,
    final: NDArray[np.float64],
) -> None:
    ensemble_errors = np.abs(data.labels - final)
    best_name, best_errors = "", np.empty(0)
    for name, rows in runs.items():
        raw = np.array([rows[sample.image_id].raw for sample in data.samples])
        errors = np.abs(data.labels - _final(data, raw, facts))
        _logger.info("member    %-32s MAE %.3f", name, errors.mean())
        if not best_name or errors.mean() < best_errors.mean():
            best_name, best_errors = name, errors
    _logger.info("ensemble  %-32s MAE %.3f", f"{len(runs)} members", ensemble_errors.mean())
    if len(runs) > 1:
        mean, spread = paired_difference(ensemble_errors, best_errors)
        _logger.info("ensemble against the best member %s: %+.3f ± %.3f", best_name, mean, spread)


def _compare_reference(reference: Path, ensemble: Blend, final: NDArray[np.float64]) -> None:
    theirs = read_oof(reference)
    shared = [index for index, image_id in enumerate(ensemble.ids) if image_id in theirs]
    if not shared:
        raise CommandError(f"{reference} shares no photo with the runs")
    labels = ensemble.labels[shared]
    ours = np.abs(labels - final[shared])
    baseline = np.abs(labels - np.array([theirs[ensemble.ids[i]].predicted for i in shared]))
    mean, spread = paired_difference(ours, baseline)
    _logger.info(
        "against %s on %d photos: %.3f -> %.3f, %+.3f ± %.3f",
        reference,
        len(shared),
        baseline.mean(),
        ours.mean(),
        mean,
        spread,
    )


def _search(data: Dataset, runs: Mapping[str, Mapping[str, OofRow]], facts: Facts) -> None:
    """Scores every combination of the runs; the list is for choosing, not a result.

    Picking the best of many blends on the same folds flatters the winner a little, so its
    number is confirmed on the other splits before it is trusted.
    """
    scored: list[tuple[float, tuple[str, ...]]] = []
    for group in subsets(list(runs)):
        ensemble = _blend(data, {name: runs[name] for name in group})
        scored.append(
            (float(np.abs(data.labels - _final(data, ensemble.raw, facts)).mean()), group)
        )
    scored.sort()
    _logger.info("best of %d combinations:", len(scored))
    for score, group in scored[:TOP_SHOWN]:
        _logger.info("  MAE %.3f   %s", score, " + ".join(group))


def _submit(args: argparse.Namespace, ensemble: Blend, facts: Facts, folder: Path) -> None:
    """Averages the members' test predictions and calibrates them with the ensemble's chain.

    The chain is fitted on the ensemble's out-of-fold predictions: those are made by models
    that never saw the photos, which is how the test photos are predicted too.
    """
    if args.submit:
        by_settings = {_settings_key(path): path for path in args.runs}
        full = {_settings_key(path): path for path in args.submit}
        if set(full) != set(by_settings) or len(full) != len(args.submit):
            raise CommandError(
                "--submit needs exactly one --full run for every member, same settings"
            )
        sources = list(full.values())
    else:
        sources = list(args.runs)
    test = load_split(args.data, "test")
    ids = [sample.image_id for sample in test]
    raw = np.mean([_read_test_raw(path, ids) for path in sources], axis=0)
    chain = [(name, dict(params)) for name, params in DEFAULT_CHAIN]
    fitted = fit_chain(chain, ensemble.raw, ensemble.labels, facts)
    loads = fitted(raw, photo_facts(test))
    write_submission(folder / SUBMISSION_FILE, ids, loads)
    _logger.info(
        "submission: %d photos   min %.1f   mean %.1f   max %.1f",
        len(ids),
        float(loads.min()),
        float(loads.mean()),
        float(loads.max()),
    )


def _settings_key(folder: Path) -> tuple[tuple[str, str], ...]:
    settings = _settings(folder)
    return tuple(
        sorted((key, str(value)) for key, value in settings.items() if key not in _INCIDENTAL)
    )


def _read_test_raw(folder: Path, ids: Sequence[str]) -> NDArray[np.float64]:
    path = folder / TEST_RAW_FILE
    if not path.is_file():
        raise CommandError(
            f"{path} is missing: the run predates keeping test predictions per fold; pass "
            "--full runs of the same members after --submit"
        )
    with path.open(newline="", encoding="utf-8") as file:
        raw = {row["image_id"]: float(row["raw"]) for row in csv.DictReader(file)}
    if set(raw) != set(ids):
        raise CommandError(f"{path} does not cover exactly the test photos")
    return np.array([raw[image_id] for image_id in ids])


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
