"""Stage that turns out-of-fold predictions into the metrics a run is judged by."""

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from loadvision.ml.evaluation.metrics import mae, within_tolerance
from loadvision.ml.evaluation.metrics import summary as overall_summary
from loadvision.ml.evaluation.slices import Photo, assign, describe
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.ml.pipeline.stages.build_manifest import MANIFEST_FILE
from loadvision.ml.pipeline.stages.train_folds import OOF_FILE

METRICS_FILE: Final = "metrics.json"
WORST_FILE: Final = "worst_oof.csv"
CASES_FILE: Final = "cases.json"

EXAMPLES_KEPT: Final = 6

WORST_KEPT: Final = 40
_REQUIRED: Final = ("mae", "within_10", "p95_error", "mae_fold_spread")


@dataclass(frozen=True, slots=True)
class Evaluate:
    """Reads `oof.csv` and writes the metrics of a run, overall and per scenario.

    Kept apart from training on purpose: the same metrics must be computable for a candidate
    model version without refitting anything, which is what the promotion gate needs.

    The scenario table is the part a reviewer reads: an average that improves while a scenario
    gets worse is a reason to stop, and that is only visible when both are reported.

    Attributes:
        worst_kept: How many worst photos to list for a human to look at.
        examples_kept: How many example photos to keep per scenario.
    """

    worst_kept: int = WORST_KEPT
    examples_kept: int = EXAMPLES_KEPT

    @property
    def name(self) -> str:
        """Stage name."""
        return "evaluate"

    @property
    def version(self) -> str:
        """Stage version."""
        return "1"

    def inputs(self, ctx: RunContext) -> Mapping[str, str]:
        """What the result depends on."""
        return {
            "oof": str(ctx.run_dir / OOF_FILE),
            "worst_kept": str(self.worst_kept),
            "examples_kept": str(self.examples_kept),
        }

    def run(self, ctx: RunContext) -> StageResult:
        """Computes the metrics of the run.

        Raises:
            FileNotFoundError: If the training stage left no predictions.
        """
        path = ctx.run_dir / OOF_FILE
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; train_folds did not run")
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        labels = np.array([float(row["label"]) for row in rows])
        predicted = np.array([float(row["predicted"]) for row in rows])
        raw = np.array([float(row["raw"]) for row in rows])
        folds = [int(row["fold"]) for row in rows]

        metrics = overall_summary(labels, predicted, folds)
        metrics["mae_before_postprocess"] = float(np.abs(labels - raw).mean())
        errors = np.abs(labels - predicted)

        cases = _cases(ctx, rows, labels, predicted, errors, self.examples_kept)
        metrics.update({f"mae_case_{name}": case["mae"] for name, case in cases.items()})

        metrics_path = ctx.run_dir / METRICS_FILE
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cases_path = ctx.run_dir / CASES_FILE
        cases_path.write_text(
            json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        worst_path = ctx.run_dir / WORST_FILE
        _write_worst(worst_path, rows, errors, self.worst_kept)
        return StageResult(
            artifacts={"metrics": metrics_path, "cases": cases_path, "worst": worst_path},
            counts={"photos": len(rows), "folds": len(set(folds)), "cases": len(cases)},
            metrics=metrics,
        )

    def check(self, ctx: RunContext, result: StageResult) -> None:
        """Verifies the report is complete and every fold is represented.

        Raises:
            StageCheckError: If a required metric is missing, or a fold has no photos.
        """
        missing = [name for name in _REQUIRED if name not in result.metrics]
        if missing:
            raise StageCheckError(f"{self.name}: metrics missing {', '.join(missing)}")
        per_fold = [name for name in result.metrics if name.startswith("mae_fold")]
        # mae_fold_spread is not a fold, so the count is one less than the names.
        if len(per_fold) - 1 != result.counts["folds"]:
            raise StageCheckError(
                f"{self.name}: {len(per_fold) - 1} folds reported, {result.counts['folds']} present"
            )
        for artifact in ("metrics", "cases"):
            path = result.artifacts.get(artifact)
            if path is None or not path.is_file():
                raise StageCheckError(f"{self.name}: {artifact} was not written")
        if not result.counts["cases"]:
            raise StageCheckError(f"{self.name}: no scenario matched a single photo")


def _write_worst(path: Path, rows: list[dict[str, str]], errors: np.ndarray, kept: int) -> None:
    order = np.argsort(-errors)[:kept]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", "fold", "label", "predicted", "error"))
        for index in order:
            row = rows[int(index)]
            writer.writerow(
                [
                    row["image_id"],
                    row["fold"],
                    row["label"],
                    row["predicted"],
                    f"{errors[int(index)]:.2f}",
                ]
            )


def _cases(
    ctx: RunContext,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    predicted: np.ndarray,
    errors: np.ndarray,
    examples_kept: int,
) -> dict[str, dict[str, Any]]:
    photos = _photos(ctx, rows, labels)
    report: dict[str, dict[str, Any]] = {}
    for name, positions in assign(photos).items():
        picked = np.array(positions)
        ranked = [int(index) for index in picked[np.argsort(-errors[picked])]]
        report[name] = {
            "description": describe(name),
            "photos": len(positions),
            "share": len(positions) / len(rows),
            "mae": mae(labels[picked], predicted[picked]),
            "within_10": within_tolerance(labels[picked], predicted[picked]),
            "bias": float((predicted[picked] - labels[picked]).mean()),
            "worst": _examples(rows, labels, predicted, ranked[:examples_kept]),
            "best": _examples(rows, labels, predicted, ranked[::-1][:examples_kept]),
        }
    return report


def _photos(ctx: RunContext, rows: list[dict[str, str]], labels: np.ndarray) -> list[Photo]:
    path = ctx.run_dir / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; build_manifest did not run")
    with path.open(newline="", encoding="utf-8") as file:
        facts = {row["image_id"]: row for row in csv.DictReader(file)}
    scene_sizes = Counter(row["group_id"] for row in facts.values() if row["group_id"])
    photos: list[Photo] = []
    for row, label in zip(rows, labels, strict=True):
        fact = facts[row["image_id"]]
        photos.append(
            Photo(
                label=float(label),
                brightness=float(fact["brightness"]),
                aspect=float(fact["aspect"]),
                scene_size=scene_sizes[fact["group_id"]],
            )
        )
    return photos


def _examples(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    predicted: np.ndarray,
    positions: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        {
            "image_id": rows[int(index)]["image_id"],
            "label": float(labels[int(index)]),
            "predicted": round(float(predicted[int(index)]), 2),
            "error": round(abs(float(labels[int(index)] - predicted[int(index)])), 2),
        }
        for index in positions
    ]
