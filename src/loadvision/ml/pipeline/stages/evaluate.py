"""Stage that turns out-of-fold predictions into the metrics a run is judged by."""

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from loadvision.ml.evaluation.metrics import summary
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.ml.pipeline.stages.train_folds import OOF_FILE

METRICS_FILE: Final = "metrics.json"
WORST_FILE: Final = "worst_oof.csv"

WORST_KEPT: Final = 40
_REQUIRED: Final = ("mae", "within_10", "p95_error", "mae_fold_spread")


@dataclass(frozen=True, slots=True)
class Evaluate:
    """Reads `oof.csv` and writes `metrics.json` plus the worst predictions.

    Kept apart from training on purpose: the same metrics must be computable for a candidate
    model version without refitting anything, which is what the promotion gate needs.

    Attributes:
        worst_kept: How many worst photos to list for a human to look at.
    """

    worst_kept: int = WORST_KEPT

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
        return {"oof": str(ctx.run_dir / OOF_FILE), "worst_kept": str(self.worst_kept)}

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

        metrics = summary(labels, predicted, folds)
        metrics["mae_before_postprocess"] = float(np.abs(labels - raw).mean())
        metrics_path = ctx.run_dir / METRICS_FILE
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        worst_path = ctx.run_dir / WORST_FILE
        _write_worst(worst_path, rows, np.abs(labels - predicted), self.worst_kept)
        return StageResult(
            artifacts={"metrics": metrics_path, "worst": worst_path},
            counts={"photos": len(rows), "folds": len(set(folds))},
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
        path = result.artifacts.get("metrics")
        if path is None or not path.is_file():
            raise StageCheckError(f"{self.name}: metrics.json was not written")


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
