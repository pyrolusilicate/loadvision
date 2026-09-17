"""Runs the whole ML pipeline of one experiment and writes a run folder under `runs/`.

One command goes from the dataset to `submission.csv`, checking the postconditions of every
stage and recording what happened in `summary.json` and `run.jsonl`. This is the path the
worker takes when it retrains on confirmed labels, so research and retraining follow the same
rules.

Usage:
    uv run python scripts/run_pipeline.py                     # the shipped configuration
    uv run python scripts/run_pipeline.py <config> --device mps
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from loadvision.ml.pipeline.runner import PipelineRunner, RunStatus
from loadvision.ml.pipeline.stage import RunContext, new_run_id
from loadvision.ml.pipeline.stages import build_pipeline

DEFAULT_CONFIG: Final = Path("configs/experiments/ens4_apache_by_aspect.json")
DEFAULT_RUNS: Final = Path("runs")
DEFAULT_CACHE: Final = Path("runs/cache")
DEFAULT_DATA: Final = Path("data")
DEFAULT_SEED: Final = 0

logger = logging.getLogger("pipeline")


def main() -> int:
    """Runs the pipeline and returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"experiment config in JSON; without it the shipped one, {DEFAULT_CONFIG}",
    )
    parser.add_argument("--device", default="cpu", help="torch device, such as mps or cuda")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--folds",
        type=Path,
        default=None,
        help="folds file to use instead of the one named by the config; for measuring how "
        "much a metric moves when only the split changes",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.folds is not None:
        config["folds"] = str(args.folds)
    now = datetime.now(UTC)
    ctx = RunContext(
        run_id=new_run_id(now),
        runs_dir=args.runs_dir,
        cache_dir=args.cache_dir,
        data_dir=args.data,
        config=config,
        seed=args.seed,
    )
    stages = build_pipeline(config, device=args.device)
    logger.info("run %s: %s", ctx.run_id, ", ".join(stage.name for stage in stages))

    summary = PipelineRunner(clock=lambda: datetime.now(UTC)).run(stages, ctx)
    for stage in summary.stages:
        line = f"  {stage.name:20s} {stage.status.value:10s} {stage.duration_ms or 0:7d} ms"
        if stage.error_detail:
            line += f"  {stage.error_detail}"
        logger.info("%s", line)
    metrics = next(
        (dict(s.result.metrics) for s in summary.stages if s.name == "evaluate" and s.result),
        {},
    )
    if metrics:
        logger.info(
            "mae %.3f   within10 %.3f   p95 %.1f   spread %.2f   before postprocess %.3f",
            metrics["mae"],
            metrics["within_10"],
            metrics["p95_error"],
            metrics["mae_fold_spread"],
            metrics.get("mae_before_postprocess", float("nan")),
        )
    logger.info("%s -> %s", ctx.run_dir, summary.status.value)
    return 0 if summary.status is RunStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
