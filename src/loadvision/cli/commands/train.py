"""CLI commands for training."""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from loadvision.cli.errors import CommandError
from loadvision.cli.options import add_data_option, add_device_option
from loadvision.cli.paths import cache_dir, runs_dir
from loadvision.ml.pipeline.runner import PipelineRunner, RunStatus
from loadvision.ml.pipeline.stage import RunContext, new_run_id
from loadvision.ml.pipeline.stages import build_pipeline
from loadvision.settings import RUNS_CACHE_SUBDIR

DEFAULT_SEED: Final = 0

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `train` command.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser(
        "train",
        help="run the ML pipeline of one experiment",
        description=(
            "Runs the whole pipeline of one experiment, from the dataset to submission.csv, and "
            "writes a run folder with summary.json and run.jsonl. Retraining on confirmed "
            "labels takes the same path."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help="experiment config in JSON, such as configs/experiments/<name>.json",
    )
    add_device_option(parser, default="cpu")
    add_data_option(parser)
    parser.add_argument(
        "--runs-dir", type=Path, help="folder of run folders (default: runs.dir from settings)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=f"cache kept between runs (default: <runs-dir>/{RUNS_CACHE_SUBDIR})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="seed of the run (default: %(default)s)"
    )
    parser.set_defaults(run=train)


def train(args: argparse.Namespace) -> int:
    """Runs the pipeline of the experiment and logs every stage and the metrics.

    Returns:
        Exit code: 0 if every stage succeeded, 1 otherwise.

    Raises:
        CommandError: If the config is not a JSON object or does not describe a pipeline.
        OSError: If the config cannot be read.
    """
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CommandError(f"{args.config} is not valid JSON: {error}") from error
    if not isinstance(config, dict):
        raise CommandError(f"{args.config} must hold a JSON object")
    try:
        stages = build_pipeline(config, device=args.device)
    except ValueError as error:
        raise CommandError(f"{args.config}: {error}") from error

    runs = runs_dir(args.runs_dir)
    ctx = RunContext(
        run_id=new_run_id(datetime.now(UTC)),
        runs_dir=runs,
        cache_dir=cache_dir(runs) if args.cache_dir is None else args.cache_dir,
        data_dir=args.data,
        config=config,
        seed=args.seed,
    )
    _logger.info("run %s: %s", ctx.run_id, ", ".join(stage.name for stage in stages))

    summary = PipelineRunner(clock=lambda: datetime.now(UTC)).run(stages, ctx)
    for stage in summary.stages:
        line = f"  {stage.name:20s} {stage.status.value:10s} {stage.duration_ms or 0:7d} ms"
        if stage.error_detail:
            line += f"  {stage.error_detail}"
        _logger.info("%s", line)
    metrics = next(
        (dict(s.result.metrics) for s in summary.stages if s.name == "evaluate" and s.result),
        {},
    )
    if metrics:
        _logger.info(
            "mae %.3f   within10 %.3f   p95 %.1f   spread %.2f   before postprocess %.3f",
            metrics["mae"],
            metrics["within_10"],
            metrics["p95_error"],
            metrics["mae_fold_spread"],
            metrics.get("mae_before_postprocess", float("nan")),
        )
    _logger.info("%s -> %s", ctx.run_dir, summary.status.value)
    return 0 if summary.status is RunStatus.SUCCEEDED else 1
