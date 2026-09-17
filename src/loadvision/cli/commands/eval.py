"""CLI commands for evaluation reports and run comparison."""

import argparse
import csv
import logging
import time
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np

from loadvision.cli.oof import (
    add_member_option,
    dataset_of,
    ensemble_predictions,
    members_of,
)
from loadvision.cli.options import add_data_option, add_device_option
from loadvision.cli.paths import (
    ERROR_REPORT_FILE,
    OOF_HEADS_FILE,
    OOF_POSTPROCESS_FILE,
    OOF_PREDICTIONS_FILE,
    embeddings_cache,
    runs_dir,
)
from loadvision.cli.provenance import describe_commit
from loadvision.cli.records import jsonl_sink
from loadvision.ml.evaluation.cross_validation import out_of_fold
from loadvision.ml.evaluation.error_report import brightness_of, build_report
from loadvision.ml.evaluation.metrics import summary
from loadvision.ml.evaluation.oof_heads import knn_predictor, ridge_predictor
from loadvision.ml.evaluation.postprocess_search import CHAINS, cross_fitted
from loadvision.ml.models.backbones import load_backbone
from loadvision.ml.pipeline.stages import DEFAULT_FOLDS
from loadvision.ml.registry import BACKBONES
from loadvision.ml.training.embedding_cache import embed_samples

# The ensemble the packages ship today; K7 replaces the encoders under a trade-controlled licence.
DEFAULT_MEMBERS: Final = (
    "dinov3_vitb16:200",
    "pe_spatial_vitb16:50",
    "dinov3_vitb16_grid4x3:3000",
    "pe_spatial_vitb16_grid4x3:1000",
)
DEFAULT_ALPHAS: Final = (100.0, 300.0, 1000.0)
DEFAULT_NEIGHBOURS: Final = (5, 10)

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `eval` group and its commands.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser("eval", help="out-of-fold measurements and error reports")
    commands = parser.add_subparsers(title="commands", required=True)

    heads = commands.add_parser(
        "heads",
        help="measure heads on the canonical folds",
        description=(
            "Every model version is scored the same way: five heads, each fitted without one "
            "fold, each predicting the fold it never saw. That is the number to compare "
            "versions by, and the number the promotion gate will use."
        ),
    )
    heads.add_argument("names", nargs="*", help="registry keys of encoders (default: all)")
    heads.add_argument(
        "--head", choices=("ridge", "knn", "all"), default="all", help="which heads to fit"
    )
    heads.add_argument(
        "--alpha", type=float, nargs="*", default=list(DEFAULT_ALPHAS), help="ridge penalties"
    )
    heads.add_argument(
        "--k",
        type=int,
        nargs="*",
        default=list(DEFAULT_NEIGHBOURS),
        help="neighbours of the kNN head",
    )
    add_device_option(heads, default="mps")
    add_data_option(heads)
    heads.set_defaults(run=evaluate_heads)

    postprocess = commands.add_parser(
        "postprocess",
        help="compare postprocessing chains on ensemble predictions",
        description=(
            "Every chain is fitted on four folds and measured on the fifth, so a threshold or a "
            "curve never sees the photos it is judged on."
        ),
    )
    add_member_option(postprocess, DEFAULT_MEMBERS)
    add_device_option(postprocess, default="mps")
    add_data_option(postprocess)
    postprocess.set_defaults(run=evaluate_postprocess)

    errors = commands.add_parser(
        "errors",
        help="explain where the model is wrong",
        description=(
            "Writes one row per photo and an HTML report: which loads, which scenes and which "
            "shooting conditions carry the error."
        ),
    )
    add_member_option(errors, DEFAULT_MEMBERS)
    add_device_option(errors, default="mps")
    add_data_option(errors)
    errors.set_defaults(run=explain_errors)


def evaluate_heads(args: argparse.Namespace) -> int:
    """Measures every asked-for head on every asked-for encoder and appends the rows.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the dataset is broken.
        KeyError: If an encoder is not in the registry.
    """
    runs = runs_dir()
    data = dataset_of(args.data)
    commit = describe_commit()
    heads: list[tuple[str, dict[str, float], Any]] = []
    if args.head in ("ridge", "all"):
        heads += [(f"ridge_a{a:g}", {"alpha": a}, ridge_predictor(a)) for a in args.alpha]
    if args.head in ("knn", "all"):
        heads += [(f"knn_k{k}", {"k": float(k)}, knn_predictor(k)) for k in args.k]

    rows: list[dict[str, Any]] = []
    with jsonl_sink(runs / OOF_HEADS_FILE) as write:
        for encoder in args.names or list(BACKBONES):
            backbone = load_backbone(encoder, args.device)
            features = embed_samples(
                data.samples, backbone, cache_dir=embeddings_cache(runs), device=args.device
            ).astype(np.float64)
            for name, params, fit in heads:
                started = time.perf_counter()
                result = out_of_fold(features, data.labels, data.numbers, fit)
                row = {
                    "at": _now(),
                    "commit": commit,
                    "encoder": encoder,
                    "head": name,
                    "params": params,
                    "folds_file": str(DEFAULT_FOLDS),
                    "folds_created_at": data.folds.created_at,
                    "photos": len(data.samples),
                    **result.metrics,
                    "seconds": round(time.perf_counter() - started, 2),
                }
                write(row)
                rows.append(row)

    _logger.info(
        "\n%-28s%-14s%8s%10s%10s%10s", "encoder", "head", "mae", "within10", "p95", "spread"
    )
    for row in sorted(rows, key=lambda item: float(item["mae"])):
        _logger.info(
            "%-28s%-14s%8.3f%10.3f%10.2f%10.2f",
            row["encoder"],
            row["head"],
            row["mae"],
            row["within_10"],
            row["p95_error"],
            row["mae_fold_spread"],
        )
    _logger.info("\nappended %d rows to %s", len(rows), runs / OOF_HEADS_FILE)
    return 0


def evaluate_postprocess(args: argparse.Namespace) -> int:
    """Compares the candidate chains on the out-of-fold predictions of one ensemble.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If a member or the dataset is malformed.
        KeyError: If an encoder is not in the registry.
    """
    runs = runs_dir()
    data = dataset_of(args.data)
    members = members_of(args, DEFAULT_MEMBERS)
    raw = ensemble_predictions(data, members, device=args.device, runs=runs)
    base = np.abs(data.labels - np.clip(raw, 0.0, 100.0))
    commit = describe_commit()

    rows: list[dict[str, Any]] = []
    with jsonl_sink(runs / OOF_POSTPROCESS_FILE) as write:
        for name, chain in CHAINS.items():
            corrected = cross_fitted(chain, raw, data.labels, data.numbers)
            difference = base - np.abs(data.labels - corrected)
            row = {
                "at": _now(),
                "commit": commit,
                "members": [str(member) for member in members],
                "chain": name,
                "steps": [[step, dict(params)] for step, params in chain],
                "folds_created_at": data.folds.created_at,
                **summary(data.labels, corrected, data.numbers),
                "gain_vs_clip": float(difference.mean()),
                "gain_2se": 2 * float(difference.std(ddof=1) / np.sqrt(difference.size)),
            }
            write(row)
            rows.append(row)

    _logger.info("")
    _logger.info(
        "%-22s%8s%10s%8s%9s%9s%10s", "chain", "mae", "within10", "p95", "spread", "gain", "2se"
    )
    for row in sorted(rows, key=lambda item: float(item["mae"])):
        verdict = "*" if row["gain_vs_clip"] > row["gain_2se"] else " "
        _logger.info(
            "%-22s%8.3f%10.3f%8.1f%9.2f%8.3f%s%9.3f",
            row["chain"],
            row["mae"],
            row["within_10"],
            row["p95_error"],
            row["mae_fold_spread"],
            row["gain_vs_clip"],
            verdict,
            row["gain_2se"],
        )
    _logger.info("\n* - выигрыш больше двух стандартных ошибок")
    return 0


def explain_errors(args: argparse.Namespace) -> int:
    """Writes the per-photo predictions and the HTML report of one ensemble.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If a member or the dataset is malformed.
        KeyError: If an encoder is not in the registry.
        OSError: If a photo cannot be read or a report cannot be written.
    """
    runs = runs_dir()
    data = dataset_of(args.data)
    members = members_of(args, DEFAULT_MEMBERS)
    predicted = ensemble_predictions(data, members, device=args.device, runs=runs)
    light = brightness_of(data.samples)

    predictions_path = runs / OOF_PREDICTIONS_FILE
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink)
        writer.writerow(
            ["image_id", "group_id", "fold", "label", "predicted", "error", "brightness"]
        )
        for item, fold, true, pred, value in zip(
            data.samples, data.numbers, data.labels, predicted, light, strict=True
        ):
            writer.writerow(
                [
                    item.image_id,
                    item.group_id,
                    fold,
                    f"{true:g}",
                    f"{pred:.2f}",
                    f"{pred - true:.2f}",
                    f"{value:.1f}",
                ]
            )

    report_path = runs / ERROR_REPORT_FILE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(data.samples, data.labels, predicted, data.numbers, light, members),
        encoding="utf-8",
    )
    metrics = summary(data.labels, predicted, data.numbers)
    _logger.info("")
    _logger.info("ensemble oof mae %.3f", metrics["mae"])
    _logger.info("written %s and %s", predictions_path, report_path)
    return 0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
