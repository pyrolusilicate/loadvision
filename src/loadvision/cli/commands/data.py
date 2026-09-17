"""CLI commands for dataset manifest, deduplication and splits."""

import argparse
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np

from loadvision.cli.dataset import train_samples
from loadvision.cli.errors import CommandError
from loadvision.cli.options import add_data_option
from loadvision.ml.data.splits import (
    DEFAULT_SEED,
    balance_report,
    build_folds,
    check_covers,
    read_folds,
    write_folds,
)
from loadvision.ml.pipeline.stages import DEFAULT_FOLDS

# The near-duplicate search of 16.09 found no pair above the threshold, so scenes are exactly
# the organizers' groups. The numbers are recorded so a later run can be compared with it.
DEDUP: Final = {
    "encoder": "siglip2_vitb16",
    "metric": "cosine",
    "threshold": 0.98,
    "pairs_found": 0,
    "max_similarity": 0.964,
}

_SHOWN_MOVED: Final = 3

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `data` group and its commands.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser("data", help="dataset splits")
    commands = parser.add_subparsers(title="commands", required=True)

    folds = commands.add_parser(
        "folds",
        help="build or check the canonical cross-validation folds",
        description=(
            "Writes the folds file every model comparison and the promotion gate read. The file "
            "is committed, so a rebuilt split shows up as a reviewable diff instead of silently "
            "changing what the metrics mean."
        ),
    )
    folds.add_argument(
        "--check", action="store_true", help="validate the existing file instead of writing"
    )
    folds.add_argument(
        "--out", type=Path, default=DEFAULT_FOLDS, help="folds file (default: %(default)s)"
    )
    folds.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="split seed (default: %(default)s)"
    )
    add_data_option(folds)
    folds.set_defaults(run=make_folds)


def make_folds(args: argparse.Namespace) -> int:
    """Builds the folds file, or checks the existing one, and reports how balanced it is.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If rebuilding would move photos to another fold; stored folds never
            change, because every comparison of model versions rests on them.
            The folds also have to cover the dataset and stay balanced.
        FileNotFoundError: If a dataset file or, with `--check`, the folds file is missing.
    """
    if not args.check:
        samples = train_samples(args.data)
        try:
            folds = build_folds(
                samples, now=datetime.now(UTC).replace(microsecond=0), dedup=DEDUP, seed=args.seed
            )
        except ValueError as error:
            raise CommandError(f"cannot build balanced folds: {error}") from error
        if args.out.exists():
            previous = read_folds(args.out)
            moved = [
                image_id
                for image_id in previous.images
                if image_id in folds.images
                and previous.fold_of(image_id) != folds.fold_of(image_id)
            ]
            if moved:
                raise CommandError(
                    f"{args.out}: rebuilding would move {len(moved)} photos to another fold, "
                    f"for example {moved[:_SHOWN_MOVED]}. Stored folds never change; delete the "
                    f"file deliberately if that is really what you want."
                )
        write_folds(args.out, folds)
        _logger.info("written %s", args.out)

    _report(args.out, args.data)
    return 0


def _report(folds_path: Path, data_dir: Path) -> None:
    folds = read_folds(folds_path)
    samples = train_samples(data_dir)
    try:
        check_covers(folds, samples)
    except ValueError as error:
        raise CommandError(f"{folds_path}: {error}") from error

    numbers = folds.fold_indices(samples)
    labels = np.array([s.label for s in samples], dtype=np.float64)
    scenes_per_fold = Counter(fold for fold in folds.scene_groups.values())

    _logger.info("%s", folds_path)
    _logger.info("fold  photos  scenes  label=100  mae of the mean")
    for fold in range(folds.n_folds):
        picked = [i for i, value in enumerate(numbers) if value == fold]
        inside = labels[picked]
        _logger.info(
            "%4d  %6d  %6d  %8.1f%%  %15.2f",
            fold,
            len(picked),
            scenes_per_fold[fold],
            100 * float((inside == 100).mean()),
            float(np.abs(inside - inside.mean()).mean()),
        )
    gaps = balance_report(folds, samples)
    _logger.info(
        "fold_size_gap %.4f   bin_share_gap %.4f   (limit 0.05)",
        gaps["fold_size_gap"],
        gaps["bin_share_gap"],
    )
    # A scene maps to exactly one fold by construction, so the only thing worth logging is size.
    _logger.info("scenes %d, photos %d", len(folds.scene_groups), len(folds.images))
