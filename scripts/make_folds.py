"""Builds or checks the canonical cross-validation folds.

Writes `configs/splits/folds.json`, the file every model comparison and the promotion gate read.
It is committed to git, so a rebuilt split shows up as a reviewable diff instead of silently
changing what the metrics mean.

Usage:
    uv run python scripts/make_folds.py            # build and write
    uv run python scripts/make_folds.py --check    # validate the existing file
    uv run python scripts/make_folds.py --seed 1 --out configs/splits/folds_seed1.json
"""

import argparse
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np

from loadvision.ml.data.manifest import load_split
from loadvision.ml.data.splits import (
    DEFAULT_SEED,
    balance_report,
    build_folds,
    check_covers,
    read_folds,
    write_folds,
)

CANONICAL: Final = Path("configs/splits/folds.json")

# The near-duplicate search of 16.09 found no pair above the threshold, so scenes are exactly
# the organizers' groups. The numbers are recorded so a later run can be compared with it.
DEDUP: Final = {
    "encoder": "siglip2_vitb16",
    "metric": "cosine",
    "threshold": 0.98,
    "pairs_found": 0,
    "max_similarity": 0.964,
}

logger = logging.getLogger("folds")


def report(folds_path: Path, data_dir: Path) -> None:
    folds = read_folds(folds_path)
    samples = load_split(data_dir, "train")
    check_covers(folds, samples)

    numbers = folds.fold_indices(samples)
    labels = np.array([s.label for s in samples], dtype=np.float64)
    scenes_per_fold = Counter(fold for fold in folds.scene_groups.values())

    logger.info("%s", folds_path)
    logger.info("fold  photos  scenes  label=100  mae of the mean")
    for fold in range(folds.n_folds):
        picked = [i for i, value in enumerate(numbers) if value == fold]
        inside = labels[picked]
        logger.info(
            "%4d  %6d  %6d  %8.1f%%  %15.2f",
            fold,
            len(picked),
            scenes_per_fold[fold],
            100 * float((inside == 100).mean()),
            float(np.abs(inside - inside.mean()).mean()),
        )
    gaps = balance_report(folds, samples)
    logger.info(
        "fold_size_gap %.4f   bin_share_gap %.4f   (limit 0.05)",
        gaps["fold_size_gap"],
        gaps["bin_share_gap"],
    )
    # A scene maps to exactly one fold by construction, so the only thing worth logging is size.
    logger.info("scenes %d, photos %d", len(folds.scene_groups), len(folds.images))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate instead of writing")
    parser.add_argument("--out", type=Path, default=CANONICAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.check:
        samples = load_split(args.data, "train")
        folds = build_folds(
            samples, now=datetime.now(UTC).replace(microsecond=0), dedup=DEDUP, seed=args.seed
        )
        if args.out.exists():
            previous = read_folds(args.out)
            moved = [
                image_id
                for image_id in previous.images
                if image_id in folds.images
                and previous.fold_of(image_id) != folds.fold_of(image_id)
            ]
            if moved:
                raise SystemExit(
                    f"{args.out}: rebuilding would move {len(moved)} photos to another fold, "
                    f"for example {moved[:3]}. Stored folds never change; delete the file "
                    f"deliberately if that is really what you want."
                )
        write_folds(args.out, folds)
        logger.info("written %s", args.out)

    report(args.out, args.data)


if __name__ == "__main__":
    main()
