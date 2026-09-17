"""Compares postprocessing chains on out-of-fold predictions of one ensemble.

Every chain is fitted on four folds and measured on the fifth, so a threshold or a curve never
sees the photos it is judged on. Results go to `runs/oof/postprocess.jsonl`.

Usage:
    uv run python scripts/evaluate_postprocess.py
    uv run python scripts/evaluate_postprocess.py --member pe_spatial_vitb16_grid4x3:1000
"""

import argparse
import json
import logging
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.data.manifest import load_split
from loadvision.ml.data.splits import read_folds
from loadvision.ml.evaluation.cross_validation import out_of_fold
from loadvision.ml.evaluation.metrics import summary
from loadvision.ml.models.backbones import load_backbone
from loadvision.ml.training.embedding_cache import embed_samples
from loadvision.ml.training.head_trainer import FittedHead
from loadvision.ml.training.postprocess_fit import fit_chain

CACHE_DIR: Final = Path("runs/cache/embeddings")
FOLDS: Final = Path("configs/splits/folds.json")
OUTPUT: Final = Path("runs/oof/postprocess.jsonl")

DEFAULT_MEMBERS: Final = (
    "dinov3_vitb16:200",
    "pe_spatial_vitb16:50",
    "dinov3_vitb16_grid4x3:3000",
    "pe_spatial_vitb16_grid4x3:1000",
)

type Chain = Sequence[tuple[str, Mapping[str, float]]]

CHAINS: Final[dict[str, Chain]] = {
    "none": [("clip", {})],
    "round5": [("round_to", {"step": 5.0})],
    "expand": [("linear_expand", {})],
    "isotonic": [("isotonic", {})],
    "median16": [("median_bins", {"bins": 16.0})],
    "snap_high": [("snap_high", {})],
    "iso+snap_high": [("isotonic", {}), ("snap_high", {})],
    "iso+snap_both": [("isotonic", {}), ("snap_high", {}), ("snap_low", {})],
    "iso+round5": [("isotonic", {}), ("round_to", {"step": 5.0})],
    "iso+snap_high+round5": [
        ("isotonic", {}),
        ("snap_high", {}),
        ("round_to", {"step": 5.0}),
    ],
    "median16+snap_high": [("median_bins", {"bins": 16.0}), ("snap_high", {})],
    "median16+round5": [("median_bins", {"bins": 16.0}), ("round_to", {"step": 5.0})],
    "median16+snap_high+round5": [
        ("median_bins", {"bins": 16.0}),
        ("snap_high", {}),
        ("round_to", {"step": 5.0}),
    ],
    "iso+round10": [("isotonic", {}), ("round_to", {"step": 10.0})],
    "expand+snap_high": [("linear_expand", {}), ("snap_high", {})],
}

logger = logging.getLogger("postprocess")


def ensemble_oof(members: Sequence[str], device: str, data_dir: Path) -> tuple[Any, ...]:
    samples = load_split(data_dir, "train")
    folds = read_folds(FOLDS)
    numbers = folds.fold_indices(samples)
    labels = np.array([s.label for s in samples], dtype=np.float64)

    parts: list[NDArray[np.float64]] = []
    for member in members:
        encoder, _, alpha = member.partition(":")
        penalty = float(alpha)
        backbone = load_backbone(encoder, device)
        features = embed_samples(samples, backbone, cache_dir=CACHE_DIR, device=device).astype(
            np.float64
        )
        result = out_of_fold(
            features,
            labels,
            numbers,
            lambda a, b, value=penalty: FittedHead.fit(a, b, alpha=value).predict,
        )
        logger.info("%s: oof mae %.3f", member, result.metrics["mae"])
        parts.append(result.predictions)
    return np.mean(parts, axis=0), labels, numbers, folds


def cross_fitted(
    chain: Chain, raw: NDArray[np.float64], labels: NDArray[np.float64], numbers: list[int]
) -> NDArray[np.float64]:
    out = np.empty_like(raw)
    for fold in sorted(set(numbers)):
        validation = np.array([value == fold for value in numbers])
        training = ~validation
        apply = fit_chain(chain, raw[training], labels[training])
        out[validation] = apply(raw[validation])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", action="append", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    members = args.member or list(DEFAULT_MEMBERS)
    raw, labels, numbers, folds = ensemble_oof(members, args.device, args.data)
    base = np.abs(labels - np.clip(raw, 0.0, 100.0))

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with OUTPUT.open("a", encoding="utf-8") as sink:
        for name, chain in CHAINS.items():
            corrected = cross_fitted(chain, raw, labels, numbers)
            metrics = summary(labels, corrected, numbers)
            difference = base - np.abs(labels - corrected)
            error = float(difference.std(ddof=1) / np.sqrt(difference.size))
            row = {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "commit": commit,
                "members": members,
                "chain": name,
                "steps": [[step, dict(params)] for step, params in chain],
                "folds_created_at": folds.created_at,
                **metrics,
                "gain_vs_clip": float(difference.mean()),
                "gain_2se": 2 * error,
            }
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    logger.info("")
    header = "%-22s%8s%10s%8s%9s%9s%10s"
    logger.info(header, "chain", "mae", "within10", "p95", "spread", "gain", "2se")
    for row in sorted(rows, key=lambda item: float(item["mae"])):
        verdict = "*" if row["gain_vs_clip"] > row["gain_2se"] else " "
        logger.info(
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
    logger.info("\n* - выигрыш больше двух стандартных ошибок")


if __name__ == "__main__":
    main()
