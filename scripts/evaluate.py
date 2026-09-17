"""Measures heads on the canonical folds and appends the numbers to `runs/oof/history.jsonl`.

Every model version is scored the same way: five heads, each fitted without one fold, each
predicting the fold it never saw. That is the number to compare versions by, and the number the
promotion gate will use.

Usage:
    uv run python scripts/evaluate.py                      # every encoder, every head
    uv run python scripts/evaluate.py siglip2_vitb16       # one encoder
    uv run python scripts/evaluate.py --head ridge --alpha 300
"""

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.data.manifest import load_split
from loadvision.ml.data.splits import read_folds
from loadvision.ml.evaluation.cross_validation import Predictor, out_of_fold
from loadvision.ml.models.backbones import load_backbone
from loadvision.ml.registry import BACKBONES
from loadvision.ml.training.embedding_cache import embed_samples
from loadvision.ml.training.head_trainer import FittedHead

OUTPUT: Final = Path("runs/oof/history.jsonl")
CACHE_DIR: Final = Path("runs/cache/embeddings")
FOLDS: Final = Path("configs/splits/folds.json")

logger = logging.getLogger("evaluate")


def ridge_head(alpha: float) -> Callable[[NDArray[Any], NDArray[np.float64]], Predictor]:
    def fit(features: NDArray[Any], labels: NDArray[np.float64]) -> Predictor:
        return FittedHead.fit(features, labels, alpha=alpha).predict

    return fit


def knn_head(k: int) -> Callable[[NDArray[Any], NDArray[np.float64]], Predictor]:
    def fit(features: NDArray[Any], labels: NDArray[np.float64]) -> Predictor:
        unit = features / np.linalg.norm(features, axis=1, keepdims=True)

        def predict(batch: NDArray[Any]) -> NDArray[np.float64]:
            query = batch / np.linalg.norm(batch, axis=1, keepdims=True)
            nearest = np.argsort(-(query @ unit.T), axis=1)[:, :k]
            return np.asarray(labels[nearest].mean(axis=1), dtype=np.float64)

        return predict

    return fit


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="registry keys, default all")
    parser.add_argument("--head", choices=("ridge", "knn", "all"), default="all")
    parser.add_argument("--alpha", type=float, nargs="*", default=[100.0, 300.0, 1000.0])
    parser.add_argument("--k", type=int, nargs="*", default=[5, 10])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    samples = load_split(args.data, "train")
    folds = read_folds(FOLDS)
    numbers = folds.fold_indices(samples)
    labels = np.array([s.label for s in samples], dtype=np.float64)
    commit = git_commit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    heads: list[tuple[str, dict[str, float], Callable[..., Predictor]]] = []
    if args.head in ("ridge", "all"):
        heads += [(f"ridge_a{a:g}", {"alpha": a}, ridge_head(a)) for a in args.alpha]
    if args.head in ("knn", "all"):
        heads += [(f"knn_k{k}", {"k": float(k)}, knn_head(k)) for k in args.k]

    rows: list[dict[str, Any]] = []
    with OUTPUT.open("a", encoding="utf-8") as sink:
        for encoder in args.names or list(BACKBONES):
            backbone = load_backbone(encoder, args.device)
            features = embed_samples(
                samples, backbone, cache_dir=CACHE_DIR, device=args.device
            ).astype(np.float64)
            for name, params, fit in heads:
                started = time.perf_counter()
                result = out_of_fold(features, labels, numbers, fit)
                row = {
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "commit": commit,
                    "encoder": encoder,
                    "head": name,
                    "params": params,
                    "folds_file": str(FOLDS),
                    "folds_created_at": folds.created_at,
                    "photos": len(samples),
                    **result.metrics,
                    "seconds": round(time.perf_counter() - started, 2),
                }
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)

    header = "\n%-28s%-14s%8s%10s%10s%10s"
    logger.info(header, "encoder", "head", "mae", "within10", "p95", "spread")
    for row in sorted(rows, key=lambda item: float(item["mae"])):
        logger.info(
            "%-28s%-14s%8.3f%10.3f%10.2f%10.2f",
            row["encoder"],
            row["head"],
            row["mae"],
            row["within_10"],
            row["p95_error"],
            row["mae_fold_spread"],
        )
    logger.info("\nappended %d rows to %s", len(rows), OUTPUT)


if __name__ == "__main__":
    main()
