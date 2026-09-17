"""Ranks registry encoders by a kNN probe on cached embeddings, without training a head.

The probe answers one question: how much load information the frozen embeddings already carry.
For every training photo it averages the labels of its nearest neighbours, skipping its own
group, and reports the mean absolute error. Every run appends to `runs/probes/encoders.jsonl`,
so results accumulate across days and encoders instead of living in a terminal.

Usage:
    uv run python scripts/probe_encoders.py [name ...] [--device mps] [--data data]
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.data.manifest import Sample, load_split
from loadvision.ml.models.backbones import load_backbone
from loadvision.ml.registry import BACKBONES
from loadvision.ml.training.embedding_cache import embed_samples, preprocessing_fingerprint

NEIGHBOURS: Final = (1, 5, 10, 20)
OUTPUT: Final = Path("runs/probes/encoders.jsonl")
CACHE_DIR: Final = Path("runs/cache/embeddings")

logger = logging.getLogger("probe")


def knn_mae(vectors: NDArray[np.float32], samples: list[Sample]) -> dict[str, float]:
    """Mean absolute error of averaging the labels of the k nearest photos of other groups."""
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = np.array([s.label for s in samples], dtype=np.float64)
    groups = np.array([s.group_id for s in samples])

    similarity = unit @ unit.T
    # A photo may not vote for itself, and neither may another photo of the same truck: the
    # split rule keeps whole groups together, so the probe must respect it too.
    similarity[groups[:, None] == groups[None, :]] = -1.0
    ranked = np.argsort(-similarity, axis=1)
    return {
        f"mae_k{k}": float(np.abs(labels - labels[ranked[:, :k]].mean(axis=1)).mean())
        for k in NEIGHBOURS
    }


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def probe(name: str, data_dir: Path, device: str, commit: str) -> dict[str, Any]:
    started = time.perf_counter()
    backbone = load_backbone(name, device)
    train = load_split(data_dir, "train")
    vectors = embed_samples(train, backbone, cache_dir=CACHE_DIR, device=device)
    embed_samples(load_split(data_dir, "test"), backbone, cache_dir=CACHE_DIR, device=device)

    spec = BACKBONES[name]
    return {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "encoder": name,
        "timm_name": spec.timm_name,
        "input": f"{spec.height}x{spec.width}",
        "patch": spec.patch,
        "dim": backbone.dim,
        "preprocessing": preprocessing_fingerprint(backbone.params)[:12],
        "photos": len(train),
        "device": device,
        **knn_mae(vectors, train),
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=None, help="registry keys, default all")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    commit = git_commit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with OUTPUT.open("a", encoding="utf-8") as sink:
        for name in args.names or list(BACKBONES):
            row = probe(name, args.data, args.device, commit)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            logger.info("%s done in %ss", name, row["seconds"])

    header = f"{'encoder':20s}{'input':10s}{'dim':>5s}" + "".join(
        f"{'k=' + str(k):>8s}" for k in NEIGHBOURS
    )
    logger.info("\n%s", header)
    for row in sorted(rows, key=lambda r: r[f"mae_k{NEIGHBOURS[2]}"]):
        scores = "".join(f"{row[f'mae_k{k}']:8.2f}" for k in NEIGHBOURS)
        logger.info("%-20s%-10s%5d%s", row["encoder"], row["input"], row["dim"], scores)
    logger.info("\nappended %d rows to %s", len(rows), OUTPUT)


if __name__ == "__main__":
    main()
