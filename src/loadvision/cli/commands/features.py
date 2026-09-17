"""CLI commands for embeddings of the frozen encoders."""

import argparse
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from loadvision.cli.dataset import train_samples
from loadvision.cli.options import add_data_option, add_device_option
from loadvision.cli.paths import PROBES_FILE, embeddings_cache, runs_dir
from loadvision.cli.provenance import describe_commit
from loadvision.cli.records import jsonl_sink
from loadvision.ml.data.manifest import load_split
from loadvision.ml.evaluation.probe import knn_mae
from loadvision.ml.models.backbones import load_backbone
from loadvision.ml.registry import BACKBONES
from loadvision.ml.training.embedding_cache import embed_samples, preprocessing_fingerprint

NEIGHBOURS: Final = (1, 5, 10, 20)
_FINGERPRINT_SHOWN: Final = 12

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `features` group and its commands.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser("features", help="embeddings of the frozen encoders")
    commands = parser.add_subparsers(title="commands", required=True)

    probe = commands.add_parser(
        "probe",
        help="rank encoders by a kNN probe on cached embeddings",
        description=(
            "Answers one question: how much load information the frozen embeddings already "
            "carry, without training a head. Every run appends to the probe record, so results "
            "accumulate across days and encoders instead of living in a terminal."
        ),
    )
    probe.add_argument("names", nargs="*", help="registry keys of encoders (default: all)")
    add_device_option(probe, default="mps")
    add_data_option(probe)
    probe.set_defaults(run=probe_encoders)


def probe_encoders(args: argparse.Namespace) -> int:
    """Embeds the dataset with every asked-for encoder and reports the kNN error.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the dataset is broken.
        KeyError: If an encoder is not in the registry.
        FileNotFoundError: If a dataset file is missing.
    """
    runs = runs_dir()
    commit = describe_commit()
    rows: list[dict[str, Any]] = []
    with jsonl_sink(runs / PROBES_FILE) as write:
        for name in args.names or list(BACKBONES):
            row = _probe(name, args, commit, runs)
            write(row)
            rows.append(row)
            _logger.info("%s done in %ss", name, row["seconds"])

    header = f"{'encoder':20s}{'input':10s}{'dim':>5s}" + "".join(
        f"{'k=' + str(k):>8s}" for k in NEIGHBOURS
    )
    _logger.info("\n%s", header)
    for row in sorted(rows, key=lambda item: float(item[f"mae_k{NEIGHBOURS[2]}"])):
        scores = "".join(f"{row[f'mae_k{k}']:8.2f}" for k in NEIGHBOURS)
        _logger.info("%-20s%-10s%5d%s", row["encoder"], row["input"], row["dim"], scores)
    _logger.info("\nappended %d rows to %s", len(rows), runs / PROBES_FILE)
    return 0


def _probe(name: str, args: argparse.Namespace, commit: str, runs: Path) -> dict[str, Any]:
    started = time.perf_counter()
    backbone = load_backbone(name, args.device)
    cache = embeddings_cache(runs)
    train = train_samples(args.data)
    vectors = embed_samples(train, backbone, cache_dir=cache, device=args.device)
    # The test split goes into the cache too: every later command then starts from warm files.
    embed_samples(load_split(args.data, "test"), backbone, cache_dir=cache, device=args.device)

    spec = BACKBONES[name]
    return {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "encoder": name,
        "timm_name": spec.timm_name,
        "input": f"{spec.height}x{spec.width}",
        "patch": spec.patch,
        "dim": backbone.dim,
        "preprocessing": preprocessing_fingerprint(
            backbone.preprocessing, backbone.params, views=backbone.spec.views
        )[:_FINGERPRINT_SHOWN],
        "photos": len(train),
        "device": args.device,
        **knn_mae(vectors, train, NEIGHBOURS),
        "seconds": round(time.perf_counter() - started, 1),
    }
