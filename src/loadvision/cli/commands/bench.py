"""CLI command that measures how fast a model package answers, the way the service calls it."""

import argparse
import contextlib
import json
import logging
import os
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from loadvision.cli.errors import CommandError
from loadvision.cli.options import add_data_option
from loadvision.cli.paths import runs_dir
from loadvision.cli.standalone import run_alone
from loadvision.ml.data.manifest import load_split
from loadvision.ml.inference.active_model import ActiveModel
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.model_package import ModelPackage
from loadvision.ml.pipeline.stage import new_run_id
from loadvision.ml.training.embedding_cache import read_image

BENCH_DIR: Final = Path("bench")
_MS: Final = 1000.0

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `bench` command.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser(
        "bench",
        help="measure latency and throughput of a model package",
        description=(
            "Decodes test photos and predicts them through the same one-at-a-time wrapper the "
            "service uses: latency per photo (p50, p95) alone, then requests per second and "
            "latency with several clients at once. Writes the numbers and the hardware to "
            "runs/bench/<run>.json. HTTP, XML and the database are not included."
        ),
    )
    parser.add_argument("package", type=Path, help="model package folder")
    add_data_option(parser)
    parser.add_argument("--device", default="cpu", help="torch device (default: %(default)s)")
    parser.add_argument(
        "--bf16", action="store_true", help="run the encoder in bfloat16 (CUDA only)"
    )
    parser.add_argument(
        "--threads", type=int, help="torch CPU threads (default: torch's own choice)"
    )
    parser.add_argument(
        "--photos", type=int, default=100, help="test photos to time (default: %(default)s)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="untimed predictions first (default: %(default)s)"
    )
    parser.add_argument(
        "--clients",
        default="1,4,8",
        help="parallel clients to simulate, comma-separated (default: %(default)s)",
    )
    parser.add_argument(
        "--runs-dir", type=Path, help="folder of run folders (default: runs.dir from settings)"
    )
    parser.set_defaults(run=run_bench)


def run_bench(args: argparse.Namespace) -> int:
    """Times the package and writes `runs/bench/<run>.json`.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the settings make no sense, the package does not load or there are
            no test photos.
    """
    import torch

    clients = _clients(args.clients)
    if args.bf16 and not args.device.startswith("cuda"):
        raise CommandError("--bf16 needs a CUDA device")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    paths = [sample.path for sample in load_split(args.data, "test")][: args.photos]
    if not paths:
        raise CommandError(f"no test photos in {args.data}")

    loaded = time.perf_counter()
    active = ActiveModel()
    active.activate(PackagePredictor(ModelPackage.load(args.package), device=args.device))
    load_seconds = time.perf_counter() - loaded
    predictor = active.current()
    assert predictor is not None  # activated just above

    def one(path: Path) -> float:
        started = time.perf_counter()
        # The autocast context is per thread, so every client opens its own.
        precision = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.bf16
            else contextlib.nullcontext()
        )
        with precision:
            predictor.predict([read_image(path)])
        return time.perf_counter() - started

    for index in range(args.warmup):
        one(paths[index % len(paths)])

    alone = [one(path) for path in paths]
    _logger.info("one client: %s", _describe(alone))
    loads = {n: _under_load(one, paths, n) for n in clients}
    for n, (latencies, rps) in loads.items():
        _logger.info("%d clients: %s   %.1f photos/s", n, _describe(latencies), rps)

    report = {
        "package": str(args.package),
        "version": predictor.version.value,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "conditions": {
            "device": args.device,
            "bf16": args.bf16,
            "torch_threads": torch.get_num_threads(),
            "photos": len(paths),
            "warmup": args.warmup,
            "path": "decode from file + PackagePredictor behind the service's one-at-a-time lock",
            "not_included": "HTTP, XML parsing and rendering, database",
        },
        "hardware": _hardware(torch),
        "load_seconds": round(load_seconds, 2),
        "one_client_ms": _summary(alone),
        "clients": {
            str(n): {"latency_ms": _summary(latencies), "photos_per_second": round(rps, 2)}
            for n, (latencies, rps) in loads.items()
        },
    }
    folder = runs_dir(args.runs_dir) / BENCH_DIR
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{new_run_id(datetime.now(UTC))}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _logger.info("%s", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Runs `bench` on its own: `python -m loadvision.cli.commands.bench ...`."""
    return run_alone(add_commands, "bench", argv)


def _clients(text: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise CommandError(f"--clients must be numbers, got {text!r}") from error
    if not values or min(values) < 1:
        raise CommandError("--clients needs at least one positive number")
    return values


def _under_load(
    one: Callable[[Path], float], paths: Sequence[Path], clients: int
) -> tuple[list[float], float]:
    """Sends every photo from `clients` threads at once; latency includes waiting for the lock."""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        latencies = list(pool.map(one, paths))
    return latencies, len(paths) / (time.perf_counter() - started)


def _summary(seconds: Sequence[float]) -> dict[str, float]:
    values = np.asarray(seconds) * _MS
    return {
        "p50": round(float(np.percentile(values, 50)), 1),
        "p95": round(float(np.percentile(values, 95)), 1),
        "mean": round(float(values.mean()), 1),
        "max": round(float(values.max()), 1),
    }


def _describe(seconds: Sequence[float]) -> str:
    summary = _summary(seconds)
    return f"p50 {summary['p50']:.0f} ms   p95 {summary['p95']:.0f} ms"


def _hardware(torch: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu": _cpu_name(),
        "cpu_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    with contextlib.suppress(OSError):
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
