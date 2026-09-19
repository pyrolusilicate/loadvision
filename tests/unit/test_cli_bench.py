import argparse
import json
from pathlib import Path

import pytest

from loadvision.cli.commands import bench
from loadvision.cli.errors import CommandError
from tests.ml.test_baseline_package import build
from tests.ml.test_stages import make_dataset


def run(*argv: str) -> int:
    parser = argparse.ArgumentParser()
    bench.add_commands(parser.add_subparsers())
    args = parser.parse_args(["bench", *argv])
    return args.run(args)


@pytest.mark.unit
def test_bench_times_alone_and_under_load_and_names_the_hardware(tmp_path: Path) -> None:
    package = build(tmp_path).root
    data = make_dataset(tmp_path / "data", train_photos=21, test_photos=12)

    assert (
        run(
            str(package),
            "--data",
            str(data),
            "--clients",
            "1,3",
            "--warmup",
            "1",
            "--runs-dir",
            str(tmp_path / "runs"),
        )
        == 0
    )

    [report_file] = (tmp_path / "runs" / "bench").iterdir()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["conditions"]["photos"] == 12
    assert report["one_client_ms"]["p50"] <= report["one_client_ms"]["p95"]
    assert set(report["clients"]) == {"1", "3"}
    assert report["clients"]["3"]["photos_per_second"] > 0
    assert report["hardware"]["cpu"]


@pytest.mark.unit
def test_bf16_without_a_gpu_is_refused(tmp_path: Path) -> None:
    package = build(tmp_path).root

    with pytest.raises(CommandError, match="CUDA"):
        run(str(package), "--bf16")
