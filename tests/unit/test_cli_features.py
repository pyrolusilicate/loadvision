import json
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from loadvision.cli.main import main
from loadvision.cli.paths import PROBES_FILE
from loadvision.ml.data.manifest import Sample
from loadvision.ml.registry import BACKBONES
from tests.ml.test_dataset_manifest import write_split
from tests.settings_files import use_settings

ENCODER = "siglip2_vitb16"


class FakeBackbone:
    """Stands in for a loaded encoder, with the fields the cache fingerprint reads."""

    name = ENCODER
    dim = 2
    preprocessing = BACKBONES[ENCODER].preprocessing
    spec = BACKBONES[ENCODER]
    params: ClassVar[dict[str, object]] = {
        "height": 384,
        "width": 512,
        "mean_r": 0.5,
        "mean_g": 0.5,
        "mean_b": 0.5,
        "std_r": 0.5,
        "std_g": 0.5,
        "std_b": 0.5,
    }


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, list[Path]]:
    """A tiny dataset, a runs folder and encoders replaced by fixed vectors."""
    runs = tmp_path / "runs"
    use_settings(tmp_path, monkeypatch, runs={"dir": str(runs)})
    data_dir = tmp_path / "data"
    write_split(
        data_dir,
        "train",
        [("img_a", "10.0"), ("img_b", "90.0")],
        groups=[("img_a", "grp_1"), ("img_b", "grp_2")],
    )
    write_split(data_dir, "test", [("img_c", "")])
    caches: list[Path] = []

    def embed_samples(
        samples: list[Sample], backbone: FakeBackbone, *, cache_dir: Path, device: str
    ) -> np.ndarray:
        caches.append(cache_dir)
        return np.array([[1.0, float(index)] for index, _ in enumerate(samples)], dtype=np.float32)

    monkeypatch.setattr(
        "loadvision.cli.commands.features.load_backbone", lambda name, device: FakeBackbone()
    )
    monkeypatch.setattr("loadvision.cli.commands.features.embed_samples", embed_samples)
    monkeypatch.setattr(
        "loadvision.cli.commands.features.preprocessing_fingerprint",
        lambda name, params, views=(): "f" * 16,
    )
    return runs, data_dir, caches


@pytest.mark.unit
def test_probe_records_one_row_per_encoder(
    workspace: tuple[Path, Path, list[Path]], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    runs, data_dir, caches = workspace

    assert main(["features", "probe", ENCODER, "--data", str(data_dir), "--device", "cpu"]) == 0

    [row] = [json.loads(line) for line in (runs / PROBES_FILE).read_text().splitlines()]
    assert row["encoder"] == ENCODER
    assert (row["photos"], row["device"], row["dim"]) == (2, "cpu", 2)
    assert row["preprocessing"] == "f" * 12
    assert "mae_k1" in row
    assert ENCODER in caplog.text
    # Both splits go into the same cache folder, the one the pipeline stages use.
    assert caches == [runs / "cache" / "embeddings"] * 2


@pytest.mark.unit
def test_probe_appends_to_the_record(workspace: tuple[Path, Path, list[Path]]) -> None:
    runs, data_dir, _ = workspace
    for _ in range(2):
        assert main(["features", "probe", ENCODER, "--data", str(data_dir)]) == 0

    lines = (runs / PROBES_FILE).read_text().splitlines()
    assert len(lines) == 2


@pytest.mark.unit
def test_probe_reports_an_unknown_encoder(
    workspace: tuple[Path, Path, list[Path]], caplog: pytest.LogCaptureFixture
) -> None:
    _, data_dir, _ = workspace
    with pytest.raises(KeyError):
        main(["features", "probe", "no_such_encoder", "--data", str(data_dir)])
