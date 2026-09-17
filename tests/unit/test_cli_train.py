import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from loadvision.cli.main import main
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.observability.run_recorder import EVENTS_FILE, SUMMARY_FILE
from tests.settings_files import use_settings

EXPERIMENT = {"members": [["dinov3_vitb16", 200.0]]}


@dataclass
class RecordingStage:
    """Stage that records the context it ran with and fails when a test says so."""

    name: str = "train_folds"
    version: str = "1"
    result: StageResult = field(default_factory=StageResult)
    error: Exception | None = None
    contexts: list[RunContext] = field(default_factory=list[RunContext])

    def inputs(self, ctx: RunContext) -> dict[str, str]:
        return {}

    def run(self, ctx: RunContext) -> StageResult:
        self.contexts.append(ctx)
        if self.error is not None:
            raise self.error
        return self.result

    def check(self, ctx: RunContext, result: StageResult) -> None:
        return None


@dataclass
class Pipeline:
    """Replacement for `build_pipeline` that records what the command asked for."""

    stage: RecordingStage
    devices: list[str] = field(default_factory=list[str])
    configs: list[dict[str, object]] = field(default_factory=list[dict[str, object]])

    def __call__(self, config: dict[str, object], *, device: str) -> list[RecordingStage]:
        self.devices.append(device)
        self.configs.append(config)
        return [self.stage]


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    built = Pipeline(stage=RecordingStage())
    monkeypatch.setattr("loadvision.cli.commands.train.build_pipeline", built)
    return built


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(EXPERIMENT), encoding="utf-8")
    return path


def run(config: Path, *extra: str) -> int:
    return main(["train", str(config), *extra])


@pytest.mark.unit
def test_run_folder_is_written_and_the_stages_get_the_asked_for_context(
    pipeline: Pipeline, config: Path, tmp_path: Path
) -> None:
    code = run(
        config,
        "--runs-dir",
        str(tmp_path / "runs"),
        "--data",
        str(tmp_path / "dataset"),
        "--device",
        "mps",
        "--seed",
        "7",
    )

    assert code == 0
    [ctx] = pipeline.stage.contexts
    assert pipeline.devices == ["mps"]
    assert pipeline.configs == [EXPERIMENT]
    assert ctx.runs_dir == tmp_path / "runs"
    # The cache lives next to the run folders and outside every one of them.
    assert ctx.cache_dir == tmp_path / "runs" / "cache"
    assert (ctx.data_dir, ctx.seed) == (tmp_path / "dataset", 7)
    assert (ctx.run_dir / SUMMARY_FILE).is_file()
    assert (ctx.run_dir / EVENTS_FILE).is_file()


@pytest.mark.unit
def test_runs_folder_and_cache_come_from_the_settings(
    pipeline: Pipeline, config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_settings(tmp_path, monkeypatch, runs={"dir": str(tmp_path / "elsewhere")})

    assert run(config) == 0

    [ctx] = pipeline.stage.contexts
    assert ctx.runs_dir == tmp_path / "elsewhere"
    assert ctx.cache_dir == tmp_path / "elsewhere" / "cache"


@pytest.mark.unit
def test_cache_folder_can_be_moved(pipeline: Pipeline, config: Path, tmp_path: Path) -> None:
    assert (
        run(
            config,
            "--runs-dir",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "shared-cache"),
        )
        == 0
    )
    [ctx] = pipeline.stage.contexts
    assert ctx.cache_dir == tmp_path / "shared-cache"


@pytest.mark.unit
def test_failed_stage_is_reported_with_its_reason(
    pipeline: Pipeline, config: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    pipeline.stage.error = StageCheckError("train_folds: fold 2 came out empty")

    assert run(config, "--runs-dir", str(tmp_path / "runs")) == 1

    assert "fold 2 came out empty" in caplog.text
    assert "failed" in caplog.text


@pytest.mark.unit
def test_metrics_of_the_evaluation_stage_are_shown(
    pipeline: Pipeline, config: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    pipeline.stage.name = "evaluate"
    pipeline.stage.result = StageResult(
        metrics={"mae": 7.124, "within_10": 0.75, "p95_error": 21.0, "mae_fold_spread": 0.4}
    )

    assert run(config, "--runs-dir", str(tmp_path / "runs")) == 0

    assert "mae 7.124" in caplog.text
    assert "within10 0.750" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "message"),
    [("{,}", "not valid JSON"), ('["members"]', "must hold a JSON object")],
    ids=["broken", "not-an-object"],
)
def test_unreadable_config_is_refused(
    pipeline: Pipeline, tmp_path: Path, caplog: pytest.LogCaptureFixture, text: str, message: str
) -> None:
    config = tmp_path / "experiment.json"
    config.write_text(text, encoding="utf-8")

    assert run(config) == 1

    assert message in caplog.text
    assert pipeline.stage.contexts == []


@pytest.mark.unit
def test_missing_config_is_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert run(tmp_path / "absent.json") == 1
    assert "absent.json" in caplog.text


@pytest.mark.unit
def test_config_without_members_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = tmp_path / "experiment.json"
    config.write_text("{}", encoding="utf-8")

    assert run(config) == 1

    assert "members" in caplog.text
