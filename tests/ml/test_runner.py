import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from loadvision.errors import ERROR_CATALOG, ErrorCode, LoadvisionError
from loadvision.ml.pipeline.runner import RunStatus, RunSummary, StageStatus, TrainingRunner
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.observability.run_recorder import EVENTS_FILE, SUMMARY_FILE
from tests.builders import T0

RUNNER_MODULE = "loadvision.ml.pipeline.runner"


class TickingClock:
    """Clock that moves one second forward on every call."""

    def __init__(self) -> None:
        self._calls = 0

    def __call__(self) -> datetime:
        self._calls += 1
        return T0 + timedelta(seconds=self._calls - 1)


class StepTimer:
    """Monotonic timer that moves 0.25 s forward on every call."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        self._now += 0.25
        return self._now


@dataclass
class ScriptedStage:
    """Stage whose run and check fail or succeed as the test tells them."""

    name: str
    log: list[str] = field(default_factory=list[str])
    result: StageResult = field(default_factory=StageResult)
    run_error: Exception | None = None
    check_error: Exception | None = None
    version: str = "1"
    checked: list[StageResult] = field(default_factory=list[StageResult])

    def inputs(self, ctx: RunContext) -> dict[str, str]:
        return {}

    def run(self, ctx: RunContext) -> StageResult:
        self.log.append(f"{self.name}.run")
        if self.run_error is not None:
            raise self.run_error
        return self.result

    def check(self, ctx: RunContext, result: StageResult) -> None:
        self.log.append(f"{self.name}.check")
        self.checked.append(result)
        if self.check_error is not None:
            raise self.check_error


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        run_id="run-1",
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        config={},
        seed=42,
    )


def run(stages: list[ScriptedStage], ctx: RunContext) -> RunSummary:
    return TrainingRunner(clock=TickingClock(), timer=StepTimer()).run(stages, ctx)


def read_events(ctx: RunContext) -> list[dict[str, object]]:
    lines = (ctx.run_dir / EVENTS_FILE).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def read_summary(ctx: RunContext) -> dict[str, object]:
    return json.loads((ctx.run_dir / SUMMARY_FILE).read_text(encoding="utf-8"))


@pytest.mark.ml
def test_runs_every_stage_in_order_and_checks_each(ctx: RunContext) -> None:
    log: list[str] = []
    result = StageResult(metrics={"mae": 7.5}, counts={"rows": 1200})
    first, second = ScriptedStage("split", log, result), ScriptedStage("train", log)

    summary = run([first, second], ctx)

    assert log == ["split.run", "split.check", "train.run", "train.check"]
    assert first.checked == [result]
    assert summary.status is RunStatus.SUCCEEDED
    assert summary.failed_stage is None
    assert [stage.status for stage in summary.stages] == [StageStatus.SUCCEEDED] * 2
    assert summary.stages[0].result is result


@pytest.mark.ml
def test_checks_postconditions_of_cached_result(ctx: RunContext) -> None:
    cached = ScriptedStage(
        "embed", result=StageResult(cached=True), check_error=StageCheckError("dim 768 != 1024")
    )

    summary = run([cached], ctx)

    assert cached.checked == [cached.result]
    assert summary.status is RunStatus.FAILED
    assert summary.stages[0].error_code is ErrorCode.STAGE_CHECK_FAILED
    assert summary.stages[0].error_detail == "dim 768 != 1024"


@pytest.mark.ml
def test_failed_stage_stops_the_run(ctx: RunContext) -> None:
    log: list[str] = []
    stages = [
        ScriptedStage("split", log),
        ScriptedStage("train", log, check_error=StageCheckError("oof misses 5 rows")),
        ScriptedStage("evaluate", log),
        ScriptedStage("package", log),
    ]

    summary = run(stages, ctx)

    assert log == ["split.run", "split.check", "train.run", "train.check"]
    assert [stage.status for stage in summary.stages] == [
        StageStatus.SUCCEEDED,
        StageStatus.FAILED,
        StageStatus.NOT_RUN,
        StageStatus.NOT_RUN,
    ]
    assert summary.failed_stage is summary.stages[1]
    assert summary.stages[2].duration_ms is None


@pytest.mark.ml
def test_check_is_skipped_when_run_raises(ctx: RunContext) -> None:
    stage = ScriptedStage("split", run_error=StageCheckError("labels out of range"))
    run([stage], ctx)
    assert stage.log == ["split.run"]


@pytest.mark.ml
def test_keeps_code_of_catalog_error(ctx: RunContext) -> None:
    stage = ScriptedStage(
        "predict", run_error=LoadvisionError(ErrorCode.CORRUPTED_IMAGE, "0042.jpg")
    )
    [outcome] = run([stage], ctx).stages
    assert (outcome.error_code, outcome.error_detail) == (ErrorCode.CORRUPTED_IMAGE, "0042.jpg")


@pytest.mark.ml
def test_unexpected_exception_becomes_internal_error_with_its_type(ctx: RunContext) -> None:
    stage = ScriptedStage("train", run_error=RuntimeError("CUDA out of memory"))
    [outcome] = run([stage], ctx).stages
    assert outcome.error_code is ErrorCode.INTERNAL_ERROR
    assert outcome.error_detail == "RuntimeError: CUDA out of memory"


@pytest.mark.ml
def test_summary_file_matches_returned_summary(ctx: RunContext) -> None:
    stages = [
        ScriptedStage(
            "split",
            result=StageResult(
                artifacts={"folds": Path("runs/run-1/folds.json")},
                metrics={"bin_share_gap": 0.02},
                counts={"rows": 1200},
                cached=True,
            ),
        ),
        ScriptedStage("train", check_error=StageCheckError("oof misses 5 rows")),
        ScriptedStage("evaluate"),
    ]

    summary = run(stages, ctx)

    assert read_summary(ctx) == summary.to_json()
    assert read_summary(ctx) == {
        "run_id": "run-1",
        "status": "failed",
        "started_at": "2026-09-16T12:00:00Z",
        "finished_at": "2026-09-16T12:00:05Z",
        "failed_stage": "train",
        "error_code": "STAGE_CHECK_FAILED",
        "hint": ERROR_CATALOG[ErrorCode.STAGE_CHECK_FAILED].hint,
        "stages": [
            {
                "name": "split",
                "version": "1",
                "status": "succeeded",
                "cached": True,
                "duration_ms": 250,
                "counts": {"rows": 1200},
                "metrics": {"bin_share_gap": 0.02},
                "artifacts": {"folds": "runs/run-1/folds.json"},
                "error_code": None,
                "error_detail": None,
            },
            {
                "name": "train",
                "version": "1",
                "status": "failed",
                "cached": False,
                "duration_ms": 250,
                "counts": {},
                "metrics": {},
                "artifacts": {},
                "error_code": "STAGE_CHECK_FAILED",
                "error_detail": "oof misses 5 rows",
            },
            {
                "name": "evaluate",
                "version": "1",
                "status": "not_run",
                "cached": False,
                "duration_ms": None,
                "counts": {},
                "metrics": {},
                "artifacts": {},
                "error_code": None,
                "error_detail": None,
            },
        ],
    }


@pytest.mark.ml
def test_successful_summary_has_no_error(ctx: RunContext) -> None:
    run([ScriptedStage("split")], ctx)
    summary = read_summary(ctx)
    assert (summary["status"], summary["failed_stage"], summary["error_code"], summary["hint"]) == (
        "succeeded",
        None,
        None,
        None,
    )


@pytest.mark.ml
def test_events_mark_run_and_stage_boundaries(ctx: RunContext) -> None:
    stages = [
        ScriptedStage("split", result=StageResult(cached=True)),
        ScriptedStage("train", run_error=RuntimeError("CUDA out of memory")),
        ScriptedStage("evaluate"),
    ]

    summary = run(stages, ctx)
    events = read_events(ctx)

    assert [(event["event"], event.get("stage")) for event in events] == [
        ("run.started", None),
        ("stage.started", "split"),
        ("stage.finished", "split"),
        ("stage.started", "train"),
        ("stage.failed", "train"),
        ("run.finished", None),
    ]
    assert all(event["module"] == RUNNER_MODULE for event in events)
    assert all(event["run_id"] == "run-1" for event in events)
    assert events[0]["ts"] == "2026-09-16T12:00:00Z"
    assert events[-1]["ts"] == "2026-09-16T12:00:05Z"
    assert (events[2]["cached"], events[2]["duration_ms"]) == (True, 250)
    assert (events[-1]["level"], events[-1]["status"]) == ("error", summary.status.value)


@pytest.mark.ml
def test_failure_event_has_code_type_and_traceback(ctx: RunContext) -> None:
    run([ScriptedStage("train", run_error=RuntimeError("CUDA out of memory"))], ctx)
    [failed] = [event for event in read_events(ctx) if event["event"] == "stage.failed"]
    assert failed["level"] == "error"
    assert failed["error_code"] == "INTERNAL_ERROR"
    assert failed["detail"] == "RuntimeError: CUDA out of memory"
    assert failed["exc_type"] == "RuntimeError"
    assert failed["duration_ms"] == 250
    assert isinstance(failed["traceback"], str)
    assert "in run" in failed["traceback"]
    assert failed["traceback"].endswith("RuntimeError: CUDA out of memory\n")


@pytest.mark.ml
def test_refuses_to_reuse_run_folder(ctx: RunContext) -> None:
    run([ScriptedStage("split")], ctx)
    events, summary = read_events(ctx), read_summary(ctx)
    log: list[str] = []

    with pytest.raises(FileExistsError):
        run([ScriptedStage("train", log)], ctx)

    assert log == []
    assert (read_events(ctx), read_summary(ctx)) == (events, summary)


@pytest.mark.ml
@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["split", "Train"], "must match"),
        (["split", "../evil"], "must match"),
        (["split", "train", "split"], "repeat: split"),
    ],
)
def test_rejects_bad_stage_names_before_running(
    ctx: RunContext, names: list[str], message: str
) -> None:
    log: list[str] = []
    with pytest.raises(ValueError, match=message):
        run([ScriptedStage(name, log) for name in names], ctx)
    assert log == []
    assert not ctx.run_dir.exists()
