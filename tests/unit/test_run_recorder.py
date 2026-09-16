import json
from pathlib import Path

import pytest

from loadvision.observability.events import Event
from loadvision.observability.run_recorder import EVENTS_FILE, SUMMARY_FILE, RunRecorder

TS = "2026-09-16T12:00:00Z"


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "run-1"


def read_events(run_dir: Path) -> list[dict[str, object]]:
    lines = (run_dir / EVENTS_FILE).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.mark.unit
def test_creates_run_folder_with_missing_parents(run_dir: Path) -> None:
    RunRecorder(run_dir)
    assert run_dir.is_dir()


@pytest.mark.unit
def test_refuses_existing_run_folder_and_keeps_its_records(run_dir: Path) -> None:
    RunRecorder(run_dir).event(Event.RUN_STARTED, ts=TS, level="info", module="tests")
    before = (run_dir / EVENTS_FILE).read_bytes()
    with pytest.raises(FileExistsError):
        RunRecorder(run_dir)
    assert (run_dir / EVENTS_FILE).read_bytes() == before


@pytest.mark.unit
def test_event_is_one_json_line_with_required_fields_first(run_dir: Path) -> None:
    RunRecorder(run_dir).event(
        Event.STAGE_STARTED, ts=TS, level="info", module="tests", run_id="run-1", stage="split"
    )
    [record] = read_events(run_dir)
    assert record == {
        "ts": TS,
        "level": "info",
        "event": "stage.started",
        "module": "tests",
        "run_id": "run-1",
        "stage": "split",
    }
    assert list(record)[:4] == ["ts", "level", "event", "module"]


@pytest.mark.unit
def test_events_are_appended_in_order(run_dir: Path) -> None:
    recorder = RunRecorder(run_dir)
    recorder.event(Event.RUN_STARTED, ts=TS, level="info", module="tests")
    recorder.event(Event.RUN_FINISHED, ts=TS, level="info", module="tests")
    assert [record["event"] for record in read_events(run_dir)] == ["run.started", "run.finished"]


@pytest.mark.unit
def test_event_keeps_cyrillic_readable(run_dir: Path) -> None:
    RunRecorder(run_dir).event(
        Event.STAGE_FAILED, ts=TS, level="error", module="tests", detail="нет файла"
    )
    assert "нет файла" in (run_dir / EVENTS_FILE).read_text(encoding="utf-8")


@pytest.mark.unit
def test_event_with_nan_is_rejected_before_writing(run_dir: Path) -> None:
    recorder = RunRecorder(run_dir)
    with pytest.raises(ValueError, match="JSON"):
        recorder.event(Event.STAGE_FINISHED, ts=TS, level="info", module="tests", mae=float("nan"))
    assert not (run_dir / EVENTS_FILE).exists()


@pytest.mark.unit
def test_summary_is_written_and_replaced(run_dir: Path) -> None:
    recorder = RunRecorder(run_dir)
    recorder.write_summary({"status": "failed", "hint": "Откройте summary.json"})
    path = recorder.write_summary({"status": "succeeded"})
    assert path == run_dir / SUMMARY_FILE
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "succeeded"}
    assert sorted(item.name for item in run_dir.iterdir()) == [SUMMARY_FILE]


@pytest.mark.unit
def test_summary_with_nan_keeps_previous_summary(run_dir: Path) -> None:
    recorder = RunRecorder(run_dir)
    path = recorder.write_summary({"status": "succeeded"})
    with pytest.raises(ValueError, match="JSON"):
        recorder.write_summary({"status": "succeeded", "mae": float("inf")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "succeeded"}
    assert sorted(item.name for item in run_dir.iterdir()) == [SUMMARY_FILE]
