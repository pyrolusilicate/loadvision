import json
from pathlib import Path

import pytest

from loadvision.observability.events import Event
from loadvision.observability.run_recorder import EVENTS_FILE, SUMMARY_FILE, RunRecorder

TS = "2026-09-16T12:00:00Z"


def read_events(run_dir: Path) -> list[dict[str, object]]:
    lines = (run_dir / EVENTS_FILE).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.mark.unit
def test_creates_missing_run_folder(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    RunRecorder(run_dir)
    assert run_dir.is_dir()


@pytest.mark.unit
def test_event_is_one_json_line_with_required_fields_first(tmp_path: Path) -> None:
    RunRecorder(tmp_path).event(
        Event.STAGE_STARTED, ts=TS, level="info", module="tests", run_id="run-1", stage="split"
    )
    [record] = read_events(tmp_path)
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
def test_events_are_appended_in_order(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path)
    recorder.event(Event.RUN_STARTED, ts=TS, level="info", module="tests")
    recorder.event(Event.RUN_FINISHED, ts=TS, level="info", module="tests")
    assert [record["event"] for record in read_events(tmp_path)] == ["run.started", "run.finished"]


@pytest.mark.unit
def test_event_keeps_cyrillic_readable(tmp_path: Path) -> None:
    RunRecorder(tmp_path).event(
        Event.STAGE_FAILED, ts=TS, level="error", module="tests", detail="нет файла"
    )
    assert "нет файла" in (tmp_path / EVENTS_FILE).read_text(encoding="utf-8")


@pytest.mark.unit
def test_event_with_nan_is_rejected_before_writing(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path)
    with pytest.raises(ValueError, match="JSON"):
        recorder.event(Event.STAGE_FINISHED, ts=TS, level="info", module="tests", mae=float("nan"))
    assert not (tmp_path / EVENTS_FILE).exists()


@pytest.mark.unit
def test_summary_is_written_and_replaced(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path)
    recorder.write_summary({"status": "failed", "hint": "Откройте summary.json"})
    path = recorder.write_summary({"status": "succeeded"})
    assert path == tmp_path / SUMMARY_FILE
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "succeeded"}
    assert sorted(item.name for item in tmp_path.iterdir()) == [SUMMARY_FILE]


@pytest.mark.unit
def test_summary_with_nan_keeps_previous_summary(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path)
    path = recorder.write_summary({"status": "succeeded"})
    with pytest.raises(ValueError, match="JSON"):
        recorder.write_summary({"status": "succeeded", "mae": float("inf")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "succeeded"}
    assert sorted(item.name for item in tmp_path.iterdir()) == [SUMMARY_FILE]
