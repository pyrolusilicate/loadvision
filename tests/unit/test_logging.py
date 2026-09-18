import json
import logging
import sys
from collections.abc import Iterator
from uuid import UUID

import pytest

from loadvision.observability.logging import JsonFormatter, LogFormat, configure_logging


@pytest.fixture(autouse=True)
def root_logger() -> Iterator[None]:
    """Gives the root logger back as it was: the tests here replace its handlers and level."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def record(message: str = "request finished", **extra: object) -> logging.LogRecord:
    made = logging.LogRecord("loadvision.api", logging.INFO, "file.py", 1, message, None, None)
    made.created = 1789660108.25
    for key, value in extra.items():
        setattr(made, key, value)
    return made


@pytest.mark.unit
def test_a_json_line_has_the_keys_of_a_run_record() -> None:
    line = json.loads(JsonFormatter().format(record(event="request.finished", status=201)))

    assert line == {
        "ts": "2026-09-17T15:48:28.250000Z",
        "level": "info",
        "module": "loadvision.api",
        "message": "request finished",
        "event": "request.finished",
        "status": 201,
    }


@pytest.mark.unit
def test_a_value_json_cannot_hold_is_written_as_text() -> None:
    line = json.loads(JsonFormatter().format(record(job_id=UUID(int=1))))
    assert line["job_id"] == "00000000-0000-0000-0000-000000000001"


@pytest.mark.unit
def test_an_exception_keeps_its_traceback() -> None:
    try:
        raise RuntimeError("database is down")
    except RuntimeError:
        failed = record()
        failed.exc_info = sys.exc_info()

    line = json.loads(JsonFormatter().format(failed))

    assert "RuntimeError: database is down" in line["exception"]


@pytest.mark.unit
@pytest.mark.parametrize(("log_format", "json_lines"), [("json", True), ("console", False)])
def test_the_chosen_format_reaches_standard_error(
    capsys: pytest.CaptureFixture[str], log_format: LogFormat, json_lines: bool
) -> None:
    configure_logging(log_format, "INFO")
    configure_logging(log_format, "INFO")

    logging.getLogger("loadvision.test").info("hello", extra={"event": "run.started"})

    lines = capsys.readouterr().err.splitlines()
    # Configured twice, written once: the second call replaces the handler.
    assert len(lines) == 1
    if json_lines:
        assert json.loads(lines[0])["event"] == "run.started"
    else:
        assert lines[0].endswith("loadvision.test: hello")


@pytest.mark.unit
def test_records_below_the_level_are_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("json", "WARNING")

    logging.getLogger("loadvision.test").info("quiet")

    assert capsys.readouterr().err == ""
