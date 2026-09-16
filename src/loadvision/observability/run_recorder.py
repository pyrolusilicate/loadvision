"""Records pipeline runs: events and the summary of each run in its folder."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from loadvision.observability.events import Event

type JsonScalar = str | int | float | bool | None
type LogLevel = Literal["debug", "info", "warning", "error"]

EVENTS_FILE: Final = "run.jsonl"
SUMMARY_FILE: Final = "summary.json"


class RunRecorder:
    """Writes `run.jsonl` and `summary.json` into the folder of one run.

    Args:
        run_dir: Folder of the run; created if missing.
    """

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir

    def event(
        self, event: Event, *, ts: str, level: LogLevel, module: str, **fields: JsonScalar
    ) -> None:
        """Appends one event to `run.jsonl` as a single JSON line.

        Args:
            event: Event name from the catalog.
            ts: Event time, already formatted by the caller.
            level: Log level of the event.
            module: Module that emits the event, usually its `__name__`.
            **fields: Other event fields, such as `run_id`, `stage` and `duration_ms`.

        Raises:
            ValueError: If a field value is NaN or infinite.
        """
        record = {"ts": ts, "level": level, "event": event.value, "module": module, **fields}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with (self._run_dir / EVENTS_FILE).open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def write_summary(self, summary: Mapping[str, object]) -> Path:
        """Writes `summary.json` atomically, so a reader never sees a half-written file.

        Args:
            summary: JSON-compatible summary of the run.

        Returns:
            Path of the written file.

        Raises:
            ValueError: If the summary contains NaN or infinite numbers.
        """
        path = self._run_dir / SUMMARY_FILE
        text = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        partial = path.with_name(f"{SUMMARY_FILE}.partial")
        partial.write_text(text + "\n", encoding="utf-8")
        partial.replace(path)
        return path
