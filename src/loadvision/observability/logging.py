"""Log output of the service processes: JSON lines for machines, plain lines for people.

A JSON line has the keys of a run record (`ts`, `level`, `event`, `module`), so one search by
an event name finds it in the logs and in `runs/` alike.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Final, Literal

type LogFormat = Literal["json", "console"]

CONSOLE_FORMAT: Final = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Attributes every record has; anything else on a record came from `extra=` and goes into the line.
_RECORD_FIELDS: Final = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None))
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Writes a record as one JSON object: time, level, module, message and the extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Returns the record as a JSON line.

        Args:
            record: Log record, with extra fields given through `extra=`.
        """
        line: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "module": record.name,
            "message": record.getMessage(),
        }
        line.update(
            (key, value) for key, value in vars(record).items() if key not in _RECORD_FIELDS
        )
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        # default=str: an extra field that is not JSON, such as a UUID, is written as text
        # instead of losing the whole line.
        return json.dumps(line, ensure_ascii=False, default=str)


def configure_logging(log_format: LogFormat, level: str) -> None:
    """Sends every log record of the process to standard error in the chosen format.

    Earlier handlers are replaced, so calling it again changes the format instead of doubling
    the lines.

    Args:
        log_format: `json` for machines, `console` for people.
        level: Lowest level written, such as ``INFO``.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if log_format == "json" else logging.Formatter(CONSOLE_FORMAT)
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
