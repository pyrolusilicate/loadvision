"""Research records: one JSON object per line, appended so results accumulate across days."""

import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

type Row = Mapping[str, object]


@contextmanager
def jsonl_sink(path: Path) -> Iterator[Callable[[Row], None]]:
    """Opens a record file for appending and yields a function that writes one row.

    Rows are written as they are produced, so results of a long run that dies halfway are kept.

    Args:
        path: Record file; its folder is created if missing.

    Raises:
        OSError: If the file cannot be opened or written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:

        def write(row: Row) -> None:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()

        yield write
