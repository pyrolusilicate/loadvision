"""Stage protocol and stage result types."""

import math
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, Self

from loadvision.domain.timestamps import ensure_aware
from loadvision.errors import ErrorCode, LoadvisionError

# Run ids and stage names become folder names and CLI arguments, so neither may start with a dot
# or contain a path separator.
RUN_ID_PATTERN: Final = "[A-Za-z0-9][A-Za-z0-9._-]{0,99}"
STAGE_NAME_PATTERN: Final = "[a-z][a-z0-9_]{0,63}"

_RUN_ID_RE: Final = re.compile(RUN_ID_PATTERN)


def new_run_id(now: datetime) -> str:
    """Makes an id for a new run that sorts by start time, such as ``run_20260918_140155_a3f9c2``.

    Every run, a restart after a failure included, takes a new id.

    Args:
        now: Start time of the run, timezone-aware; written in UTC.

    Returns:
        Id matching `RUN_ID_PATTERN`; the random tail separates runs started in the same second.

    Raises:
        ValueError: If `now` has no UTC offset.
    """
    ensure_aware(now, "now")
    return f"run_{now.astimezone(UTC):%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"


@dataclass(frozen=True, slots=True, kw_only=True)
class RunContext:
    """What a stage knows about the current run.

    Attributes:
        run_id: Unique id of the run; also the name of its folder.
        runs_dir: Folder that holds the folders of all runs.
        cache_dir: Cache kept between runs, such as embeddings; outside every run folder.
        data_dir: Dataset root with `train/` and `test/`.
        config: Resolved experiment config.
        seed: Seed for everything random in the run.
    """

    run_id: str
    runs_dir: Path
    cache_dir: Path
    data_dir: Path
    config: Mapping[str, object]
    seed: int

    def __post_init__(self) -> None:
        """Validates the run id.

        Raises:
            ValueError: If `run_id` does not match `RUN_ID_PATTERN`.
        """
        if _RUN_ID_RE.fullmatch(self.run_id) is None:
            raise ValueError(f"run id must match {RUN_ID_PATTERN}")

    @property
    def run_dir(self) -> Path:
        """Folder of this run, `<runs_dir>/<run_id>`."""
        return self.runs_dir / self.run_id


@dataclass(frozen=True, slots=True, kw_only=True)
class StageResult:
    """What a stage reports to the runner and to `summary.json`.

    Attributes:
        artifacts: Files the stage produced, by name.
        metrics: Numbers worth comparing between runs, by name.
        counts: Row or item counts, by name.
        cached: Whether the stage reused earlier results instead of computing them.
    """

    artifacts: Mapping[str, Path] = field(default_factory=dict[str, Path])
    metrics: Mapping[str, float] = field(default_factory=dict[str, float])
    counts: Mapping[str, int] = field(default_factory=dict[str, int])
    cached: bool = False

    def __post_init__(self) -> None:
        """Validates the metrics.

        Raises:
            ValueError: If a metric is NaN or infinite; `summary.json` must stay valid JSON.
        """
        for name, value in self.metrics.items():
            if not math.isfinite(value):
                raise ValueError(f"metric {name} must be finite, got {value!r}")


class StageCheckError(LoadvisionError):
    """A stage postcondition does not hold; the run stops at this stage.

    Args:
        detail: Which postcondition failed, for `summary.json` and logs.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.STAGE_CHECK_FAILED, detail)

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...]]:
        return (type(self), (self.detail,))


class Stage(Protocol):
    """One step of a pipeline, run by `TrainingRunner`."""

    @property
    def name(self) -> str:
        """Name matching `STAGE_NAME_PATTERN`; unique within a pipeline."""
        ...

    @property
    def version(self) -> str:
        """Version of the stage logic; a new version invalidates cached results."""
        ...

    def inputs(self, ctx: RunContext) -> Mapping[str, str]:
        """Returns what the result depends on, for the cache key: file hashes, config values."""
        ...

    def run(self, ctx: RunContext) -> StageResult:
        """Computes the stage or reuses cached results."""
        ...

    def check(self, ctx: RunContext, result: StageResult) -> None:
        """Verifies the postconditions of the result.

        Raises:
            StageCheckError: If a postcondition does not hold.
        """
        ...
