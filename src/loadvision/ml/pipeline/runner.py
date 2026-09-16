"""Runs pipeline stages in order with postconditions and run records."""

import re
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from loadvision.domain.timestamps import format_utc
from loadvision.errors import ERROR_CATALOG, ErrorCode, LoadvisionError
from loadvision.ml.pipeline.stage import STAGE_NAME_PATTERN, RunContext, Stage, StageResult
from loadvision.observability.events import Event
from loadvision.observability.run_recorder import RunRecorder

_STAGE_NAME_RE: Final = re.compile(STAGE_NAME_PATTERN)
# Frames kept from the end of a failure traceback in run.jsonl: enough to find the failing
# line inside a stage without pasting the whole call stack into every record.
_TRACEBACK_FRAMES: Final = 5


class StageStatus(StrEnum):
    """Outcome of one stage in a run."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_RUN = "not_run"


class RunStatus(StrEnum):
    """Outcome of a whole run."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class StageOutcome:
    """What happened to one stage.

    Attributes:
        name: Stage name.
        version: Stage version.
        status: Whether the stage succeeded, failed or did not run after an earlier failure.
        result: Result of a succeeded stage.
        duration_ms: Time spent in `run` and `check`; None for a stage that did not run.
        error_code: Catalog code of the failure.
        error_detail: Failure context for logs.
    """

    name: str
    version: str
    status: StageStatus
    result: StageResult | None = None
    duration_ms: int | None = None
    error_code: ErrorCode | None = None
    error_detail: str | None = None

    def to_json(self) -> dict[str, object]:
        """Returns the stage entry of `summary.json`."""
        result = self.result or StageResult()
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "cached": result.cached,
            "duration_ms": self.duration_ms,
            "counts": dict(result.counts),
            "metrics": dict(result.metrics),
            "artifacts": {name: str(path) for name, path in result.artifacts.items()},
            "error_code": None if self.error_code is None else self.error_code.value,
            "error_detail": self.error_detail,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSummary:
    """Outcome of a run, written to `summary.json`.

    Attributes:
        run_id: Run id.
        started_at: Start time, timezone-aware.
        finished_at: Finish time, timezone-aware.
        stages: Outcome of every stage, in pipeline order.
    """

    run_id: str
    started_at: datetime
    finished_at: datetime
    stages: tuple[StageOutcome, ...]

    @property
    def failed_stage(self) -> StageOutcome | None:
        """The stage that stopped the run, or None if every stage succeeded."""
        return next((stage for stage in self.stages if stage.status is StageStatus.FAILED), None)

    @property
    def status(self) -> RunStatus:
        """Failed if any stage failed; derived, so it cannot disagree with the stages."""
        return RunStatus.SUCCEEDED if self.failed_stage is None else RunStatus.FAILED

    def to_json(self) -> dict[str, object]:
        """Returns the content of `summary.json`; the hint comes from the error catalog."""
        failed = self.failed_stage
        code = None if failed is None else failed.error_code
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "started_at": format_utc(self.started_at),
            "finished_at": format_utc(self.finished_at),
            "failed_stage": None if failed is None else failed.name,
            "error_code": None if code is None else code.value,
            "hint": None if code is None else ERROR_CATALOG[code].hint,
            "stages": [stage.to_json() for stage in self.stages],
        }


class TrainingRunner:
    """Runs stages in order, checks their postconditions and records the run.

    The CLI and the worker use the same runner, so research runs and retraining follow the
    same rules.

    Args:
        clock: Returns the current timezone-aware time.
        timer: Returns monotonic seconds, for stage durations.
    """

    def __init__(
        self, clock: Callable[[], datetime], timer: Callable[[], float] = time.perf_counter
    ) -> None:
        self._clock = clock
        self._timer = timer

    def run(self, stages: Sequence[Stage], ctx: RunContext) -> RunSummary:
        """Runs the stages; a failed stage stops the run and is reported, not raised.

        Postconditions are checked after every stage, including stages that reuse cached
        results. The caller decides what a failed run means, such as the CLI exit code.

        Args:
            stages: Stages in execution order.
            ctx: Context of this run.

        Returns:
            Summary of the run, also written to `summary.json` in the run folder.

        Raises:
            ValueError: If a stage name is malformed or repeated; nothing runs and nothing is
                written then.
        """
        _validate_stage_names(stages)
        recorder = RunRecorder(ctx.run_dir)
        started_at = self._clock()
        recorder.event(
            Event.RUN_STARTED,
            ts=format_utc(started_at),
            level="info",
            module=__name__,
            run_id=ctx.run_id,
        )
        outcomes: list[StageOutcome] = []
        for stage in stages:
            if outcomes and outcomes[-1].status is not StageStatus.SUCCEEDED:
                outcomes.append(
                    StageOutcome(name=stage.name, version=stage.version, status=StageStatus.NOT_RUN)
                )
                continue
            outcomes.append(self._run_stage(stage, ctx, recorder))
        summary = RunSummary(
            run_id=ctx.run_id,
            started_at=started_at,
            finished_at=self._clock(),
            stages=tuple(outcomes),
        )
        recorder.event(
            Event.RUN_FINISHED,
            ts=format_utc(summary.finished_at),
            level="info" if summary.status is RunStatus.SUCCEEDED else "error",
            module=__name__,
            run_id=ctx.run_id,
            status=summary.status.value,
        )
        recorder.write_summary(summary.to_json())
        return summary

    def _run_stage(self, stage: Stage, ctx: RunContext, recorder: RunRecorder) -> StageOutcome:
        recorder.event(
            Event.STAGE_STARTED,
            ts=format_utc(self._clock()),
            level="info",
            module=__name__,
            run_id=ctx.run_id,
            stage=stage.name,
        )
        started = self._timer()
        try:
            result = stage.run(ctx)
            stage.check(ctx, result)
        except Exception as error:
            code, detail = _describe(error)
            duration_ms = self._elapsed_ms(started)
            recorder.event(
                Event.STAGE_FAILED,
                ts=format_utc(self._clock()),
                level="error",
                module=__name__,
                run_id=ctx.run_id,
                stage=stage.name,
                error_code=code.value,
                detail=detail,
                exc_type=type(error).__name__,
                traceback="".join(traceback.format_exception(error, limit=-_TRACEBACK_FRAMES)),
                duration_ms=duration_ms,
            )
            return StageOutcome(
                name=stage.name,
                version=stage.version,
                status=StageStatus.FAILED,
                duration_ms=duration_ms,
                error_code=code,
                error_detail=detail,
            )
        duration_ms = self._elapsed_ms(started)
        recorder.event(
            Event.STAGE_FINISHED,
            ts=format_utc(self._clock()),
            level="info",
            module=__name__,
            run_id=ctx.run_id,
            stage=stage.name,
            cached=result.cached,
            duration_ms=duration_ms,
        )
        return StageOutcome(
            name=stage.name,
            version=stage.version,
            status=StageStatus.SUCCEEDED,
            result=result,
            duration_ms=duration_ms,
        )

    def _elapsed_ms(self, started: float) -> int:
        return round((self._timer() - started) * 1000)


def _validate_stage_names(stages: Sequence[Stage]) -> None:
    names = [stage.name for stage in stages]
    for name in names:
        if _STAGE_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"stage name {name!r} must match {STAGE_NAME_PATTERN}")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"stage names repeat: {', '.join(repeated)}")


def _describe(error: Exception) -> tuple[ErrorCode, str | None]:
    if isinstance(error, LoadvisionError):
        return error.code, error.detail
    return ErrorCode.INTERNAL_ERROR, f"{type(error).__name__}: {error}"
