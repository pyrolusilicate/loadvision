"""Training job entity: one retraining from the queue, with its statuses and transitions."""

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.timestamps import ensure_aware
from loadvision.errors import ErrorCode


class TrainingMode(StrEnum):
    """How much of the pipeline a retraining run repeats.

    Attributes:
        FAST: Heads and ensemble weights on cached embeddings of a frozen encoder. Minutes on
            a CPU, so the worker can run it.
        FULL: The whole pipeline, fine-tuning included. Needs a GPU and runs from the CLI.
    """

    FAST = "fast"
    FULL = "full"


class TrainingJobStatus(StrEnum):
    """Where a job is: waiting, running, or done one way or the other."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_DONE: frozenset[TrainingJobStatus] = frozenset(
    {TrainingJobStatus.SUCCEEDED, TrainingJobStatus.FAILED}
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingJob:
    """One retraining asked for by a person or by a schedule.

    A job is a record of what happened, not a handle on a process: the worker claims it, runs
    the pipeline and writes the outcome back. A failed run is still a finished job, and the
    service keeps answering with the model it already has.

    Attributes:
        id: Job identifier.
        mode: How much of the pipeline to repeat.
        status: Where the job is.
        created_at: When it was queued, timezone-aware.
        started_at: When a worker claimed it, or None while it waits.
        finished_at: When it ended, or None while it is not done.
        run_id: Id of the pipeline run, once there is one; its folder holds the events.
        base_version: Version the candidate is compared against, or None before the first
            activation.
        result_version: Version the run produced, or None if it produced none.
        gate_passed: Whether the candidate may replace the active version; None until the run
            ends.
        error_code: Why the run failed, or None.
    """

    id: UUID
    mode: TrainingMode
    status: TrainingJobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    run_id: str | None = None
    base_version: ModelVersionId | None = None
    result_version: ModelVersionId | None = None
    gate_passed: bool | None = None
    error_code: ErrorCode | None = None

    def __post_init__(self) -> None:
        """Validates the timestamps and their agreement with the status.

        The same rules stand as CHECK constraints in the database, so a row written by anything
        else cannot disagree with this entity.

        Raises:
            ValueError: If a timestamp has no UTC offset or is out of order, or if a timestamp,
                an error code or a gate decision does not match the status.
        """
        for name in ("created_at", "started_at", "finished_at"):
            value: datetime | None = getattr(self, name)
            if value is not None:
                ensure_aware(value, name)
        waiting = self.status is TrainingJobStatus.QUEUED
        if (self.started_at is None) != waiting:
            raise ValueError(f"started_at does not match status {self.status}")
        if (self.finished_at is not None) != (self.status in _DONE):
            raise ValueError(f"finished_at does not match status {self.status}")
        if (self.error_code is not None) != (self.status is TrainingJobStatus.FAILED):
            raise ValueError(f"error_code does not match status {self.status}")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at must not be before created_at")
        started, finished = self.started_at, self.finished_at
        if finished is not None and started is not None and finished < started:
            raise ValueError("finished_at must not be before started_at")

    @property
    def done(self) -> bool:
        """Whether the job has ended, successfully or not."""
        return self.status in _DONE

    @classmethod
    def queued(cls, *, job_id: UUID, mode: TrainingMode, at: datetime) -> Self:
        """Creates a job waiting for a worker.

        Args:
            job_id: Job identifier.
            mode: How much of the pipeline to repeat.
            at: Time the job was asked for, timezone-aware.
        """
        return cls(id=job_id, mode=mode, status=TrainingJobStatus.QUEUED, created_at=at)

    def started(self, *, at: datetime, base_version: ModelVersionId | None) -> Self:
        """Returns the job claimed by a worker.

        Args:
            at: Time of the claim, timezone-aware.
            base_version: Active version the candidate will be compared against.

        Raises:
            ValueError: If the job is not waiting.
        """
        self._require(TrainingJobStatus.QUEUED, "started")
        return replace(
            self, status=TrainingJobStatus.RUNNING, started_at=at, base_version=base_version
        )

    def succeeded(
        self,
        *,
        at: datetime,
        run_id: str,
        result_version: ModelVersionId | None,
        gate_passed: bool,
    ) -> Self:
        """Returns the job that produced a package; a rejected candidate is still a success.

        Args:
            at: Time the run ended, timezone-aware.
            run_id: Id of the pipeline run.
            result_version: Version registered from the package, if it was registered.
            gate_passed: Whether the candidate may replace the active version.

        Raises:
            ValueError: If the job is not running.
        """
        self._require(TrainingJobStatus.RUNNING, "succeeded")
        return replace(
            self,
            status=TrainingJobStatus.SUCCEEDED,
            finished_at=at,
            run_id=run_id,
            result_version=result_version,
            gate_passed=gate_passed,
        )

    def failed(self, *, at: datetime, run_id: str | None, error_code: ErrorCode) -> Self:
        """Returns the job that produced no package.

        Args:
            at: Time the run ended, timezone-aware.
            run_id: Id of the pipeline run, if it got that far.
            error_code: Why it failed.

        Raises:
            ValueError: If the job is not running.
        """
        self._require(TrainingJobStatus.RUNNING, "failed")
        return replace(
            self,
            status=TrainingJobStatus.FAILED,
            finished_at=at,
            run_id=run_id,
            error_code=error_code,
            gate_passed=False,
        )

    def _require(self, status: TrainingJobStatus, move: str) -> None:
        if self.status is not status:
            raise ValueError(f"a {self.status} job cannot be {move}")
