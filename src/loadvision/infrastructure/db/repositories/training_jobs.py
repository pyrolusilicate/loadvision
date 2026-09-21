"""Postgres repository for training jobs."""

from typing import Any
from uuid import UUID

from sqlalchemy import Connection, Row, func, insert, select, update

from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.training_job import TrainingJob, TrainingJobStatus, TrainingMode
from loadvision.errors import ErrorCode
from loadvision.infrastructure.db.tables import training_jobs

_UNFINISHED = (TrainingJobStatus.QUEUED.value, TrainingJobStatus.RUNNING.value)


class PostgresTrainingJobRepository:
    """`TrainingJobRepository` on the connection of one unit of work."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def add(self, job: TrainingJob) -> None:
        self._connection.execute(insert(training_jobs).values(_values(job)))

    def get(self, job_id: UUID) -> TrainingJob | None:
        row = self._connection.execute(
            select(training_jobs).where(training_jobs.c.id == job_id)
        ).one_or_none()
        return None if row is None else _job(row)

    def find_unfinished(self) -> TrainingJob | None:
        row = self._connection.execute(
            select(training_jobs)
            .where(training_jobs.c.status.in_(_UNFINISHED))
            .order_by(training_jobs.c.created_at)
            .limit(1)
        ).first()
        return None if row is None else _job(row)

    def claim_next(self) -> TrainingJob | None:
        """Takes the oldest queued job and locks its row until the transaction ends.

        `SKIP LOCKED` lets a second worker walk past a row a first one already took, instead
        of waiting for it. Nothing is claimed while another job runs: that is the queue rule,
        and the partial unique index of the table is the last line of defence behind it.
        """
        running = select(training_jobs.c.id).where(
            training_jobs.c.status == TrainingJobStatus.RUNNING.value
        )
        row = self._connection.execute(
            select(training_jobs)
            .where(
                training_jobs.c.status == TrainingJobStatus.QUEUED.value,
                ~running.exists(),
            )
            .order_by(training_jobs.c.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).first()
        return None if row is None else _job(row)

    def list_recent(self, limit: int, offset: int = 0) -> list[TrainingJob]:
        rows = self._connection.execute(
            select(training_jobs)
            .order_by(training_jobs.c.created_at.desc(), training_jobs.c.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return [_job(row) for row in rows]

    def count(self) -> int:
        return self._connection.execute(
            select(func.count()).select_from(training_jobs)
        ).scalar_one()

    def update(self, job: TrainingJob) -> None:
        """Saves the status and the outcome.

        Raises:
            ValueError: If the job was never stored; an update of no rows would pass silently.
        """
        result = self._connection.execute(
            update(training_jobs).where(training_jobs.c.id == job.id).values(_values(job))
        )
        if result.rowcount == 0:
            raise ValueError(f"training job {job.id} is not stored")


def _values(job: TrainingJob) -> dict[str, object]:
    return {
        "id": job.id,
        "mode": job.mode.value,
        "status": job.status.value,
        "run_id": job.run_id,
        "base_version": None if job.base_version is None else job.base_version.value,
        "result_version": None if job.result_version is None else job.result_version.value,
        "gate_passed": job.gate_passed,
        "gate_reason": job.gate_reason,
        "error_code": None if job.error_code is None else job.error_code.value,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def _job(row: Row[Any]) -> TrainingJob:
    return TrainingJob(
        id=row.id,
        mode=TrainingMode(row.mode),
        status=TrainingJobStatus(row.status),
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        run_id=row.run_id,
        base_version=None if row.base_version is None else ModelVersionId(row.base_version),
        result_version=None if row.result_version is None else ModelVersionId(row.result_version),
        gate_passed=row.gate_passed,
        gate_reason=row.gate_reason,
        error_code=None if row.error_code is None else ErrorCode(row.error_code),
    )
