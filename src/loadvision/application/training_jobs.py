"""Use cases for the retraining queue: queue a job, read what the queue has been doing."""

from dataclasses import dataclass
from typing import Final
from uuid import UUID

from loadvision.application.paging import Page, page_number
from loadvision.application.ports import Clock, IdFactory, UnitOfWorkFactory
from loadvision.domain.training_job import TrainingJob, TrainingJobStatus, TrainingMode
from loadvision.errors import ErrorCode, LoadvisionError

DEFAULT_LIMIT: Final = 20


@dataclass(frozen=True, slots=True, kw_only=True)
class StartTraining:
    """Queues a retraining for the worker.

    Only one retraining is queued or running at a time: a second one would fight the first for
    the same folds file, the same cache and the same active version.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
        new_id: New job ids.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock
    new_id: IdFactory

    def __call__(self, mode: TrainingMode = TrainingMode.FAST) -> TrainingJob:
        """Queues the job.

        Args:
            mode: How much of the pipeline to repeat.

        Returns:
            The queued job.

        Raises:
            LoadvisionError: `TRAINING_MODE_NOT_SUPPORTED` for any mode but `FAST`;
                `TRAINING_ALREADY_RUNNING` if a job is queued or running.
        """
        # The queue has one runner and it retrains heads on frozen encoders. A job of another
        # mode would wait, get taken and fail minutes later; a refusal here says it at once.
        if mode is not TrainingMode.FAST:
            raise LoadvisionError(
                ErrorCode.TRAINING_MODE_NOT_SUPPORTED, f"the queue runs {TrainingMode.FAST} only"
            )
        with self.uow_factory() as uow:
            unfinished = uow.training_jobs.find_unfinished()
            if unfinished is not None:
                raise LoadvisionError(
                    ErrorCode.TRAINING_ALREADY_RUNNING,
                    f"job {unfinished.id} is {unfinished.status}",
                )
            job = TrainingJob.queued(job_id=self.new_id(), mode=mode, at=self.clock())
            uow.training_jobs.add(job)
            uow.commit()
        return job


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseTrainingJob:
    """Ends a job whose worker is gone, so the queue takes new ones again.

    A worker that dies in the middle of a run - a killed container, a machine that went away -
    leaves its job running, and the queue refuses every new job while one is unfinished. From
    the outside nothing tells a dead run from a slow one, so a person says it: the pages offer
    this only while no worker is connected at all.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock

    def __call__(self) -> TrainingJob | None:
        """Ends the running job.

        Returns:
            The job that was ended, or None when nothing was running. A queued job is left
            alone: it waits for a worker, it does not block anything.
        """
        with self.uow_factory() as uow:
            job = uow.training_jobs.find_unfinished()
            if job is None or job.status is not TrainingJobStatus.RUNNING:
                return None
            # The same code a crash of the runner writes: the job produced no package, and
            # nobody can say more than that about a worker that is not there any more.
            ended = job.failed(at=self.clock(), run_id=None, error_code=ErrorCode.INTERNAL_ERROR)
            uow.training_jobs.update(ended)
            uow.commit()
        return ended


@dataclass(frozen=True, slots=True, kw_only=True)
class ListTrainingJobs:
    """Reads the recent jobs, newest first.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self, limit: int = DEFAULT_LIMIT) -> list[TrainingJob]:
        """Returns at most `limit` jobs, newest first."""
        with self.uow_factory() as uow:
            return uow.training_jobs.list_recent(limit)


@dataclass(frozen=True, slots=True, kw_only=True)
class ListTrainingJobPage:
    """Reads one page of jobs, newest first.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self, *, page: int, size: int) -> Page[TrainingJob]:
        """Returns the page; a number past the end shows the last page."""
        with self.uow_factory() as uow:
            total = uow.training_jobs.count()
            number = page_number(page, total, size)
            jobs = uow.training_jobs.list_recent(size, (number - 1) * size)
        return Page(items=tuple(jobs), number=number, size=size, total=total)


@dataclass(frozen=True, slots=True, kw_only=True)
class CountTrainingLabels:
    """Counts the confirmed labels a retraining would learn from.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self) -> int:
        """Returns how many photos have a confirmed load; unusable photos do not count."""
        with self.uow_factory() as uow:
            return len(uow.labels.list_for_training())


@dataclass(frozen=True, slots=True, kw_only=True)
class GetTrainingJob:
    """Reads one job by id.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self, job_id: UUID) -> TrainingJob:
        """Returns the job.

        Raises:
            LoadvisionError: `TRAINING_JOB_NOT_FOUND` if the job is unknown.
        """
        with self.uow_factory() as uow:
            job = uow.training_jobs.get(job_id)
        if job is None:
            raise LoadvisionError(ErrorCode.TRAINING_JOB_NOT_FOUND, str(job_id))
        return job
