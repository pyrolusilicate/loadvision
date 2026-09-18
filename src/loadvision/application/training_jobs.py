"""Use case: put a retraining into the queue."""

from dataclasses import dataclass

from loadvision.application.ports import Clock, IdFactory, UnitOfWorkFactory
from loadvision.domain.training_job import TrainingJob, TrainingMode
from loadvision.errors import ErrorCode, LoadvisionError


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
            LoadvisionError: `TRAINING_ALREADY_RUNNING` if a job is queued or running.
        """
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
