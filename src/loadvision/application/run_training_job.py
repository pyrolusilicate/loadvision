"""Use case: take one job from the queue, retrain, and record what came out of it."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loadvision.application.model_versions import ActivateModel
from loadvision.application.ports import (
    Clock,
    LabeledImage,
    TrainingOutcome,
    TrainingRequest,
    TrainingRunner,
    TrainingStatus,
    UnitOfWorkFactory,
)
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.training_job import TrainingJob
from loadvision.errors import ErrorCode, LoadvisionError

type PackagePublisher = Callable[[Path], ModelVersionId]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunTrainingJob:
    """Runs one queued job: claim, retrain, publish the package, record the outcome.

    The pipeline runs outside any transaction: it takes minutes, and a database connection
    must not wait for it. The claim and the outcome are separate short transactions.

    A rejected candidate is a finished job, not a failure: training worked and the gate did
    its job. The service keeps answering with the model it already has.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
        runner: Retraining itself.
        publish: Puts a package where the service reads packages from and registers its
            version; returns that version.
        activate: Activation use case, called only for a candidate that passed the gate.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock
    runner: TrainingRunner
    publish: PackagePublisher
    activate: ActivateModel

    def __call__(self) -> TrainingJob | None:
        """Runs the next job.

        Returns:
            The finished job, or None when the queue is empty or another job is running.

        Raises:
            Exception: Whatever the runner raises. The job is recorded as failed first, so a
                crash does not leave it running forever and block the queue.
        """
        claimed = self._claim()
        if claimed is None:
            return None
        job, request = claimed
        _logger.info("job %s started with %d labels", job.id, len(request.labels))
        try:
            outcome = self.runner.run(request)
        except Exception as error:
            self._save(
                job.failed(at=self.clock(), run_id=None, error_code=ErrorCode.INTERNAL_ERROR)
            )
            _logger.exception("job %s crashed: %s", job.id, error)
            raise
        finished = self._outcome(job, outcome)
        self._save(finished)
        _logger.info(
            "job %s %s, run %s, gate %s",
            finished.id,
            finished.status,
            finished.run_id,
            "passed" if finished.gate_passed else "rejected",
        )
        return finished

    def _claim(self) -> tuple[TrainingJob, TrainingRequest] | None:
        now = self.clock()
        with self.uow_factory() as uow:
            job = uow.training_jobs.claim_next()
            if job is None:
                return None
            active = uow.model_versions.get_active()
            labels = tuple(
                LabeledImage.from_label(label) for label in uow.labels.list_for_training()
            )
            running = job.started(at=now, base_version=None if active is None else active.version)
            uow.training_jobs.update(running)
            uow.commit()
        return running, TrainingRequest(
            mode=running.mode, labels=labels, base_version=running.base_version
        )

    def _outcome(self, job: TrainingJob, outcome: TrainingOutcome) -> TrainingJob:
        now = self.clock()
        if outcome.status is TrainingStatus.FAILED or outcome.package_path is None:
            return job.failed(
                at=now,
                run_id=outcome.run_id,
                error_code=outcome.error_code or ErrorCode.INTERNAL_ERROR,
            )
        try:
            version = self.publish(outcome.package_path)
            if outcome.gate_passed:
                self.activate(version)
        except (LoadvisionError, OSError, ValueError) as error:
            _logger.exception("job %s produced a package that cannot be served: %s", job.id, error)
            code = error.code if isinstance(error, LoadvisionError) else ErrorCode.INTERNAL_ERROR
            return job.failed(at=now, run_id=outcome.run_id, error_code=code)
        return job.succeeded(
            at=now,
            run_id=outcome.run_id,
            result_version=version,
            gate_passed=outcome.gate_passed,
            gate_reason=_reason_of(outcome),
        )

    def _save(self, job: TrainingJob) -> None:
        with self.uow_factory() as uow:
            uow.training_jobs.update(job)
            uow.commit()


def _reason_of(outcome: TrainingOutcome) -> str | None:
    """Returns the one line the gate wrote about its decision, if the run reached the gate."""
    reason = outcome.gate_report.get("reason")
    return reason if isinstance(reason, str) else None
