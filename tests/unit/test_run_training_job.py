from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest

from loadvision.application.model_versions import ActivateModel, RegisterModel
from loadvision.application.ports import TrainingOutcome, TrainingStatus
from loadvision.application.run_training_job import RunTrainingJob
from loadvision.domain.model_version import ModelVersionId, ModelVersionStatus
from loadvision.domain.training_job import TrainingJobStatus
from loadvision.errors import ErrorCode
from loadvision.ml.packaging.model_package import ModelPackageError
from tests.builders import T0, make_image_record, make_job, make_label
from tests.fakes.clock import FixedClock
from tests.fakes.in_memory_db import InMemoryStore
from tests.fakes.training_runner import RUN_ID, FakeTrainingRunner, succeeded

NEW_VERSION = ModelVersionId("2026-09-18.1")
ACTIVE_VERSION = ModelVersionId("baseline-v0")
LATER = T0 + timedelta(minutes=30)


@dataclass
class FakePublisher:
    """Stands for putting a package into the models folder and registering its version."""

    version: ModelVersionId = NEW_VERSION
    error: Exception | None = None
    packages: list[Path] = field(default_factory=list[Path])

    def __call__(self, package_path: Path) -> ModelVersionId:
        self.packages.append(package_path)
        if self.error is not None:
            raise self.error
        return self.version


@dataclass
class Worker:
    store: InMemoryStore
    clock: FixedClock
    runner: FakeTrainingRunner
    publisher: FakePublisher

    def run(self) -> RunTrainingJob:
        return RunTrainingJob(
            uow_factory=self.store.unit_of_work,
            clock=self.clock,
            runner=self.runner,
            publish=self.publisher,
            activate=ActivateModel(uow_factory=self.store.unit_of_work, clock=self.clock),
        )


def make_worker(
    *, outcome: TrainingOutcome | None = None, error: Exception | None = None
) -> Worker:
    store = InMemoryStore()
    clock = FixedClock(T0)
    RegisterModel(uow_factory=store.unit_of_work, clock=clock)(NEW_VERSION)
    return Worker(
        store=store,
        clock=clock,
        runner=FakeTrainingRunner(outcome, error),
        publisher=FakePublisher(),
    )


def queue_job(worker: Worker) -> None:
    with worker.store.unit_of_work() as uow:
        uow.training_jobs.add(make_job())
        uow.commit()


def with_labels_and_active_version(worker: Worker) -> None:
    activate = ActivateModel(uow_factory=worker.store.unit_of_work, clock=worker.clock)
    RegisterModel(uow_factory=worker.store.unit_of_work, clock=worker.clock)(ACTIVE_VERSION)
    activate(ACTIVE_VERSION)
    with worker.store.unit_of_work() as uow:
        for raw, load in ((b"photo-a", 60.0), (b"photo-b", None)):
            uow.images.add(make_image_record(raw))
            uow.labels.save(make_label(raw, load))
        uow.commit()


@pytest.mark.unit
def test_an_empty_queue_runs_nothing() -> None:
    worker = make_worker()

    assert worker.run()() is None
    assert worker.runner.requests == []


@pytest.mark.unit
def test_a_passed_candidate_is_published_and_activated() -> None:
    worker = make_worker()
    with_labels_and_active_version(worker)
    queue_job(worker)
    worker.clock.advance(timedelta(minutes=30))

    finished = worker.run()()

    assert finished is not None
    assert finished.status is TrainingJobStatus.SUCCEEDED
    assert (finished.run_id, finished.gate_passed) == (RUN_ID, True)
    assert (finished.base_version, finished.result_version) == (ACTIVE_VERSION, NEW_VERSION)
    assert finished.started_at == LATER
    assert worker.publisher.packages == [Path("runs/run-1/package")]
    assert worker.store.model_versions[NEW_VERSION].status is ModelVersionStatus.ACTIVE
    assert worker.store.model_versions[ACTIVE_VERSION].status is ModelVersionStatus.RETIRED
    assert worker.store.training_jobs[UUID(int=1)] == finished


@pytest.mark.unit
def test_the_claimed_job_is_visible_as_running_while_it_runs() -> None:
    worker = make_worker()
    queue_job(worker)
    seen: list[TrainingJobStatus] = []
    worker.runner.during = lambda: seen.append(worker.store.training_jobs[UUID(int=1)].status)

    worker.run()()

    # Another worker must see the claim, not a job that still looks free.
    assert seen == [TrainingJobStatus.RUNNING]


@pytest.mark.unit
def test_training_gets_the_usable_labels_and_the_active_version() -> None:
    worker = make_worker()
    with_labels_and_active_version(worker)
    queue_job(worker)

    worker.run()()

    [request] = worker.runner.requests
    assert [label.load_pct.value for label in request.labels] == [60.0]
    assert request.base_version == ACTIVE_VERSION


@pytest.mark.unit
def test_a_rejected_candidate_is_registered_but_not_activated() -> None:
    worker = make_worker(outcome=succeeded(gate_passed=False))
    with_labels_and_active_version(worker)
    queue_job(worker)

    finished = worker.run()()

    assert finished is not None
    assert (finished.status, finished.gate_passed) == (TrainingJobStatus.SUCCEEDED, False)
    assert finished.result_version == NEW_VERSION
    assert worker.store.model_versions[ACTIVE_VERSION].status is ModelVersionStatus.ACTIVE
    assert worker.publisher.packages == [Path("runs/run-1/package")]


@pytest.mark.unit
def test_a_failed_run_keeps_its_error_code_and_publishes_nothing() -> None:
    outcome = TrainingOutcome(
        run_id=RUN_ID,
        status=TrainingStatus.FAILED,
        error_code=ErrorCode.STAGE_CHECK_FAILED,
        error_detail="stage train_folds: fold 2 came out empty",
    )
    worker = make_worker(outcome=outcome)
    queue_job(worker)

    finished = worker.run()()

    assert finished is not None
    assert finished.status is TrainingJobStatus.FAILED
    assert finished.error_code is ErrorCode.STAGE_CHECK_FAILED
    assert worker.publisher.packages == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ModelPackageError("head.npz is missing"), ErrorCode.MODEL_PACKAGE_INVALID),
        (FileExistsError("storage/models/2026-09-18.1"), ErrorCode.INTERNAL_ERROR),
    ],
    ids=["invalid-package", "folder-taken"],
)
def test_a_package_that_cannot_be_served_fails_the_job(error: Exception, code: ErrorCode) -> None:
    worker = make_worker()
    worker.publisher.error = error
    queue_job(worker)

    finished = worker.run()()

    assert finished is not None
    assert finished.status is TrainingJobStatus.FAILED
    assert finished.error_code is code
    assert worker.store.model_versions[NEW_VERSION].status is ModelVersionStatus.CANDIDATE


@pytest.mark.unit
def test_a_crashing_runner_leaves_no_running_job() -> None:
    worker = make_worker(error=RuntimeError("out of memory"))
    queue_job(worker)

    with pytest.raises(RuntimeError, match="out of memory"):
        worker.run()()

    stored = worker.store.training_jobs[UUID(int=1)]
    assert stored.status is TrainingJobStatus.FAILED
    assert stored.error_code is ErrorCode.INTERNAL_ERROR


@pytest.mark.unit
def test_only_one_job_runs_at_a_time() -> None:
    worker = make_worker()
    queue_job(worker)
    with worker.store.unit_of_work() as uow:
        uow.training_jobs.update(make_job().started(at=T0, base_version=None))
        uow.training_jobs.add(make_job(UUID(int=2), at=T0 + timedelta(minutes=1)))
        uow.commit()

    assert worker.run()() is None
    assert worker.runner.requests == []


@pytest.mark.unit
def test_an_activation_conflict_fails_the_job() -> None:
    worker = make_worker()
    queue_job(worker)
    with worker.store.unit_of_work() as uow:
        rejected = worker.store.model_versions[NEW_VERSION]
        uow.model_versions.update(rejected.reject())
        uow.commit()

    finished = worker.run()()

    assert finished is not None
    assert finished.status is TrainingJobStatus.FAILED
    assert finished.error_code is ErrorCode.MODEL_VERSION_STATUS_CONFLICT
