import functools
import threading
from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import Engine

from loadvision.application.model_versions import ActivateModel, RegisterModel
from loadvision.application.ports import UnitOfWorkFactory
from loadvision.application.run_training_job import RunTrainingJob
from loadvision.application.start_training import StartTraining
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.training_job import TrainingJob, TrainingJobStatus
from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.infrastructure.db.session import PostgresUnitOfWork
from tests.builders import T0, make_job
from tests.fakes.clock import FixedClock, SequentialIds
from tests.fakes.training_runner import RUN_ID, FakeTrainingRunner

NEW_VERSION = ModelVersionId("2026-09-18.1")
BLOCKED_SECONDS = 0.5


def factory_of(engine: Engine) -> UnitOfWorkFactory:
    return functools.partial(PostgresUnitOfWork, engine)


@pytest.mark.integration
def test_a_job_goes_from_the_queue_to_an_activated_version(engine: Engine) -> None:
    factory = factory_of(engine)
    clock = FixedClock(T0)
    RegisterModel(uow_factory=factory, clock=clock)(NEW_VERSION)
    job = StartTraining(uow_factory=factory, clock=clock, new_id=SequentialIds())()
    worker = RunTrainingJob(
        uow_factory=factory,
        clock=clock,
        runner=FakeTrainingRunner(),
        publish=lambda _path: NEW_VERSION,
        activate=ActivateModel(uow_factory=factory, clock=clock),
    )

    finished = worker()

    assert finished is not None
    assert finished.id == job.id
    assert (finished.status, finished.run_id) == (TrainingJobStatus.SUCCEEDED, RUN_ID)
    with factory() as uow:
        stored = uow.training_jobs.get(job.id)
        active = uow.model_versions.get_active()
    assert stored == finished
    assert active is not None and active.version == NEW_VERSION
    assert stored is not None and stored.result_version == NEW_VERSION


@pytest.mark.integration
def test_a_second_worker_finds_nothing_while_the_first_holds_the_job(engine: Engine) -> None:
    factory = factory_of(engine)
    with factory() as uow:
        uow.training_jobs.add(make_job(UUID(int=1), at=T0))
        uow.training_jobs.add(make_job(UUID(int=2), at=T0 + timedelta(minutes=1)))
        uow.commit()
    second: list[TrainingJob | None] = []

    def claim_in_another_worker() -> None:
        with factory() as uow:
            second.append(uow.training_jobs.claim_next())

    with factory() as first:
        claimed = first.training_jobs.claim_next()
        thread = threading.Thread(target=claim_in_another_worker)
        thread.start()
        thread.join(BLOCKED_SECONDS)
        assert not thread.is_alive(), "a locked row must be skipped, not waited for"

    assert claimed is not None and claimed.id == UUID(int=1)
    # The second worker skipped the locked row and took the next one.
    assert [job.id for job in second if job is not None] == [UUID(int=2)]


@pytest.mark.integration
def test_the_database_allows_only_one_running_job(engine: Engine) -> None:
    factory = factory_of(engine)
    clock = FixedClock(T0)
    start = StartTraining(uow_factory=factory, clock=clock, new_id=SequentialIds())
    job = start()
    with factory() as uow:
        uow.training_jobs.update(job.started(at=T0, base_version=None))
        uow.commit()

    with pytest.raises(LoadvisionError) as raised:
        start()
    assert raised.value.code is ErrorCode.TRAINING_ALREADY_RUNNING

    with factory() as uow:
        assert uow.training_jobs.claim_next() is None
