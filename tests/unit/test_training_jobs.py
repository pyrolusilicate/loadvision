from datetime import timedelta
from uuid import UUID

import pytest

from loadvision.application.training_jobs import GetTrainingJob, ListTrainingJobs, StartTraining
from loadvision.domain.training_job import TrainingJobStatus, TrainingMode
from loadvision.errors import ErrorCode, LoadvisionError
from tests.builders import T0
from tests.fakes.clock import FixedClock, SequentialIds
from tests.fakes.in_memory_db import InMemoryStore


def start(store: InMemoryStore, clock: FixedClock, ids: SequentialIds) -> StartTraining:
    return StartTraining(uow_factory=store.unit_of_work, clock=clock, new_id=ids)


@pytest.mark.unit
def test_a_queued_job_waits_for_the_worker() -> None:
    store = InMemoryStore()

    job = start(store, FixedClock(T0), SequentialIds())()

    assert (job.id, job.mode, job.status) == (
        UUID(int=1),
        TrainingMode.FAST,
        TrainingJobStatus.QUEUED,
    )
    assert job.created_at == T0
    assert store.training_jobs == {job.id: job}


@pytest.mark.unit
def test_the_mode_is_recorded() -> None:
    store = InMemoryStore()

    job = start(store, FixedClock(T0), SequentialIds())(TrainingMode.FULL)

    assert job.mode is TrainingMode.FULL


@pytest.mark.unit
@pytest.mark.parametrize("claimed", [False, True], ids=["queued", "running"])
def test_a_second_retraining_is_refused_while_one_is_unfinished(claimed: bool) -> None:
    store = InMemoryStore()
    clock = FixedClock(T0)
    ids = SequentialIds()
    first = start(store, clock, ids)()
    if claimed:
        with store.unit_of_work() as uow:
            uow.training_jobs.update(first.started(at=T0, base_version=None))
            uow.commit()
    clock.advance(timedelta(minutes=1))

    with pytest.raises(LoadvisionError) as raised:
        start(store, clock, ids)()

    assert raised.value.code is ErrorCode.TRAINING_ALREADY_RUNNING
    assert len(store.training_jobs) == 1


@pytest.mark.unit
def test_a_new_job_is_queued_once_the_previous_one_ended() -> None:
    store = InMemoryStore()
    clock = FixedClock(T0)
    ids = SequentialIds()
    first = start(store, clock, ids)()
    with store.unit_of_work() as uow:
        uow.training_jobs.update(
            first.started(at=T0, base_version=None).failed(
                at=T0, run_id="run-1", error_code=ErrorCode.INTERNAL_ERROR
            )
        )
        uow.commit()
    clock.advance(timedelta(minutes=1))

    second = start(store, clock, ids)()

    assert second.id == UUID(int=2)
    assert len(store.training_jobs) == 2


@pytest.mark.unit
def test_recent_jobs_come_back_newest_first() -> None:
    store = InMemoryStore()
    clock = FixedClock(T0)
    ids = SequentialIds()
    first = start(store, clock, ids)()
    with store.unit_of_work() as uow:
        uow.training_jobs.update(
            first.started(at=T0, base_version=None).failed(
                at=T0, run_id="run-1", error_code=ErrorCode.INTERNAL_ERROR
            )
        )
        uow.commit()
    clock.advance(timedelta(minutes=1))
    second = start(store, clock, ids)()

    jobs = ListTrainingJobs(uow_factory=store.unit_of_work)()

    assert [job.id for job in jobs] == [second.id, first.id]
    assert ListTrainingJobs(uow_factory=store.unit_of_work)(1) == [second]


@pytest.mark.unit
def test_a_job_is_read_by_id() -> None:
    store = InMemoryStore()
    job = start(store, FixedClock(T0), SequentialIds())()

    assert GetTrainingJob(uow_factory=store.unit_of_work)(job.id) == job


@pytest.mark.unit
def test_an_unknown_job_is_not_found() -> None:
    store = InMemoryStore()

    with pytest.raises(LoadvisionError) as raised:
        GetTrainingJob(uow_factory=store.unit_of_work)(UUID(int=404))
    assert raised.value.code is ErrorCode.TRAINING_JOB_NOT_FOUND
