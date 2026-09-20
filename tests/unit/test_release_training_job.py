import pytest

from loadvision.application.training_jobs import ReleaseTrainingJob, StartTraining
from loadvision.domain.training_job import TrainingJobStatus
from loadvision.errors import ErrorCode
from tests.builders import T0
from tests.fakes.clock import FixedClock, SequentialIds
from tests.fakes.in_memory_db import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def release(store: InMemoryStore) -> ReleaseTrainingJob:
    return ReleaseTrainingJob(uow_factory=store.unit_of_work, clock=FixedClock(T0))


def start(store: InMemoryStore) -> StartTraining:
    return StartTraining(
        uow_factory=store.unit_of_work, clock=FixedClock(T0), new_id=SequentialIds()
    )


def claim(store: InMemoryStore) -> None:
    """Takes the queued job and starts it, the way a worker does."""
    with store.unit_of_work() as uow:
        job = uow.training_jobs.claim_next()
        assert job is not None
        uow.training_jobs.update(job.started(at=T0, base_version=None))
        uow.commit()


@pytest.mark.unit
def test_a_job_whose_worker_is_gone_is_ended_and_the_queue_takes_a_new_one(
    store: InMemoryStore,
) -> None:
    queue = start(store)
    queued = queue()
    claim(store)

    ended = release(store)()

    assert ended is not None
    assert (ended.id, ended.status) == (queued.id, TrainingJobStatus.FAILED)
    assert ended.error_code is ErrorCode.INTERNAL_ERROR
    assert ended.finished_at == T0
    # The queue was blocked by that job and takes work again.
    assert queue().status is TrainingJobStatus.QUEUED


@pytest.mark.unit
def test_a_job_that_waits_for_a_worker_is_left_alone(store: InMemoryStore) -> None:
    queued = start(store)()

    assert release(store)() is None
    with store.unit_of_work() as uow:
        assert uow.training_jobs.get(queued.id) == queued


@pytest.mark.unit
def test_nothing_to_release_is_not_an_error(store: InMemoryStore) -> None:
    assert release(store)() is None
