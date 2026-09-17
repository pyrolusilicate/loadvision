import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from uuid import UUID

import pytest

from loadvision.application.ports import UnitOfWorkFactory
from loadvision.cli.main import main
from loadvision.domain.training_job import TrainingJobStatus, TrainingMode
from loadvision.errors import ErrorCode
from tests.builders import T0, make_job
from tests.fakes.in_memory_db import InMemoryStore


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> InMemoryStore:
    store = InMemoryStore()

    @contextmanager
    def factory() -> Iterator[UnitOfWorkFactory]:
        yield store.unit_of_work

    monkeypatch.setattr("loadvision.cli.commands.jobs.unit_of_work_factory", factory)
    return store


@pytest.mark.unit
def test_a_job_is_queued(store: InMemoryStore, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    assert main(["jobs", "start"]) == 0

    [job] = store.training_jobs.values()
    assert (job.status, job.mode) == (TrainingJobStatus.QUEUED, TrainingMode.FAST)
    assert str(job.id) in caplog.text


@pytest.mark.unit
def test_the_mode_reaches_the_job(store: InMemoryStore) -> None:
    assert main(["jobs", "start", "--mode", "full"]) == 0
    [job] = store.training_jobs.values()
    assert job.mode is TrainingMode.FULL


@pytest.mark.unit
def test_a_second_job_is_refused_while_one_waits(
    store: InMemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    assert main(["jobs", "start"]) == 0

    assert main(["jobs", "start"]) == 1

    assert "TRAINING_ALREADY_RUNNING" in caplog.text
    assert "is queued" in caplog.text
    assert len(store.training_jobs) == 1


@pytest.mark.unit
def test_an_empty_queue_says_so(store: InMemoryStore, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    assert main(["jobs", "list"]) == 0

    assert "the queue is empty" in caplog.text


@pytest.mark.unit
def test_recent_jobs_are_shown_newest_first(
    store: InMemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    finished = (
        make_job(UUID(int=1), at=T0)
        .started(at=T0, base_version=None)
        .failed(at=T0, run_id="run-1", error_code=ErrorCode.STAGE_CHECK_FAILED)
    )
    waiting = make_job(UUID(int=2), at=T0 + timedelta(minutes=5))
    with store.unit_of_work() as uow:
        uow.training_jobs.add(make_job(UUID(int=1), at=T0))
        uow.training_jobs.update(finished)
        uow.training_jobs.add(waiting)
        uow.commit()

    assert main(["jobs", "list"]) == 0

    header, first, second = (record.getMessage() for record in caplog.records)
    assert header.split() == ["job", "status", "queued", "result", "gate"]
    assert first.split() == [str(waiting.id), "queued", "2026-09-16T12:05:00Z", "-", "-"]
    assert second.split() == [
        str(finished.id),
        "failed",
        "2026-09-16T12:00:00Z",
        "STAGE_CHECK_FAILED",
        "rejected",
    ]


@pytest.mark.unit
def test_the_limit_is_honoured(store: InMemoryStore, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    with store.unit_of_work() as uow:
        for number in range(1, 4):
            uow.training_jobs.add(make_job(UUID(int=number), at=T0 + timedelta(minutes=number)))
        uow.commit()

    assert main(["jobs", "list", "--limit", "2"]) == 0

    assert len(caplog.records) == 3
