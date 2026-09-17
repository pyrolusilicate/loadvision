from datetime import timedelta
from typing import Any
from uuid import UUID

import pytest

from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.training_job import TrainingJob, TrainingJobStatus, TrainingMode
from loadvision.errors import ErrorCode
from tests.builders import T0, make_job

LATER = T0 + timedelta(minutes=5)
VERSION = ModelVersionId("2026-09-18.1")


@pytest.mark.unit
def test_a_new_job_waits_for_a_worker() -> None:
    job = make_job()

    assert job.status is TrainingJobStatus.QUEUED
    assert (job.started_at, job.finished_at, job.error_code) == (None, None, None)
    assert job.done is False


@pytest.mark.unit
def test_a_claimed_job_remembers_what_it_is_compared_against() -> None:
    running = make_job().started(at=LATER, base_version=VERSION)

    assert running.status is TrainingJobStatus.RUNNING
    assert (running.started_at, running.base_version) == (LATER, VERSION)
    assert running.done is False


@pytest.mark.unit
def test_a_rejected_candidate_is_still_a_finished_job() -> None:
    finished = (
        make_job()
        .started(at=T0, base_version=None)
        .succeeded(at=LATER, run_id="run-1", result_version=VERSION, gate_passed=False)
    )

    assert finished.status is TrainingJobStatus.SUCCEEDED
    assert (finished.gate_passed, finished.result_version) == (False, VERSION)
    assert finished.error_code is None
    assert finished.done is True


@pytest.mark.unit
def test_a_failed_job_carries_its_error_code() -> None:
    failed = (
        make_job()
        .started(at=T0, base_version=None)
        .failed(at=LATER, run_id=None, error_code=ErrorCode.STAGE_CHECK_FAILED)
    )

    assert failed.error_code is ErrorCode.STAGE_CHECK_FAILED
    assert (failed.gate_passed, failed.result_version) == (False, None)
    assert failed.done is True


@pytest.mark.unit
def test_a_job_moves_only_through_its_own_states() -> None:
    queued = make_job()
    running = queued.started(at=T0, base_version=None)
    done = running.succeeded(at=T0, run_id="run-1", result_version=None, gate_passed=True)

    with pytest.raises(ValueError, match="cannot be succeeded"):
        queued.succeeded(at=T0, run_id="run-1", result_version=None, gate_passed=True)
    with pytest.raises(ValueError, match="cannot be started"):
        running.started(at=T0, base_version=None)
    with pytest.raises(ValueError, match="cannot be failed"):
        done.failed(at=T0, run_id="run-1", error_code=ErrorCode.INTERNAL_ERROR)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"status": TrainingJobStatus.RUNNING}, "started_at"),
        ({"status": TrainingJobStatus.QUEUED, "started_at": T0}, "started_at"),
        ({"status": TrainingJobStatus.SUCCEEDED, "started_at": T0}, "finished_at"),
        (
            {"status": TrainingJobStatus.FAILED, "started_at": T0, "finished_at": T0},
            "error_code",
        ),
        (
            {
                "status": TrainingJobStatus.SUCCEEDED,
                "started_at": T0,
                "finished_at": T0,
                "error_code": ErrorCode.INTERNAL_ERROR,
            },
            "error_code",
        ),
        ({"status": TrainingJobStatus.RUNNING, "started_at": T0 - timedelta(hours=1)}, "before"),
        (
            {
                "status": TrainingJobStatus.SUCCEEDED,
                "started_at": LATER,
                "finished_at": T0,
            },
            "before",
        ),
    ],
    ids=[
        "running-without-start",
        "queued-with-start",
        "succeeded-without-finish",
        "failed-without-code",
        "succeeded-with-code",
        "started-before-created",
        "finished-before-started",
    ],
)
def test_impossible_jobs_are_refused(fields: dict[str, Any], message: str) -> None:
    values: dict[str, Any] = {
        "id": UUID(int=1),
        "mode": TrainingMode.FAST,
        "status": TrainingJobStatus.QUEUED,
        "created_at": T0,
    }
    with pytest.raises(ValueError, match=message):
        TrainingJob(**{**values, **fields})
