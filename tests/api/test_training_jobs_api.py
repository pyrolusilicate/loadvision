from datetime import timedelta
from uuid import UUID

import pytest
from lxml import etree

from loadvision.api.xml.parser import NAMESPACE
from loadvision.domain.training_job import TrainingJobStatus
from loadvision.errors import ErrorCode
from tests.api.service import XML, Service, assert_error, child, document
from tests.builders import T0, make_job

QUEUE = "/api/v1/training-jobs"


def request_body(mode: str | None = None) -> bytes:
    inner = "" if mode is None else f"<Mode>{mode}</Mode>"
    return f'<TrainingJobRequest xmlns="{NAMESPACE}">{inner}</TrainingJobRequest>'.encode()


def post(service: Service, body: bytes | None = None, headers: dict[str, str] | None = None):
    return service.client.post(
        QUEUE, content=request_body() if body is None else body, headers=headers or XML
    )


def finished_job(service: Service) -> None:
    with service.store.unit_of_work() as uow:
        uow.training_jobs.add(make_job(UUID(int=7), at=T0))
        uow.training_jobs.update(
            make_job(UUID(int=7), at=T0)
            .started(at=T0 + timedelta(minutes=1), base_version=None)
            .failed(
                at=T0 + timedelta(minutes=2),
                run_id="run_20260918_101500_ab12",
                error_code=ErrorCode.STAGE_CHECK_FAILED,
            )
        )
        uow.commit()


@pytest.mark.api
def test_a_queued_job_comes_back_with_its_id(service: Service) -> None:
    response = post(service)

    assert response.status_code == 201
    root = document(response)
    assert etree.QName(root).localname == "TrainingJob"
    assert (root.get("status"), root.get("mode")) == ("queued", "fast")
    assert root.get("createdAt") == "2026-09-16T12:00:00Z"
    assert root.find(f"{{{NAMESPACE}}}Result") is None
    [stored] = service.store.training_jobs.values()
    assert str(stored.id) == root.get("id")
    assert stored.status is TrainingJobStatus.QUEUED


@pytest.mark.api
def test_the_mode_of_the_request_is_used(service: Service) -> None:
    root = document(post(service, request_body("fast")))
    assert root.get("mode") == "fast"


@pytest.mark.api
def test_a_mode_this_service_cannot_run_is_refused_at_once(service: Service) -> None:
    # The schema knows `full`, this service does not run it: better a refusal than a job that
    # waits, gets taken and fails minutes later.
    assert_error(post(service, request_body("full")), ErrorCode.TRAINING_MODE_NOT_SUPPORTED)
    assert service.store.training_jobs == {}


@pytest.mark.api
def test_a_second_job_is_refused_while_one_waits(service: Service) -> None:
    post(service)

    assert_error(post(service), ErrorCode.TRAINING_ALREADY_RUNNING)
    assert len(service.store.training_jobs) == 1


@pytest.mark.api
@pytest.mark.parametrize(
    ("body", "headers", "code"),
    [
        (request_body("turbo"), None, ErrorCode.INVALID_XML),
        (b"<TrainingJobRequest", None, ErrorCode.INVALID_XML),
        (request_body(), {"Content-Type": "application/json"}, ErrorCode.INVALID_CONTENT_TYPE),
    ],
    ids=["unknown-mode", "not-well-formed", "json"],
)
def test_an_invalid_request_is_refused(
    service: Service, body: bytes, headers: dict[str, str] | None, code: ErrorCode
) -> None:
    assert_error(post(service, body, headers), code)
    assert service.store.training_jobs == {}


@pytest.mark.api
def test_recent_jobs_come_newest_first_with_their_outcome(service: Service) -> None:
    finished_job(service)
    service.clock.advance(timedelta(minutes=10))
    queued = document(post(service))

    root = document(service.client.get(QUEUE))

    jobs = root.findall(f"{{{NAMESPACE}}}TrainingJob")
    assert [job.get("id") for job in jobs] == [queued.get("id"), str(UUID(int=7))]
    assert jobs[0].find(f"{{{NAMESPACE}}}Result") is None
    assert dict(child(jobs[1], "Result").attrib) == {
        "runId": "run_20260918_101500_ab12",
        "gatePassed": "false",
        "errorCode": "STAGE_CHECK_FAILED",
    }
    assert jobs[1].get("finishedAt") == "2026-09-16T12:02:00Z"


@pytest.mark.api
def test_a_job_is_read_by_id(service: Service) -> None:
    created = document(post(service))

    response = service.client.get(f"{QUEUE}/{created.get('id')}")

    assert response.status_code == 200
    assert document(response).get("id") == created.get("id")


@pytest.mark.api
@pytest.mark.parametrize(
    "job_id",
    [
        "00000000-0000-0000-0000-000000000404",
        "not-a-uuid",
        "{00000000-0000-0000-0000-000000000007}",
    ],
    ids=["unknown", "not-a-uuid", "braces"],
)
def test_an_unknown_job_is_not_found(service: Service, job_id: str) -> None:
    finished_job(service)

    assert_error(service.client.get(f"{QUEUE}/{job_id}"), ErrorCode.TRAINING_JOB_NOT_FOUND)


@pytest.mark.api
def test_a_job_is_not_replaced_through_its_address(service: Service) -> None:
    created = document(post(service))

    response = service.client.put(f"{QUEUE}/{created.get('id')}", headers=XML)

    assert_error(response, ErrorCode.METHOD_NOT_ALLOWED)
    assert response.headers["allow"] == "GET"
