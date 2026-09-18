"""Routes for the retraining queue."""

from uuid import UUID

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response

from loadvision.api.services import services_of
from loadvision.api.xml.parser import check_content_type, parse_document
from loadvision.api.xml.response import xml_response
from loadvision.api.xml.training import (
    read_training_job_request,
    training_job_document,
    training_jobs_document,
)
from loadvision.errors import ErrorCode, LoadvisionError

router = APIRouter()


@router.post("/api/v1/training-jobs")
async def create_training_job(request: Request) -> Response:
    """Queues a retraining on the confirmed labels.

    Answers 201 with the queued job, or 409 `TRAINING_ALREADY_RUNNING` while another job is
    queued or running.
    """
    services = services_of(request)
    check_content_type(request.headers.get("content-type"))
    mode = read_training_job_request(parse_document(await request.body(), "TrainingJobRequest"))
    job = await run_in_threadpool(services.start_training, mode)
    return xml_response(training_job_document(job), 201)


@router.get("/api/v1/training-jobs")
async def list_training_jobs(request: Request) -> Response:
    """Answers 200 with the recent jobs, newest first."""
    jobs = await run_in_threadpool(services_of(request).list_training_jobs)
    return xml_response(training_jobs_document(jobs))


@router.get("/api/v1/training-jobs/{job_id}")
async def get_training_job(request: Request, job_id: str) -> Response:
    """Answers 200 with the job, or 404 `TRAINING_JOB_NOT_FOUND` for an unknown id."""
    job = await run_in_threadpool(services_of(request).get_training_job, _job_id(job_id))
    return xml_response(training_job_document(job))


def _job_id(text: str) -> UUID:
    try:
        job_id = UUID(text)
    except ValueError:
        job_id = None
    if job_id is None or str(job_id) != text.lower():
        raise LoadvisionError(ErrorCode.TRAINING_JOB_NOT_FOUND, "malformed job id")
    return job_id
