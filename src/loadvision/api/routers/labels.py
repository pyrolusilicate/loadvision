"""Routes for confirmed labels."""

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response

from loadvision.api.path_params import uuid_from_path
from loadvision.api.services import services_of
from loadvision.api.xml.parser import check_content_type, parse_document
from loadvision.api.xml.response import xml_response
from loadvision.api.xml.schemas import label_document, read_label_request
from loadvision.application.submit_label import SubmitLabelRequest
from loadvision.domain.confirmed_label import LabelSource
from loadvision.errors import ErrorCode

router = APIRouter()


@router.put("/api/v1/assessments/{assessment_id}/label")
async def put_label(request: Request, assessment_id: str) -> Response:
    """Confirms the load on the photo of an assessment, or marks the photo unusable.

    Answers 200 with the stored label; a repeated request replaces it. The label belongs to the
    photo, so every assessment of the same photo shows it.
    """
    services = services_of(request)
    target = uuid_from_path(assessment_id, ErrorCode.ASSESSMENT_NOT_FOUND)
    check_content_type(request.headers.get("content-type"))
    message = read_label_request(parse_document(await request.body(), "LabelRequest"))
    result = await run_in_threadpool(
        services.submit_label,
        SubmitLabelRequest(
            assessment_id=target,
            load_pct=message.load_pct,
            source=LabelSource.API,
            author=message.author,
        ),
    )
    return xml_response(label_document(result.label, result.assessment.id))
