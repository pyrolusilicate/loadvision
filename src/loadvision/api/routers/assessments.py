"""Routes for photo assessments."""

from uuid import UUID

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response

from loadvision.api.services import request_id_of, services_of
from loadvision.api.xml.parser import check_content_type, parse_document
from loadvision.api.xml.response import xml_response
from loadvision.api.xml.schemas import assessment_document, read_assessment_request
from loadvision.application.assess_photo import AssessPhotoRequest
from loadvision.errors import ErrorCode, LoadvisionError

router = APIRouter()


def assessment_id_from_path(text: str) -> UUID:
    """Reads an assessment id from a URL path.

    Only the canonical form, in any letter case, is accepted: `UUID` also parses braces and a
    ``urn:uuid:`` prefix, which would give one assessment several addresses.

    Raises:
        LoadvisionError: `ASSESSMENT_NOT_FOUND` for anything else; no assessment has such an id.
    """
    try:
        assessment_id = UUID(text)
    except ValueError:
        assessment_id = None
    if assessment_id is None or str(assessment_id) != text.lower():
        raise LoadvisionError(ErrorCode.ASSESSMENT_NOT_FOUND, "malformed assessment id")
    return assessment_id


@router.post("/api/v1/assessments")
async def create_assessment(request: Request) -> Response:
    """Assesses a photo and attaches it to a shipment.

    Answers 201 with a new assessment, or 200 with the stored one when the same photo was
    already sent for this shipment. Errors come back as XML `Error` with the catalog status.
    """
    services = services_of(request)
    check_content_type(request.headers.get("content-type"))
    message = read_assessment_request(parse_document(await request.body(), "AssessmentRequest"))
    predictor = services.model.current()
    if predictor is None:
        raise LoadvisionError(ErrorCode.MODEL_NOT_READY, services.model.failure)
    result = await run_in_threadpool(
        services.assess_photo,
        AssessPhotoRequest(
            shipment_id=message.shipment_id,
            image=message.image,
            request_id=request_id_of(request),
        ),
        predictor,
    )
    document = assessment_document(result.assessment, result.image, reused=result.reused)
    return xml_response(document, 200 if result.reused else 201)


@router.get("/api/v1/assessments/{assessment_id}")
def get_assessment(request: Request, assessment_id: str) -> Response:
    """Answers 200 with the assessment and the label of its photo, 404 for an unknown id."""
    details = services_of(request).get_assessment(assessment_id_from_path(assessment_id))
    return xml_response(assessment_document(details.assessment, details.image, label=details.label))
