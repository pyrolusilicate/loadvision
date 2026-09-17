"""Health check route."""

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response

from loadvision.api.services import services_of
from loadvision.api.xml.response import xml_response
from loadvision.api.xml.schemas import health_document
from loadvision.errors import ErrorCode, LoadvisionError

router = APIRouter()


@router.get("/health")
def health(request: Request) -> Response:
    """Answers 200 with the model version once a model is loaded, 503 `MODEL_NOT_READY` before."""
    model = services_of(request).model
    predictor = model.current()
    if predictor is None:
        raise LoadvisionError(ErrorCode.MODEL_NOT_READY, model.failure)
    return xml_response(health_document(predictor.version))
