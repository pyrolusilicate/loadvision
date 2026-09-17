"""Routes for shipment lookups."""

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response

from loadvision.api.services import services_of
from loadvision.api.xml.response import xml_response
from loadvision.api.xml.schemas import shipment_document

router = APIRouter()


@router.get("/api/v1/shipments/{shipment_id}")
def get_shipment(request: Request, shipment_id: str) -> Response:
    """Answers 200 with the current assessment and the history, newest first.

    Errors: 400 `INVALID_SHIPMENT_ID` for a malformed number, 404 `SHIPMENT_NOT_FOUND` for a
    shipment without assessments.
    """
    return xml_response(shipment_document(services_of(request).get_shipment(shipment_id)))
