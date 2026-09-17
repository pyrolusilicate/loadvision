"""What the routes need from the rest of the service, put together once at startup."""

from dataclasses import dataclass
from typing import cast

from starlette.requests import Request

from loadvision.application.assess_photo import AssessPhoto
from loadvision.application.get_assessment import GetAssessment
from loadvision.application.get_shipment import GetShipment
from loadvision.application.submit_label import SubmitLabel
from loadvision.ml.inference.active_model import ActiveModel


@dataclass(frozen=True, slots=True, kw_only=True)
class ApiServices:
    """Dependencies of the API; tests build them from fakes.

    Attributes:
        assess_photo: Use case behind `POST /api/v1/assessments`.
        get_assessment: Use case behind `GET /api/v1/assessments/{assessment_id}`.
        submit_label: Use case behind `PUT /api/v1/assessments/{assessment_id}/label`.
        get_shipment: Use case behind `GET /api/v1/shipments/{shipment_id}`.
        model: The model the service answers with.
        request_id_header: Header that carries the request id both ways.
        max_body_bytes: Largest accepted request body.
    """

    assess_photo: AssessPhoto
    get_assessment: GetAssessment
    submit_label: SubmitLabel
    get_shipment: GetShipment
    model: ActiveModel
    request_id_header: str
    max_body_bytes: int


def services_of(request: Request) -> ApiServices:
    """Returns the services of the application that handles the request."""
    return cast(ApiServices, request.app.state.services)


def request_id_of(request: Request) -> str:
    """Returns the id the request context middleware gave the request."""
    return cast(str, request.state.request_id)
