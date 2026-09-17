import base64
from datetime import timedelta

import pytest
from lxml import etree

from loadvision.api.xml.parser import NAMESPACE
from loadvision.domain.assessment import ReviewReason
from loadvision.errors import ErrorCode
from tests import image_files
from tests.api.service import (
    XML,
    Service,
    assert_error,
    child,
    document,
    post,
    request_body,
)
from tests.builders import make_assessment
from tests.port_contracts.unit_of_work import store_assessment

OTHER_PHOTO = base64.b64encode(image_files.jpeg(width=320)).decode()


def put_label(service: Service, assessment_id: str, verdict: str) -> None:
    body = f'<LabelRequest xmlns="{NAMESPACE}">{verdict}</LabelRequest>'.encode()
    response = service.client.put(
        f"/api/v1/assessments/{assessment_id}/label", content=body, headers=XML
    )
    assert response.status_code == 200


def summaries(root: etree._Element) -> list[dict[str, str]]:
    return [dict(item.attrib) for item in child(root, "History")]


@pytest.mark.api
def test_shipment_has_the_newest_assessment_as_current_and_the_whole_history(
    service: Service,
) -> None:
    older = document(post(service.client, request_body()))
    service.clock.advance(timedelta(hours=1))
    newer = document(post(service.client, request_body(image=OTHER_PHOTO)))
    put_label(service, older.get("id") or "", "<LoadPct>90</LoadPct>")
    put_label(service, newer.get("id") or "", "<Unusable/>")

    response = service.client.get("/api/v1/shipments/RP-1")

    assert response.status_code == 200
    root = document(response)
    assert root.get("id") == "RP-1"
    assert child(root, "History").get("count") == "2"
    assert summaries(root) == [
        {
            "id": newer.get("id"),
            "loadPct": child(newer, "LoadPct").text,
            "photoUnusable": "true",
            "modelVersion": "fake-1",
            "createdAt": "2026-09-16T13:00:00Z",
        },
        {
            "id": older.get("id"),
            "loadPct": "85.3",
            "confirmedLoadPct": "90.0",
            "modelVersion": "fake-1",
            "createdAt": "2026-09-16T12:00:00Z",
        },
    ]
    assert dict(child(root, "Current").attrib) == summaries(root)[0]


@pytest.mark.api
def test_assessment_with_review_reasons_needs_review(service: Service) -> None:
    with service.store.unit_of_work() as uow:
        store_assessment(uow, make_assessment(review_reasons=frozenset({ReviewReason.TOO_DARK})))
        uow.commit()

    root = document(service.client.get("/api/v1/shipments/RP-1"))

    assert child(root, "Current").get("needsReview") == "true"


@pytest.mark.api
def test_unknown_shipment_is_not_found(service: Service) -> None:
    assert_error(service.client.get("/api/v1/shipments/RP-404"), ErrorCode.SHIPMENT_NOT_FOUND)


@pytest.mark.api
@pytest.mark.parametrize("path", ["RP%201", "R" * 65, "RP-1%3F"], ids=["space", "long", "query"])
def test_malformed_shipment_number_is_rejected(service: Service, path: str) -> None:
    document(post(service.client, request_body()))
    response = service.client.get(f"/api/v1/shipments/{path}")
    assert_error(response, ErrorCode.INVALID_SHIPMENT_ID)


@pytest.mark.api
def test_shipment_is_only_read(service: Service) -> None:
    response = service.client.post("/api/v1/shipments/RP-1", content=b"", headers=XML)
    assert_error(response, ErrorCode.METHOD_NOT_ALLOWED)
    assert response.headers["allow"] == "GET"
