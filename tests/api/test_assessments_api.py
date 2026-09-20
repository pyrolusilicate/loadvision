import base64
import logging
import re
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import pytest
from lxml import etree

from loadvision.api.errors import error_response
from loadvision.api.middleware import REQUEST_ID_PATTERN
from loadvision.api.xml.parser import NAMESPACE, schema_bytes
from loadvision.domain.shipment import ShipmentId
from loadvision.errors import ErrorCode
from loadvision.imaging.prepared_image import PreparedImage
from tests import image_files
from tests.api.service import (
    MAX_BODY_BYTES,
    PHOTO,
    PHOTO_SHA256,
    SCHEMA,
    XML,
    Service,
    assert_error,
    child,
    document,
    make_service,
    post,
    request_body,
)
from tests.builders import make_assessment
from tests.fakes.image_storage import InMemoryImageStorage
from tests.fakes.predictor import FakePredictor
from tests.port_contracts.unit_of_work import store_assessment


class FailingImageStorage(InMemoryImageStorage):
    def save(self, image: PreparedImage) -> None:
        raise OSError("disk full at /srv/storage")


@pytest.mark.api
def test_new_photo_is_assessed_and_stored(service: Service) -> None:
    response = post(service.client, request_body())

    assert response.status_code == 201
    root = document(response)
    assert etree.QName(root).localname == "Assessment"
    assert (root.get("shipmentId"), root.get("reused")) == ("RP-1", "false")
    assert child(root, "LoadPct").text == "85.3"
    assert child(root, "Image").attrib == {
        "sha256": PHOTO_SHA256,
        "mediaType": "image/jpeg",
        "width": "300",
        "height": "400",
    }
    assert child(root, "Model").get("version") == "fake-1"
    assert child(root, "CreatedAt").text == "2026-09-16T12:00:00Z"
    assert re.fullmatch(REQUEST_ID_PATTERN, response.headers["X-Request-ID"])
    assert service.storage.files == {PHOTO_SHA256: PHOTO}
    [stored] = service.store.assessments.values()
    assert str(stored.id) == root.get("id")
    assert stored.request_id == response.headers["X-Request-ID"]


@pytest.mark.api
def test_stored_assessment_is_read_by_id(service: Service) -> None:
    created = document(post(service.client, request_body()))

    response = service.client.get(f"/api/v1/assessments/{created.get('id')}")

    assert response.status_code == 200
    root = document(response)
    assert "reused" not in root.attrib
    assert root.find(f"{{{NAMESPACE}}}ConfirmedLabel") is None
    assert etree.tostring(child(root, "Image")) == etree.tostring(child(created, "Image"))
    assert child(root, "LoadPct").text == "85.3"


@pytest.mark.api
def test_assessment_id_in_upper_case_is_the_same_assessment(service: Service) -> None:
    assessment = make_assessment(id=UUID("3f6c2a4e-7b1d-4c55-9a0e-5d2f8e41c7aa"))
    with service.store.unit_of_work() as uow:
        store_assessment(uow, assessment)
        uow.commit()

    response = service.client.get(f"/api/v1/assessments/{str(assessment.id).upper()}")

    assert document(response).get("id") == str(assessment.id)


@pytest.mark.api
@pytest.mark.parametrize("assessment_id", ["00000000-0000-0000-0000-000000000404", "not-a-uuid"])
def test_unknown_or_malformed_assessment_is_not_found(service: Service, assessment_id: str) -> None:
    response = service.client.get(f"/api/v1/assessments/{assessment_id}")
    assert_error(response, ErrorCode.ASSESSMENT_NOT_FOUND)


@pytest.mark.api
def test_assessment_is_not_replaced_through_its_address(service: Service) -> None:
    created = document(post(service.client, request_body()))
    response = service.client.put(f"/api/v1/assessments/{created.get('id')}", headers=XML)
    assert_error(response, ErrorCode.METHOD_NOT_ALLOWED)
    assert response.headers["allow"] == "GET"


@pytest.mark.api
def test_same_photo_again_returns_the_stored_assessment(service: Service) -> None:
    first = document(post(service.client, request_body()))
    response = post(service.client, request_body())

    assert response.status_code == 200
    again = document(response)
    assert again.get("reused") == "true"
    assert again.get("id") == first.get("id")
    assert len(service.store.assessments) == 1


@pytest.mark.api
def test_same_photo_after_a_model_change_is_assessed_again(service: Service) -> None:
    first = document(post(service.client, request_body()))

    service.model.activate(FakePredictor(version="fake-2", loads={PHOTO_SHA256: 30.0}))
    response = post(service.client, request_body())

    assert response.status_code == 201
    again = document(response)
    assert again.get("reused") == "false"
    assert again.get("id") == first.get("id")
    assert child(again, "LoadPct").text == "30.0"
    assert child(again, "Model").get("version") == "fake-2"
    assert len(service.store.assessments) == 1


@pytest.mark.api
def test_base64_wrapped_at_76_characters_is_accepted(service: Service) -> None:
    data = base64.b64encode(PHOTO).decode()
    wrapped = "\n".join(data[start : start + 76] for start in range(0, len(data), 76))
    assert post(service.client, request_body(image=wrapped)).status_code == 201


@pytest.mark.api
def test_client_request_id_is_kept(service: Service) -> None:
    headers = {**XML, "X-Request-ID": "crm-2026.09.17_42"}
    response = post(service.client, request_body(), headers)
    assert response.headers["X-Request-ID"] == "crm-2026.09.17_42"
    [stored] = service.store.assessments.values()
    assert stored.request_id == "crm-2026.09.17_42"


@pytest.mark.api
@pytest.mark.parametrize("sent", ["has space", "a" * 129, "id/with/slash", "<b>", ""])
def test_request_id_outside_the_format_is_replaced(service: Service, sent: str) -> None:
    response = post(service.client, request_body(), {**XML, "X-Request-ID": sent})
    issued = response.headers["X-Request-ID"]
    assert issued != sent
    assert re.fullmatch(REQUEST_ID_PATTERN, issued)


@pytest.mark.api
@pytest.mark.parametrize(
    "content_type",
    ["application/xml", 'Application/XML; charset="UTF-8"', "application/xml;charset=utf8"],
)
def test_xml_content_types_are_accepted(service: Service, content_type: str) -> None:
    response = post(service.client, request_body(), {"Content-Type": content_type})
    assert response.status_code == 201


@pytest.mark.api
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Content-Type": "text/xml"},
        {"Content-Type": "application/json"},
        {"Content-Type": "application/xml; charset=windows-1251"},
    ],
    ids=["missing", "text-xml", "json", "windows-1251"],
)
def test_other_content_types_are_rejected(service: Service, headers: dict[str, str]) -> None:
    assert_error(post(service.client, request_body(), headers), ErrorCode.INVALID_CONTENT_TYPE)
    assert service.store.assessments == {}


@pytest.mark.api
@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"<AssessmentRequest", ErrorCode.INVALID_XML),
        (b"", ErrorCode.INVALID_XML),
        (f'<Health xmlns="{NAMESPACE}" modelVersion="v1"/>'.encode(), ErrorCode.INVALID_XML),
        (
            request_body().replace(b"</AssessmentRequest>", b"<Note>x</Note></AssessmentRequest>"),
            ErrorCode.INVALID_XML,
        ),
        (request_body(shipment_id="RP 1"), ErrorCode.INVALID_SHIPMENT_ID),
        (request_body(shipment_id="R" * 65), ErrorCode.INVALID_SHIPMENT_ID),
        (
            request_body(image="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ=="),
            ErrorCode.INVALID_ENCODING,
        ),
        (request_body(image="@@@@"), ErrorCode.INVALID_ENCODING),
        (
            request_body(image=base64.b64encode(image_files.heic_header()).decode()),
            ErrorCode.UNSUPPORTED_FORMAT,
        ),
        (
            request_body(image=base64.b64encode(image_files.truncated(PHOTO)).decode()),
            ErrorCode.CORRUPTED_IMAGE,
        ),
    ],
    ids=[
        "not-well-formed",
        "empty",
        "other-root",
        "unknown-element",
        "shipment-id-space",
        "shipment-id-long",
        "data-prefix",
        "not-base64",
        "heic",
        "truncated-photo",
    ],
)
def test_invalid_request_gets_its_error_code(
    service: Service, body: bytes, code: ErrorCode
) -> None:
    assert_error(post(service.client, body), code)
    assert service.store.assessments == {}


@pytest.mark.api
def test_doctype_with_an_external_entity_is_rejected_without_reading_the_file(
    service: Service,
) -> None:
    body = (
        '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        f'<AssessmentRequest xmlns="{NAMESPACE}"><ShipmentId>&xxe;</ShipmentId>'
        "<Image>/9j/4AAQ</Image></AssessmentRequest>"
    ).encode()
    response = post(service.client, body)
    assert_error(response, ErrorCode.INVALID_XML)
    assert "root:" not in response.text


@pytest.mark.api
def test_body_over_the_limit_is_rejected_before_reading(service: Service) -> None:
    response = post(service.client, request_body(image="A" * MAX_BODY_BYTES))
    assert_error(response, ErrorCode.PAYLOAD_TOO_LARGE)


@pytest.mark.api
def test_streamed_body_over_the_limit_is_rejected(service: Service) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(3):
            yield b"A" * (MAX_BODY_BYTES // 2)

    response = service.client.post("/api/v1/assessments", content=chunks(), headers=XML)
    assert_error(response, ErrorCode.PAYLOAD_TOO_LARGE)


@pytest.mark.api
def test_without_a_model_assessment_and_health_answer_not_ready() -> None:
    service = make_service(predictor=None)
    assert_error(post(service.client, request_body()), ErrorCode.MODEL_NOT_READY)
    assert_error(service.client.get("/health"), ErrorCode.MODEL_NOT_READY)
    assert service.store.assessments == {}


@pytest.mark.api
def test_model_failure_is_reported(service: Service) -> None:
    failing = make_service(FakePredictor(error=RuntimeError("CUDA out of memory")))
    assert_error(post(failing.client, request_body()), ErrorCode.MODEL_INFERENCE_FAILED)


@pytest.mark.api
def test_a_code_without_an_http_status_is_answered_as_internal() -> None:
    response = error_response(ErrorCode.STAGE_CHECK_FAILED, "req-1")

    root = etree.fromstring(response.body)
    SCHEMA.assertValid(root)
    assert (response.status_code, root.get("code")) == (500, "INTERNAL_ERROR")


@pytest.mark.api
def test_unexpected_error_hides_the_details() -> None:
    service = make_service(FakePredictor(), storage=FailingImageStorage())
    response = post(service.client, request_body())
    assert_error(response, ErrorCode.INTERNAL_ERROR)
    assert "disk full" not in response.text
    assert ShipmentId("RP-1") not in service.store.shipments


@pytest.mark.api
def test_health_reports_the_model_version(service: Service) -> None:
    response = service.client.get("/health")
    assert response.status_code == 200
    assert document(response).get("modelVersion") == "fake-1"


@pytest.mark.api
def test_unknown_route_and_method_answer_in_xml(service: Service) -> None:
    assert_error(service.client.get("/api/v1/nothing"), ErrorCode.ROUTE_NOT_FOUND)
    response = service.client.get("/api/v1/assessments")
    assert_error(response, ErrorCode.METHOD_NOT_ALLOWED)
    assert response.headers["allow"] == "POST"


def finished_requests(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    """Fields of the `request.finished` lines, extras included."""
    return [vars(r) for r in caplog.records if vars(r).get("event") == "request.finished"]


@pytest.mark.api
def test_every_request_ends_with_one_log_line(
    service: Service, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    response = post(service.client, request_body())

    [line] = finished_requests(caplog)
    assert (line["method"], line["path"], line["status"]) == ("POST", "/api/v1/assessments", 201)
    assert line["request_id"] == response.headers["X-Request-ID"]
    assert line["duration_ms"] >= 0
    # The photo never reaches the log.
    assert base64.b64encode(PHOTO).decode()[:40] not in caplog.text


@pytest.mark.api
def test_a_refused_body_is_logged_with_its_status(
    service: Service, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    post(service.client, request_body(image="A" * MAX_BODY_BYTES))

    [line] = finished_requests(caplog)
    assert line["status"] == 413


@pytest.mark.api
def test_an_unexpected_error_is_logged_as_a_server_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    service = make_service(FakePredictor(), storage=FailingImageStorage())

    post(service.client, request_body())

    [line] = finished_requests(caplog)
    assert line["status"] == 500


@pytest.mark.api
def test_schema_is_served(service: Service) -> None:
    response = service.client.get("/api/v1/schema.xsd")
    assert response.status_code == 200
    assert response.content == schema_bytes()
