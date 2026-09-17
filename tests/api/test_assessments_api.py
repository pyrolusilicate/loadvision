import base64
import re
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from lxml import etree

from loadvision.api.app import create_app
from loadvision.api.middleware import REQUEST_ID_PATTERN
from loadvision.api.services import ApiServices
from loadvision.api.xml.parser import NAMESPACE, schema_bytes
from loadvision.application.assess_photo import AssessPhoto
from loadvision.application.ports import LoadPredictor
from loadvision.domain.shipment import ShipmentId
from loadvision.errors import ERROR_CATALOG, ErrorCode
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.validator import ImageLimits, prepare_image
from loadvision.ml.inference.active_model import ActiveModel
from tests import image_files
from tests.builders import T0
from tests.fakes.clock import FixedClock, SequentialIds
from tests.fakes.image_storage import InMemoryImageStorage
from tests.fakes.in_memory_db import InMemoryStore
from tests.fakes.predictor import FakePredictor

LIMITS = ImageLimits(
    max_bytes=2_000_000, max_pixels=1_000_000, min_side=256, media_types=frozenset(MediaType)
)
MAX_BODY_BYTES = 1_000_000
PHOTO = image_files.jpeg()
PHOTO_SHA256 = prepare_image(PHOTO, LIMITS).sha256
XML = {"Content-Type": "application/xml; charset=utf-8"}
SCHEMA = etree.XMLSchema(etree.fromstring(schema_bytes()))


class FailingImageStorage(InMemoryImageStorage):
    def save(self, image: PreparedImage) -> None:
        raise OSError("disk full at /srv/storage")


@dataclass
class Service:
    client: TestClient
    store: InMemoryStore
    storage: InMemoryImageStorage


def make_service(
    predictor: LoadPredictor | None = None, storage: InMemoryImageStorage | None = None
) -> Service:
    store = InMemoryStore()
    storage = InMemoryImageStorage() if storage is None else storage
    model = ActiveModel()
    if predictor is not None:
        model.activate(predictor)
    services = ApiServices(
        assess_photo=AssessPhoto(
            uow_factory=store.unit_of_work,
            storage=storage,
            limits=LIMITS,
            clock=FixedClock(T0),
            new_id=SequentialIds(),
        ),
        model=model,
        request_id_header="X-Request-ID",
        max_body_bytes=MAX_BODY_BYTES,
    )
    client = TestClient(create_app(services), raise_server_exceptions=False)
    return Service(client=client, store=store, storage=storage)


@pytest.fixture
def service() -> Service:
    return make_service(FakePredictor(loads={PHOTO_SHA256: 85.25}))


def request_body(shipment_id: str = "RP-1", image: str | None = None) -> bytes:
    data = base64.b64encode(PHOTO).decode() if image is None else image
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<AssessmentRequest xmlns="{NAMESPACE}">'
        f"<ShipmentId>{shipment_id}</ShipmentId><Image>{data}</Image>"
        "</AssessmentRequest>"
    ).encode()


def post(client: TestClient, body: bytes, headers: dict[str, str] | None = None) -> Response:
    return client.post(
        "/api/v1/assessments", content=body, headers=XML if headers is None else headers
    )


def document(response: Response) -> etree._Element:
    """Parses a response and checks it is XML valid against the published schema."""
    assert response.headers["content-type"] == "application/xml; charset=utf-8"
    root = etree.fromstring(response.content)
    SCHEMA.assertValid(root)
    return root


def child(root: etree._Element, name: str) -> etree._Element:
    found = root.find(f"{{{NAMESPACE}}}{name}")
    assert found is not None, name
    return found


def assert_error(response: Response, code: ErrorCode) -> None:
    root = document(response)
    spec = ERROR_CATALOG[code]
    assert response.status_code == spec.http_status
    assert root.get("code") == code.value
    assert child(root, "Message").text == spec.message
    assert child(root, "Hint").text == spec.hint
    assert root.get("requestId") == response.headers["X-Request-ID"]


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
def test_same_photo_again_returns_the_stored_assessment(service: Service) -> None:
    first = document(post(service.client, request_body()))
    response = post(service.client, request_body())

    assert response.status_code == 200
    again = document(response)
    assert again.get("reused") == "true"
    assert again.get("id") == first.get("id")
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


@pytest.mark.api
def test_schema_is_served(service: Service) -> None:
    response = service.client.get("/api/v1/schema.xsd")
    assert response.status_code == 200
    assert response.content == schema_bytes()
