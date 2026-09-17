from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from lxml import etree

from loadvision.api.xml.parser import NAMESPACE
from loadvision.domain.assessment import LoadPercent
from loadvision.domain.confirmed_label import LabelSource
from loadvision.errors import ErrorCode
from tests.api.service import (
    PHOTO_SHA256,
    XML,
    Service,
    assert_error,
    child,
    document,
    post,
    request_body,
)
from tests.builders import T0


def label_body(verdict: str = "<LoadPct>90</LoadPct>", author: str | None = "operator-17") -> bytes:
    author_element = "" if author is None else f"<Author>{author}</Author>"
    return (f'<LabelRequest xmlns="{NAMESPACE}">{verdict}{author_element}</LabelRequest>').encode()


def put_label(
    client: TestClient,
    assessment_id: str,
    body: bytes,
    headers: dict[str, str] | None = None,
) -> Response:
    return client.put(
        f"/api/v1/assessments/{assessment_id}/label",
        content=body,
        headers=XML if headers is None else headers,
    )


@pytest.fixture
def assessment_id(service: Service) -> str:
    root = document(post(service.client, request_body()))
    return root.get("id") or ""


@pytest.mark.api
def test_load_is_confirmed_on_the_photo_of_the_assessment(
    service: Service, assessment_id: str
) -> None:
    response = put_label(service.client, assessment_id, label_body())

    assert response.status_code == 200
    root = document(response)
    assert etree.QName(root).localname == "Label"
    assert (root.get("assessmentId"), root.get("imageSha256")) == (assessment_id, PHOTO_SHA256)
    assert child(root, "LoadPct").text == "90.0"
    assert child(root, "Author").text == "operator-17"
    assert child(root, "UpdatedAt").text == "2026-09-16T12:00:00Z"
    label = service.store.labels[PHOTO_SHA256]
    assert (label.load_pct, label.source, label.author) == (
        LoadPercent(90.0),
        LabelSource.API,
        "operator-17",
    )


@pytest.mark.api
def test_unusable_photo_replaces_an_earlier_load(service: Service, assessment_id: str) -> None:
    put_label(service.client, assessment_id, label_body())
    service.clock.advance(timedelta(minutes=5))

    root = document(put_label(service.client, assessment_id, label_body("<Unusable/>", None)))

    assert root.find(f"{{{NAMESPACE}}}Unusable") is not None
    assert root.find(f"{{{NAMESPACE}}}LoadPct") is None
    assert root.find(f"{{{NAMESPACE}}}Author") is None
    assert child(root, "UpdatedAt").text == "2026-09-16T12:05:00Z"
    label = service.store.labels[PHOTO_SHA256]
    assert (label.load_pct, label.created_at) == (None, T0)


@pytest.mark.api
@pytest.mark.parametrize(("sent", "written"), [("-0", "0.0"), (" 42.5 ", "42.5"), ("100", "100.0")])
def test_load_is_written_back_in_one_form(
    service: Service, assessment_id: str, sent: str, written: str
) -> None:
    root = document(
        put_label(service.client, assessment_id, label_body(f"<LoadPct>{sent}</LoadPct>"))
    )
    assert child(root, "LoadPct").text == written


@pytest.mark.api
def test_blank_author_is_stored_as_unknown(service: Service, assessment_id: str) -> None:
    root = document(put_label(service.client, assessment_id, label_body(author="   ")))
    assert root.find(f"{{{NAMESPACE}}}Author") is None
    assert service.store.labels[PHOTO_SHA256].author is None


@pytest.mark.api
def test_assessment_shows_the_label_of_its_photo(service: Service, assessment_id: str) -> None:
    put_label(service.client, assessment_id, label_body())

    response = service.client.get(f"/api/v1/assessments/{assessment_id}")

    assert child(document(response), "ConfirmedLabel").attrib == {
        "loadPct": "90.0",
        "author": "operator-17",
        "updatedAt": "2026-09-16T12:00:00Z",
    }


@pytest.mark.api
def test_assessment_shows_that_its_photo_is_unusable(service: Service, assessment_id: str) -> None:
    put_label(service.client, assessment_id, label_body("<Unusable/>", None))

    response = service.client.get(f"/api/v1/assessments/{assessment_id}")

    assert child(document(response), "ConfirmedLabel").attrib == {
        "unusable": "true",
        "updatedAt": "2026-09-16T12:00:00Z",
    }


@pytest.mark.api
@pytest.mark.parametrize(
    "body",
    [
        label_body("<LoadPct>100.5</LoadPct>"),
        label_body("<LoadPct>90.25</LoadPct>"),
        label_body("<LoadPct>90</LoadPct><Unusable/>"),
        label_body(""),
        label_body("<Unusable>yes</Unusable>"),
        label_body(author="a" * 129),
    ],
    ids=["over-100", "two-decimals", "both", "neither", "unusable-with-text", "long-author"],
)
def test_invalid_label_is_rejected(service: Service, assessment_id: str, body: bytes) -> None:
    assert_error(put_label(service.client, assessment_id, body), ErrorCode.INVALID_XML)
    assert service.store.labels == {}


@pytest.mark.api
def test_label_must_be_xml(service: Service, assessment_id: str) -> None:
    response = put_label(
        service.client, assessment_id, label_body(), {"Content-Type": "application/json"}
    )
    assert_error(response, ErrorCode.INVALID_CONTENT_TYPE)


@pytest.mark.api
@pytest.mark.parametrize(
    "assessment_id",
    [
        "00000000-0000-0000-0000-000000000404",
        "not-a-uuid",
        "urn:uuid:00000000-0000-0000-0000-000000000001",
        "{00000000-0000-0000-0000-000000000001}",
        "00000000000000000000000000000001",
    ],
    ids=["unknown", "not-a-uuid", "urn-prefix", "braces", "no-hyphens"],
)
def test_label_for_an_unknown_or_malformed_id_is_not_found(
    service: Service, assessment_id: str
) -> None:
    document(post(service.client, request_body()))
    assert_error(
        put_label(service.client, assessment_id, label_body()), ErrorCode.ASSESSMENT_NOT_FOUND
    )
    assert service.store.labels == {}


@pytest.mark.api
def test_label_is_only_put(service: Service, assessment_id: str) -> None:
    response = service.client.get(f"/api/v1/assessments/{assessment_id}/label")
    assert_error(response, ErrorCode.METHOD_NOT_ALLOWED)
    assert response.headers["allow"] == "PUT"
