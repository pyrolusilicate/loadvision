import logging
import re

import pytest

from loadvision.api.ui.pages import catalog_error
from loadvision.domain.shipment import SHIPMENT_ID_PATTERN
from loadvision.errors import ERROR_CATALOG, ErrorCode, LoadvisionError
from loadvision.imaging.prepared_image import PreparedImage
from tests.api.service import PHOTO_SHA256, Service, make_service
from tests.api.ui_pages import get, page, sign_in, text_of


class BrokenStorage:
    """Photo storage that fails every read with the given error."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def save(self, image: PreparedImage) -> None:
        raise self._error

    def load(self, sha256: str) -> bytes | None:
        raise self._error


@pytest.mark.api
@pytest.mark.parametrize(("path", "status"), [("/ui/nope", 404), ("/ui/?status=maybe", 404)])
def test_a_page_that_does_not_exist_says_so_in_html(
    service: Service, path: str, status: int
) -> None:
    sign_in(service)

    response = get(service, path)

    assert response.status_code == status
    assert "Такой страницы нет." in text_of(response)


@pytest.mark.api
def test_a_form_address_opened_as_a_link_lists_the_allowed_method(service: Service) -> None:
    sign_in(service)

    response = get(service, "/ui/logout")

    assert response.status_code == 405
    assert response.headers["allow"] == "POST"
    assert "Такой страницы нет." in text_of(response)


@pytest.mark.api
def test_an_error_page_names_the_request_for_the_log(service: Service) -> None:
    response = service.client.get("/ui/nope", headers={"X-Request-ID": "req-42"})

    assert response.headers["x-request-id"] == "req-42"
    assert page(response).xpath("//code/text()") == ["req-42"]


@pytest.mark.api
def test_an_unexpected_error_is_a_page_and_one_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = make_service(photos=BrokenStorage(OSError("disk is gone")))
    sign_in(service)

    with caplog.at_level(logging.ERROR):
        response = service.client.get(
            f"/ui/photos/{PHOTO_SHA256}", headers={"X-Request-ID": "req-500"}
        )

    assert response.status_code == 500
    assert ERROR_CATALOG[ErrorCode.INTERNAL_ERROR].message in text_of(response)
    assert response.headers["x-request-id"] == "req-500"
    [record] = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert getattr(record, "request_id", None) == "req-500"
    assert record.exc_info is not None


@pytest.mark.api
def test_an_error_without_an_http_status_is_answered_as_internal() -> None:
    failing = BrokenStorage(LoadvisionError(ErrorCode.STAGE_CHECK_FAILED, "not for clients"))
    service = make_service(photos=failing)
    sign_in(service)

    response = get(service, f"/ui/photos/{PHOTO_SHA256}")

    assert response.status_code == 500
    assert ERROR_CATALOG[ErrorCode.INTERNAL_ERROR].message in text_of(response)


@pytest.mark.api
@pytest.mark.parametrize("sha256", ["A" * 64, ".." + "0" * 62, "0" * 65])
def test_a_malformed_photo_hash_never_reaches_the_storage(sha256: str) -> None:
    service = make_service(photos=BrokenStorage(OSError("storage was asked")))
    sign_in(service)

    assert get(service, f"/ui/photos/{sha256}").status_code == 404


@pytest.mark.api
@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.INVALID_SHIPMENT_ID,
        ErrorCode.PAYLOAD_TOO_LARGE,
        ErrorCode.UNSUPPORTED_FORMAT,
        ErrorCode.CORRUPTED_IMAGE,
        ErrorCode.SHIPMENT_NOT_FOUND,
        ErrorCode.ASSESSMENT_NOT_FOUND,
        ErrorCode.MODEL_VERSION_NOT_FOUND,
        ErrorCode.MODEL_VERSION_STATUS_CONFLICT,
        ErrorCode.TRAINING_ALREADY_RUNNING,
        ErrorCode.MODEL_NOT_READY,
        ErrorCode.MODEL_INFERENCE_FAILED,
        ErrorCode.STAGE_CHECK_FAILED,
        ErrorCode.INTERNAL_ERROR,
    ],
)
def test_a_page_hint_speaks_to_a_person_not_to_an_api_client(code: ErrorCode) -> None:
    _, _, hint = catalog_error(code)

    for api_word in ("/api/", "GET ", "POST ", "requestId", "/health", "XML"):
        assert api_word not in hint


@pytest.mark.api
def test_a_malformed_number_is_explained_in_words() -> None:
    status, message, hint = catalog_error(ErrorCode.INVALID_SHIPMENT_ID)

    assert (status, message) == (400, ERROR_CATALOG[ErrorCode.INVALID_SHIPMENT_ID].message)
    # The example in the hint is a valid number and the limit is the one of the pattern.
    assert re.fullmatch(SHIPMENT_ID_PATTERN, "RP-2026-000123")
    assert "RP-2026-000123" in hint
    assert re.fullmatch(SHIPMENT_ID_PATTERN, "R" * 64)
    assert not re.fullmatch(SHIPMENT_ID_PATTERN, "R" * 65)
    assert "до 64 знаков" in hint
