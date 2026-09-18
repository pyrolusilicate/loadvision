from uuid import UUID

import pytest
from httpx2 import Response

from loadvision.api.ui.pages import catalog_error
from loadvision.errors import ERROR_CATALOG, ErrorCode
from tests import image_files
from tests.api.service import PHOTO, PHOTO_SHA256, Service, child, document, make_service
from tests.api.ui_pages import get, page, sign_in, text_of

FIRST = f"/ui/assessments/{UUID(int=1)}"


def upload(
    service: Service, token: str | None, number: str = "RP-1", raw: bytes = PHOTO
) -> Response:
    form = {"shipment_id": number} | ({} if token is None else {"csrf": token})
    return service.client.post(
        "/ui/assess",
        data=form,
        files={"photo": ("kuzov.jpg", raw, "application/octet-stream")},
        follow_redirects=False,
    )


@pytest.mark.api
def test_signing_in_opens_the_photo_form(service: Service) -> None:
    sign_in(service)

    form = page(get(service, "/ui/assess"))

    assert form.xpath("//form[@enctype='multipart/form-data']/@action") == ["/ui/assess"]
    assert form.xpath("//input[@type='file']/@name") == ["photo"]
    assert form.xpath("//nav/a[@class='section on']/text()") == ["Оценить фото"]


@pytest.mark.api
def test_a_photo_is_assessed_stored_with_its_shipment_and_shown(service: Service) -> None:
    token = sign_in(service)

    response = upload(service, token)

    assert (response.status_code, response.headers["location"]) == (303, f"{FIRST}?result=new")
    [stored] = service.store.assessments.values()
    assert (stored.shipment_id.value, stored.image_sha256) == ("RP-1", PHOTO_SHA256)
    result = text_of(get(service, response.headers["location"]))
    assert "Фото оценено и сохранено в перевозке." in result
    api = child(document(service.client.get("/api/v1/shipments/RP-1")), "Current")
    assert api.get("id") == str(UUID(int=1))


@pytest.mark.api
def test_the_same_photo_again_opens_the_stored_assessment(service: Service) -> None:
    token = sign_in(service)
    upload(service, token)

    response = upload(service, token)

    assert response.headers["location"] == f"{FIRST}?result=reused"
    assert len(service.store.assessments) == 1
    assert "показана сохранённая оценка" in text_of(get(service, response.headers["location"]))


@pytest.mark.api
@pytest.mark.parametrize(
    ("raw", "number", "code"),
    [
        (image_files.truncated(PHOTO), "RP-1", ErrorCode.CORRUPTED_IMAGE),
        (image_files.heic_header(), "RP-1", ErrorCode.UNSUPPORTED_FORMAT),
        (PHOTO, "RP 1", ErrorCode.INVALID_SHIPMENT_ID),
    ],
    ids=["truncated-photo", "heic", "shipment-id-space"],
)
def test_a_refused_photo_returns_the_form_with_the_catalog_reason(
    service: Service, raw: bytes, number: str, code: ErrorCode
) -> None:
    token = sign_in(service)

    response = upload(service, token, number=number, raw=raw)

    status, message, hint = catalog_error(code)
    assert response.status_code == status
    assert f"{message} {hint}" in text_of(response)
    assert page(response).xpath("//input[@name='shipment_id']/@value") == [number]
    assert service.store.assessments == {}


@pytest.mark.api
def test_a_form_without_a_chosen_file_asks_for_one(service: Service) -> None:
    token = sign_in(service)
    # What a browser sends when no file was chosen: a file part with an empty name and no bytes.
    body = (
        f'--B\r\nContent-Disposition: form-data; name="csrf"\r\n\r\n{token}\r\n'
        '--B\r\nContent-Disposition: form-data; name="shipment_id"\r\n\r\nRP-1\r\n'
        '--B\r\nContent-Disposition: form-data; name="photo"; filename=""\r\n'
        "Content-Type: application/octet-stream\r\n\r\n\r\n--B--\r\n"
    )

    response = service.client.post(
        "/ui/assess",
        content=body.encode(),
        headers={"Content-Type": "multipart/form-data; boundary=B"},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert "Фото не выбрано." in text_of(response)
    assert service.store.assessments == {}


@pytest.mark.api
def test_a_photo_waits_for_the_model() -> None:
    service = make_service()
    token = sign_in(service)

    response = upload(service, token)

    spec = ERROR_CATALOG[ErrorCode.MODEL_NOT_READY]
    assert response.status_code == spec.http_status
    assert spec.message in text_of(response)


@pytest.mark.api
def test_a_photo_without_the_session_token_is_not_assessed(service: Service) -> None:
    sign_in(service)

    response = upload(service, None)

    assert response.status_code == 403
    assert service.store.assessments == {}


@pytest.mark.api
def test_an_unknown_shipment_offers_to_assess_its_first_photo(service: Service) -> None:
    sign_in(service)

    search = page(get(service, "/ui/shipments/RP-404"))

    assert search.xpath("//input[@name='number']/@value") == ["RP-404"]
    assert search.xpath("//p[@role='alert']/a/@href") == ["/ui/assess?number=RP-404"]


@pytest.mark.api
def test_the_form_takes_the_shipment_number_from_the_address(service: Service) -> None:
    sign_in(service)

    form = page(get(service, "/ui/assess?number=RP-1"))

    assert form.xpath("//input[@name='shipment_id']/@value") == ["RP-1"]
