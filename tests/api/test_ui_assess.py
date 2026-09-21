from uuid import UUID

import pytest
from httpx2 import Response

from loadvision.api.ui.pages import catalog_error
from loadvision.api.ui.templates import REVIEW_NOTES
from loadvision.domain.assessment import ReviewReason
from loadvision.errors import ERROR_CATALOG, ErrorCode
from tests import image_files
from tests.api.service import PHOTO, PHOTO_SHA256, Service, child, document, make_service
from tests.api.ui_pages import get, page, sign_in, text_of
from tests.fakes.predictor import FakePredictor

FIRST = f"/ui/assessments/{UUID(int=1)}"


def upload(
    service: Service,
    token: str | None,
    number: str = "RP-1",
    raw: bytes = PHOTO,
    declared: str = "",
) -> Response:
    form = {"shipment_id": number, "declared_load_pct": declared}
    form |= {} if token is None else {"csrf": token}
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
def test_a_photo_shot_in_the_dark_opens_with_the_reason_to_check_it(service: Service) -> None:
    token = sign_in(service)

    response = upload(service, token, raw=image_files.dark_jpeg())

    card = text_of(get(service, response.headers["location"]))
    assert f"Прогноз может быть неверным. {REVIEW_NOTES[ReviewReason.TOO_DARK]}" in card
    # The list runs the name into the value: "Яркость кадра" then "5 из 255".
    assert "Яркость кадра5 из 255" in card


@pytest.mark.api
def test_a_new_model_assesses_the_photo_again_in_place_of_the_stored_estimate(
    service: Service,
) -> None:
    token = sign_in(service)
    upload(service, token)

    service.model.activate(FakePredictor(version="fake-2", loads={PHOTO_SHA256: 30.0}))
    response = upload(service, token)

    assert response.headers["location"] == f"{FIRST}?result=again&from=fake-1"
    card = text_of(get(service, response.headers["location"]))
    assert "Новая модель пересчитала оценку, которую раньше дала модель fake-1." in card
    assert "30.0% прогноз модели" in card
    assert len(service.store.assessments) == 1


@pytest.mark.api
def test_a_made_up_model_version_in_the_address_is_left_out(service: Service) -> None:
    token = sign_in(service)
    upload(service, token)

    card = text_of(get(service, f"{FIRST}?result=again&from=%3Cwhatever%20you%20like%3E"))

    assert "Новая модель пересчитала оценку." in card
    assert "whatever you like" not in card


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

    # The status is the one the API answers with; the page says it in words an operator can act
    # on, instead of blaming the photo that was just picked.
    assert response.status_code == ERROR_CATALOG[ErrorCode.MODEL_NOT_READY].http_status
    assert "Модель загружается." in text_of(response)


@pytest.mark.api
def test_the_form_says_the_model_is_loading_before_anything_is_sent() -> None:
    service = make_service()
    sign_in(service)

    assert "Модель загружается." in text_of(get(service, "/ui/assess"))


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


@pytest.mark.api
def test_a_load_below_the_document_is_called_out(service: Service) -> None:
    """The fake model answers 85.3 for this photo, so a document saying 30 is far off."""
    token = sign_in(service)

    upload(service, token, declared="30")

    shown = get(service, FIRST)
    note = page(shown).xpath("//div[contains(@class,'warn')]/p/strong/text()")
    assert note == ["Загружено больше, чем по документам"]
    assert "55.3 п.п." in text_of(shown)


@pytest.mark.api
def test_a_small_difference_is_not_called_a_discrepancy(service: Service) -> None:
    token = sign_in(service)

    upload(service, token, declared="80")

    card = page(get(service, FIRST))
    assert card.xpath("//div[contains(@class,'ok')]/p/strong/text()") == ["Совпадает с документами"]


@pytest.mark.api
def test_a_document_number_outside_the_range_is_refused(service: Service) -> None:
    token = sign_in(service)

    refused = upload(service, token, declared="140")

    assert refused.status_code == 400
    assert "Загрузка по документам указана неверно." in text_of(refused)


@pytest.mark.api
def test_the_same_photo_in_another_shipment_says_where_the_estimate_came_from(
    service: Service,
) -> None:
    """Two cards with the same number look like two predictions; one of them was free."""
    token = sign_in(service)
    upload(service, token, number="RP-1")

    again = upload(service, token, number="RP-2")

    # The note belongs to the answer of the form, so the page is opened the way it redirects.
    second = get(service, again.headers["location"])
    assert "уже оценивалось в перевозке RP-1" in text_of(second)
    assert page(second).xpath("//a[@href='/ui/shipments/RP-1']") != []


@pytest.mark.api
def test_a_photo_nobody_sent_before_says_nothing_about_another_shipment(
    service: Service,
) -> None:
    token = sign_in(service)

    first = upload(service, token)

    assert "уже оценивалось" not in text_of(get(service, first.headers["location"]))
