from uuid import UUID

import pytest

from loadvision.domain.confirmed_label import LabelSource
from loadvision.errors import ERROR_CATALOG, ErrorCode
from tests.api.service import (
    PHOTO,
    PHOTO_SHA256,
    Service,
    child,
    document,
    make_service,
    post,
    request_body,
)
from tests.api.ui_pages import LOGIN, get, page, sign_in, submit, text_of
from tests.fakes.predictor import FakePredictor

ASSESSMENT_ID = str(UUID(int=1))
ASSESSMENT = f"/ui/assessments/{ASSESSMENT_ID}"


def assess(service: Service) -> None:
    assert post(service.client, request_body()).status_code == 201


@pytest.mark.api
def test_the_queue_shows_photos_nobody_confirmed(service: Service) -> None:
    sign_in(service)
    assess(service)

    queue = page(get(service, "/ui/"))

    assert queue.xpath("//a[@class='tile']/@href") == [ASSESSMENT]
    assert queue.xpath("//a[@class='tile']//img/@src") == [f"/ui/photos/{PHOTO_SHA256}"]
    assert queue.xpath("//nav/a[@class='section on']/text()") == ["Очередь"]


@pytest.mark.api
def test_a_confirmed_photo_leaves_the_queue_but_stays_in_the_full_list(
    service: Service,
) -> None:
    token = sign_in(service)
    assess(service)

    submit(service, f"{ASSESSMENT}/label", csrf=token, action="confirm", load_pct="60")

    assert "Всё проверено" in text_of(get(service, "/ui/"))
    assert "подтверждено 60.0%" in text_of(get(service, "/ui/?all=true"))


@pytest.mark.api
def test_an_empty_list_says_there_is_nothing_yet(service: Service) -> None:
    sign_in(service)

    assert "Оценок пока нет." in text_of(get(service, "/ui/?all=true"))


@pytest.mark.api
def test_the_page_shows_the_prediction_as_the_api_does(service: Service) -> None:
    sign_in(service)
    assess(service)

    shown = page(get(service, ASSESSMENT)).xpath("//h1/span[@class='load']/text()")

    answered = child(
        document(service.client.get(f"/api/v1/assessments/{ASSESSMENT_ID}")), "LoadPct"
    )
    assert shown == [f"{answered.text}%"]


@pytest.mark.api
@pytest.mark.parametrize(
    ("typed", "stored"),
    [("72,5", 72.5), (" 0 ", 0.0), ("-0", 0.0), ("100", 100.0), ("33.333", 33.333)],
)
def test_a_typed_load_is_stored_as_typed(service: Service, typed: str, stored: float) -> None:
    token = sign_in(service)
    assess(service)

    response = submit(service, f"{ASSESSMENT}/label", csrf=token, action="confirm", load_pct=typed)

    assert (response.status_code, response.headers["location"]) == (303, "/ui/")
    label = service.store.labels[PHOTO_SHA256]
    assert label.load_pct is not None
    # repr tells 0.0 from -0.0, which compare equal but print differently.
    assert repr(label.load_pct.value) == repr(stored)
    assert (label.source, label.author) == (LabelSource.UI, LOGIN)


@pytest.mark.api
def test_the_label_form_sends_what_the_route_reads(service: Service) -> None:
    sign_in(service)
    assess(service)

    form = page(get(service, ASSESSMENT)).xpath("//form[@method='post']")[-1]

    assert form.get("action") == f"{ASSESSMENT}/label"
    assert form.xpath(".//input[@type='text']/@name") == ["load_pct"]
    assert form.xpath(".//button/@value") == ["confirm", "unusable"]


@pytest.mark.api
def test_an_unusable_photo_is_stored_without_a_load(service: Service) -> None:
    token = sign_in(service)
    assess(service)

    submit(service, f"{ASSESSMENT}/label", csrf=token, action="unusable", load_pct="")

    assert service.store.labels[PHOTO_SHA256].load_pct is None
    assert "Фото отмечено непригодным" in text_of(get(service, ASSESSMENT))


@pytest.mark.api
@pytest.mark.parametrize("typed", ["", "abc", "100.1", "-1", "nan", "inf"])
def test_a_load_outside_the_scale_returns_the_form_with_the_reason(
    service: Service, typed: str
) -> None:
    token = sign_in(service)
    assess(service)

    response = submit(service, f"{ASSESSMENT}/label", csrf=token, action="confirm", load_pct=typed)

    assert response.status_code == 400
    assert "Загрузка — число от 0 до 100, например 72.5." in text_of(response)
    assert page(response).xpath("//input[@name='load_pct']/@value") == [typed]
    assert service.store.labels == {}


@pytest.mark.api
def test_a_label_without_the_session_token_is_not_stored(service: Service) -> None:
    sign_in(service)
    assess(service)

    response = submit(service, f"{ASSESSMENT}/label", action="confirm", load_pct="60")

    assert response.status_code == 403
    assert service.store.labels == {}


@pytest.mark.api
@pytest.mark.parametrize("assessment_id", [str(UUID(int=7)), "not-a-uuid"])
def test_an_unknown_assessment_is_a_missing_page(service: Service, assessment_id: str) -> None:
    sign_in(service)

    response = get(service, f"/ui/assessments/{assessment_id}")

    assert response.status_code == 404
    assert ERROR_CATALOG[ErrorCode.ASSESSMENT_NOT_FOUND].message in text_of(response)


@pytest.mark.api
def test_the_photo_comes_with_its_type_and_may_be_cached(service: Service) -> None:
    sign_in(service)
    assess(service)

    response = get(service, f"/ui/photos/{PHOTO_SHA256}")

    assert response.status_code == 200
    assert response.content == PHOTO
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "private, max-age=86400, immutable"


@pytest.mark.api
@pytest.mark.parametrize("sha256", ["0" * 64, "A" * 64, "0" * 63])
def test_a_photo_that_is_not_stored_is_a_missing_page(service: Service, sha256: str) -> None:
    sign_in(service)

    assert get(service, f"/ui/photos/{sha256}").status_code == 404


@pytest.mark.api
def test_photos_are_shown_only_to_a_signed_in_operator(service: Service) -> None:
    assess(service)

    assert get(service, f"/ui/photos/{PHOTO_SHA256}").status_code == 303


@pytest.mark.api
def test_the_header_names_the_model_that_answers() -> None:
    loaded = make_service(FakePredictor(version="fake-7"))
    loading = make_service()
    sign_in(loaded)
    sign_in(loading)

    assert page(get(loaded, "/ui/")).xpath("//span[contains(@class, 'badge')]/text()") == ["fake-7"]
    assert "модель загружается" in text_of(get(loading, "/ui/"))
