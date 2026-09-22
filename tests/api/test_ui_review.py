from uuid import UUID

import pytest

from loadvision.api.ui.templates import REVIEW_NOTES
from loadvision.domain.assessment import ReviewReason, Signals
from loadvision.domain.cargo_weight import DENSITIES, CargoKind
from loadvision.domain.confirmed_label import LabelSource
from loadvision.errors import ERROR_CATALOG, ErrorCode
from tests.api.service import (
    PHOTO,
    PHOTO_SHA256,
    UNREADABLE_OOD,
    Service,
    child,
    document,
    make_service,
    post,
    request_body,
)
from tests.api.ui_pages import LOGIN, get, page, sign_in, submit, text_of
from tests.builders import make_assessment
from tests.fakes.predictor import FakePredictor
from tests.port_contracts.unit_of_work import store_assessment

ASSESSMENT_ID = str(UUID(int=1))
ASSESSMENT = f"/ui/assessments/{ASSESSMENT_ID}"


def assess(service: Service) -> None:
    assert post(service.client, request_body()).status_code == 201


def assess_with_reason(service: Service, reason: ReviewReason) -> None:
    with service.store.unit_of_work() as uow:
        store_assessment(uow, make_assessment(review_reasons=frozenset({reason})))
        uow.commit()


def assess_with_signals(service: Service, ood_score: float) -> None:
    """Stores an assessment the model marked, with the measurements a card shows."""
    with service.store.unit_of_work() as uow:
        store_assessment(
            uow,
            make_assessment(
                signals=Signals(ood_score=ood_score, brightness=219.0),
                review_reasons=frozenset({ReviewReason.UNUSUAL_SCENE}),
            ),
        )
        uow.commit()


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
    assert "подтверждено 60.0%" in text_of(get(service, "/ui/?status=confirmed"))


@pytest.mark.api
def test_an_empty_list_says_there_is_nothing_yet(service: Service) -> None:
    sign_in(service)

    assert "Оценок пока нет." in text_of(get(service, "/ui/?status=all"))


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
def test_the_measurements_stand_next_to_the_reason_to_check_the_photo(service: Service) -> None:
    sign_in(service)
    assess_with_signals(service, ood_score=0.181)

    card = page(get(service, ASSESSMENT))
    note = card.xpath("//div[@class='note warn']")

    assert len(note) == 1
    assert note[0].xpath(".//dt/text()") == ["Непохожесть на обучающие", "Яркость кадра"]
    assert note[0].xpath(".//dd/text()") == ["0.181", "219 из 255"]
    # And nowhere else: the list below the note keeps the model and the times.
    assert card.xpath("//dt/text()").count("Непохожесть на обучающие") == 1


@pytest.mark.api
def test_a_frame_the_model_refused_is_shown_without_a_prediction(service: Service) -> None:
    token = sign_in(service)
    service.model.activate(FakePredictor(ood_score=UNREADABLE_OOD))
    assert post(service.client, request_body()).status_code == 201

    card = page(get(service, ASSESSMENT))
    text = text_of(get(service, ASSESSMENT))

    assert "Загружено нечитаемое фото." in text
    assert card.xpath("//div[@class='note bad']/@role") == ["alert"]
    # No prediction, no model, no way to confirm a number nobody should trust.
    assert card.xpath("//span[@class='load']") == []
    # Everything the card still says stands inside the red block, as on a marked photo.
    assert card.xpath("//div[@class='note bad']//dt/text()") == [
        "Время оценки",
        "Непохожесть на обучающие",
        "Яркость кадра",
    ]
    assert card.xpath("//section[@class='card']/dl") == []
    assert card.xpath("//section[@class='card']//button/text()") == ["Фото читаемо"]
    assert "Подтверждённая загрузка" not in text
    # The shipment does not count the photo: the XML API answers that it is unusable.
    current = child(document(service.client.get("/api/v1/shipments/RP-1")), "Current")
    assert current.get("photoUnusable") == "true"
    assert current.get("confirmedLoadPct") is None
    assert token  # the card needs a session, and the button carries its token


@pytest.mark.api
def test_the_queue_shows_no_number_for_a_frame_the_model_refused(service: Service) -> None:
    sign_in(service)
    service.model.activate(FakePredictor(ood_score=UNREADABLE_OOD))
    post(service.client, request_body())

    tile = page(get(service, "/ui/?status=unusable")).xpath("//a[@class='tile']/span")

    assert "Нечитаемое фото" in str(tile[0].xpath("string(strong)"))
    assert tile[0].xpath(".//strong[@class='load']") == []


@pytest.mark.api
def test_the_button_cancels_the_verdict_of_the_model_and_the_photo_is_judged(
    service: Service,
) -> None:
    token = sign_in(service)
    service.model.activate(FakePredictor(ood_score=UNREADABLE_OOD))
    post(service.client, request_body())

    submit(service, f"{ASSESSMENT}/readable", csrf=token)

    card = page(get(service, ASSESSMENT))
    assert card.xpath("//span[@class='load']") != []
    assert card.xpath("//div[@class='note bad']") == []
    assert service.store.labels == {}
    assert "Всё проверено" not in text_of(get(service, "/ui/"))


@pytest.mark.api
def test_the_button_never_takes_back_a_verdict_of_a_person(service: Service) -> None:
    token = sign_in(service)
    assess(service)
    submit(service, f"{ASSESSMENT}/label", csrf=token, action="unusable")

    submit(service, f"{ASSESSMENT}/readable", csrf=token)

    card = text_of(get(service, ASSESSMENT))
    assert "Фото отмечено непригодным" in card
    assert "Загружено нечитаемое фото." not in card
    assert list(service.store.labels) == [PHOTO_SHA256]


@pytest.mark.api
@pytest.mark.parametrize("reason", list(ReviewReason))
def test_the_card_says_in_words_why_the_prediction_needs_a_check(
    service: Service, reason: ReviewReason
) -> None:
    sign_in(service)
    assess_with_reason(service, reason)

    card = text_of(get(service, ASSESSMENT))

    assert f"Прогноз может быть неверным. {REVIEW_NOTES[reason]}" in card
    assert reason.value not in card
    tile = page(get(service, "/ui/")).xpath("//a[@class='tile']//span[@class='badge warn']")
    assert [badge.text for badge in tile] == ["проверить"]


@pytest.mark.api
def test_a_confirmed_photo_no_longer_asks_for_a_check(service: Service) -> None:
    token = sign_in(service)
    assess_with_reason(service, ReviewReason.UNUSUAL_SCENE)

    submit(service, f"{ASSESSMENT}/label", csrf=token, action="confirm", load_pct="60")

    assert "Прогноз может быть неверным." not in text_of(get(service, ASSESSMENT))
    assert page(get(service, "/ui/?status=confirmed")).xpath("//span[@class='badge warn']") == []


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


@pytest.mark.api
def test_a_confirmed_photo_still_says_the_model_doubted_it(service: Service) -> None:
    """The operator's number replaces the prediction; that the model doubted it stays on record."""
    assess_with_signals(service, ood_score=0.9)
    token = sign_in(service)

    submit(service, f"{ASSESSMENT}/label", csrf=token, action="confirm", load_pct="60")

    card = text_of(get(service, ASSESSMENT))
    assert "Модель сомневалась в этом фото" in card
    assert REVIEW_NOTES[ReviewReason.UNUSUAL_SCENE] in card


@pytest.mark.api
def test_the_weight_block_answers_the_body_and_the_cargo_it_was_given(service: Service) -> None:
    """A photo gives the load; the body and the density are the operator's, so both are shown."""
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t20", cargo="pallets")

    card = text_of(get(service, ASSESSMENT))
    assert "Объём груза" in card
    assert "13.60×2.45×2.70 м" in card


@pytest.mark.api
def test_a_body_typed_by_hand_is_used_instead_of_the_list(service: Service) -> None:
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    submit(
        service,
        f"{ASSESSMENT}/weight",
        csrf=token,
        body="custom",
        cargo="sacks",
        length="5",
        width="2",
        height="2,5",
    )

    assert "5.00×2.00×2.50 м" in text_of(get(service, ASSESSMENT))


@pytest.mark.api
def test_a_body_typed_wrong_falls_back_instead_of_failing_the_page(service: Service) -> None:
    """A page is not an API: a query typed by hand draws the block, it does not break it."""
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    submit(
        service,
        f"{ASSESSMENT}/weight",
        csrf=token,
        body="custom",
        cargo="sacks",
        length="999",
        width="x",
        height="",
    )

    # The sizes were unusable, so the smallest body of the list stands in for them.
    assert "6.20×2.45×2.40 м" in text_of(get(service, ASSESSMENT))


@pytest.mark.api
def test_no_weight_is_shown_until_a_cargo_is_chosen(service: Service) -> None:
    assess_with_signals(service, ood_score=0.1)
    sign_in(service)

    # The heading of the block is always there; the answer under it is not.
    assert "Объём груза" not in text_of(get(service, ASSESSMENT))


@pytest.mark.api
def test_the_block_shows_the_density_it_weighed_with(service: Service) -> None:
    """The number has to be on the page: a weight nobody can check is worth nothing."""
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t5", cargo="roll_cages")

    card = text_of(get(service, ASSESSMENT))
    low, high = DENSITIES[CargoKind.ROLL_CAGES].low, DENSITIES[CargoKind.ROLL_CAGES].high
    assert f"{low:.0f}–{high:.0f} кг/м³" in card


@pytest.mark.api
def test_an_empty_cargo_typed_into_the_address_is_ignored(service: Service) -> None:
    """How empty the body is, the load percent says; a kind of "nothing" has no weight to give."""
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t5", cargo="empty")

    assert "Объём груза" not in text_of(get(service, ASSESSMENT))


@pytest.mark.api
def test_a_weighing_survives_a_reload(service: Service) -> None:
    """The number is stored, not recomputed: an operator acts on what he was shown."""
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t10", cargo="roll_cages")

    first = text_of(get(service, ASSESSMENT))
    second = text_of(get(service, ASSESSMENT))
    assert "8.00×2.45×2.60 м" in first
    assert first == second


@pytest.mark.api
def test_weighing_again_replaces_the_stored_numbers(service: Service) -> None:
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)
    submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t5", cargo="roll_cages")

    submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t20", cargo="pallets")

    card = text_of(get(service, ASSESSMENT))
    assert "13.60×2.45×2.70 м" in card
    assert "6.20×2.45×2.40 м" not in card


@pytest.mark.api
def test_the_weight_form_posts_to_the_route_that_stores_it(service: Service) -> None:
    """The tests reach the route directly, so only this one notices a form that never sends."""
    assess_with_signals(service, ood_score=0.1)
    sign_in(service)

    card = page(get(service, ASSESSMENT))

    form = card.xpath("//form[@class='weigh']")
    assert [element.get("method") for element in form] == ["post", "get"]
    sends_to = [element.get("action") for element in form]
    assert sends_to == [f"/ui/assessments/{ASSESSMENT_ID}/weight", None]
    assert card.xpath("//form[@class='weigh']//input[@name='csrf']")
