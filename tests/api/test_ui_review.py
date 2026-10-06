from uuid import UUID

import pytest

from loadvision.api.ui.templates import REVIEW_NOTES
from loadvision.domain.assessment import ReviewReason, Signals
from loadvision.domain.cargo_weight import (
    BODIES,
    DEFAULT_STACK_SHARE,
    DENSITIES,
    CargoKind,
    estimate_weight,
)
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
# The load `make_assessment` stores, which every weight below is computed from.
ASSESSED_LOAD = 50.0


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
def test_the_sizes_stay_typeable_with_a_body_from_the_list(service: Service) -> None:
    """A field the browser blocks is never posted, so locked sizes can never be typed at all."""
    assess_with_signals(service, ood_score=0.1)
    sign_in(service)

    sizes = page(get(service, f"{ASSESSMENT}?weigh=1")).xpath(
        "//input[@name='length' or @name='width' or @name='height']"
    )

    assert len(sizes) == 3
    assert [field for field in sizes if field.get("disabled") is not None] == []


@pytest.mark.api
def test_the_weight_uses_the_stack_measured_on_the_photo() -> None:
    """Otherwise every body is weighed as if the cargo always reached the same height."""
    measured = 0.9
    service = make_service(stack=measured)
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t5", cargo="pallets")

    expected = estimate_weight(
        ASSESSED_LOAD,
        body=BODIES["t5"],
        density=DENSITIES[CargoKind.PALLETS],
        stack_share=measured,
    )
    assert f"{expected.cubic_metres} м³" in text_of(get(service, ASSESSMENT))


@pytest.mark.api
def test_a_weight_falls_back_to_the_default_share_when_nothing_was_measured(
    service: Service,
) -> None:
    """A service without a depth model still weighs; it just cannot sharpen the assumption."""
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t5", cargo="pallets")

    expected = estimate_weight(
        ASSESSED_LOAD,
        body=BODIES["t5"],
        density=DENSITIES[CargoKind.PALLETS],
        stack_share=DEFAULT_STACK_SHARE,
    )
    assert f"{expected.cubic_metres} м³" in text_of(get(service, ASSESSMENT))


@pytest.mark.api
def test_a_stack_of_zero_does_not_weigh_the_cargo_as_nothing() -> None:
    """The depth map found nothing standing out, which a photo of a loaded body never means."""
    service = make_service(stack=0.0)
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t5", cargo="pallets")

    expected = estimate_weight(
        ASSESSED_LOAD,
        body=BODIES["t5"],
        density=DENSITIES[CargoKind.PALLETS],
        stack_share=DEFAULT_STACK_SHARE,
    )
    assert f"{expected.cubic_metres} м³" in text_of(get(service, ASSESSMENT))


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
def test_the_weight_block_waits_until_the_operator_asks_for_it(service: Service) -> None:
    """A card shows what the model answered; weighing is a separate thing the operator starts."""
    assess_with_signals(service, ood_score=0.1)
    sign_in(service)

    card = page(get(service, ASSESSMENT))

    assert card.xpath("//form[@class='weigh']") == []
    opens = card.xpath("//form[@class='weigh-start']")
    assert [element.get("method") for element in opens] == ["get"]
    assert opens[0].xpath(".//input[@name='weigh']/@value") == ["1"]


@pytest.mark.api
def test_a_stored_weighing_keeps_the_block_open(service: Service) -> None:
    """Numbers an operator has already asked for must not hide behind the button again."""
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)
    submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t10", cargo="pallets")

    card = page(get(service, ASSESSMENT))

    assert card.xpath("//form[@class='weigh']")
    assert card.xpath("//form[@class='weigh-start']") == []


@pytest.mark.api
def test_a_weighing_that_stored_nothing_leaves_the_block_open(service: Service) -> None:
    """Nothing was stored, so only the address can keep the block the operator opened."""
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    sent = submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t10", cargo="")

    assert sent.headers["location"] == f"{ASSESSMENT}?weigh=1"


@pytest.mark.api
def test_the_cargo_list_names_the_density_it_weighs_with(service: Service) -> None:
    assess_with_signals(service, ood_score=0.1)
    sign_in(service)

    options = page(get(service, f"{ASSESSMENT}?weigh=1")).xpath(
        "//select[@name='cargo']/option/text()"
    )

    pallets = DENSITIES[CargoKind.PALLETS]
    assert f"Паллеты, {round(pallets.low)}\u2013{round(pallets.high)} кг/м³" in options


@pytest.mark.api
def test_the_photo_carries_its_size_so_the_card_does_not_jump(service: Service) -> None:
    """Without the size the box is drawn short and grows when the picture arrives."""
    assess_with_signals(service, ood_score=0.1)
    sign_in(service)

    shown = page(get(service, ASSESSMENT)).xpath("//div[@class='loading']/img")

    assert len(shown) == 1
    assert int(shown[0].get("width", "0")) > 0
    assert int(shown[0].get("height", "0")) > 0


@pytest.mark.api
def test_the_weight_form_posts_to_the_route_that_stores_it(service: Service) -> None:
    """The tests reach the route directly, so only this one notices a form that never sends."""
    assess_with_signals(service, ood_score=0.1)
    sign_in(service)

    card = page(get(service, f"{ASSESSMENT}?weigh=1"))

    form = card.xpath("//form[@class='weigh']")
    assert [element.get("method") for element in form] == ["post"]
    sends_to = [element.get("action") for element in form]
    assert sends_to == [f"/ui/assessments/{ASSESSMENT_ID}/weight"]
    assert card.xpath("//form[@class='weigh']//input[@name='csrf']")
    # A disabled field is never submitted, so disabling the sides made "иной габарит"
    # unreachable: the form sent nothing, and the route fell back to the body of the list.
    named = "@name='length' or @name='width' or @name='height'"
    sides = card.xpath(f"//form[@class='weigh']//input[{named}]")
    assert len(sides) == 3
    assert [element.get("disabled") for element in sides] == [None, None, None]


@pytest.mark.api
def test_reading_the_kind_off_the_photo_is_offered_first_until_something_is_weighed() -> None:
    service = make_service(cargo=CargoKind.PALLETS)
    assess_with_signals(service, ood_score=0.1)
    sign_in(service)

    options = page(get(service, f"{ASSESSMENT}?weigh=1")).xpath("//select[@name='cargo']/option")

    assert options[0].get("value") == "auto"
    assert [option.get("value") for option in options if option.get("selected") is not None] == [
        "auto"
    ]


@pytest.mark.api
def test_a_service_without_the_model_offers_no_reading(service: Service) -> None:
    assess_with_signals(service, ood_score=0.1)
    sign_in(service)

    values = page(get(service, f"{ASSESSMENT}?weigh=1")).xpath(
        "//select[@name='cargo']/option/@value"
    )

    assert "auto" not in values


@pytest.mark.api
def test_the_kind_read_off_the_photo_is_weighed_and_named() -> None:
    service = make_service(cargo=CargoKind.PALLETS)
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    sent = submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t5", cargo="auto")
    card = text_of(get(service, sent.headers["location"]))

    pallets = DENSITIES[CargoKind.PALLETS]
    assert f"{round(pallets.low)}\u2013{round(pallets.high)} кг/м³" in card
    assert "Вид груза определён по фото: Паллеты" in card


@pytest.mark.api
def test_a_body_the_model_calls_empty_is_weighed_as_mixed() -> None:
    """How empty the body is, the load says; the weight of "nothing" was asked by nobody."""
    service = make_service(cargo=CargoKind.EMPTY)
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    sent = submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t5", cargo="auto")

    card = page(get(service, sent.headers["location"]))
    chosen = card.xpath("//select[@name='cargo']/option[@selected]/@value")
    assert chosen == [CargoKind.MIXED.value]
    assert "Вид груза определён по фото: Смешанный" in " ".join(card.text_content().split())


@pytest.mark.api
def test_asking_for_a_reading_the_service_cannot_give_weighs_as_mixed(service: Service) -> None:
    """A page is not an API: a form sent by hand still gets a weight, the least wrong one."""
    assess_with_signals(service, ood_score=0.1)
    token = sign_in(service)

    sent = submit(service, f"{ASSESSMENT}/weight", csrf=token, body="t5", cargo="auto")

    mixed = DENSITIES[CargoKind.MIXED]
    assert f"{round(mixed.low)}\u2013{round(mixed.high)} кг/м³" in text_of(
        get(service, sent.headers["location"])
    )
