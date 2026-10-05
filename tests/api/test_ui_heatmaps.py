from uuid import UUID

import pytest

from loadvision.application.ports import ExplanationKind
from loadvision.domain.model_version import ModelVersionId
from tests.api.service import PHOTO_SHA256, Service, make_service, post, request_body
from tests.api.ui_pages import form_token, get, page, sign_in, submit, text_of
from tests.fakes.heatmaps import PICTURE
from tests.fakes.predictor import FakePredictor

ASSESSMENT = f"/ui/assessments/{UUID(int=1)}"
MAP = f"/ui/photos/{PHOTO_SHA256}/heatmap"


@pytest.fixture
def drawing() -> Service:
    """A service that can draw maps, with one photo already assessed."""
    service = make_service(FakePredictor(), heatmaps=True)
    sign_in(service)
    assert post(service.client, request_body()).status_code == 201
    return service


@pytest.mark.api
def test_the_card_offers_the_photo_and_both_maps(drawing: Service) -> None:
    card = page(get(drawing, ASSESSMENT))

    switch = card.xpath("//nav[@class='views']/a")
    assert [link.text for link in switch] == ["Фото", "Внимание", "Вклад в ответ"]
    assert [link.get("href") for link in switch] == [
        "?view=",
        "?view=attention",
        "?view=contribution",
    ]
    # Nothing is drawn while the operator looks at the photo itself.
    assert card.xpath("//figure[@class='photo']//img/@src") == [f"/ui/photos/{PHOTO_SHA256}"]
    assert drawing.heatmaps is not None
    assert drawing.heatmaps.asked == []


@pytest.mark.api
@pytest.mark.parametrize("kind", list(ExplanationKind))
def test_a_chosen_map_is_shown_in_place_of_the_photo(
    drawing: Service, kind: ExplanationKind
) -> None:
    card = page(get(drawing, f"{ASSESSMENT}?view={kind.value}"))

    assert card.xpath("//figure[@class='photo']//img/@src") == [f"{MAP}?kind={kind.value}"]
    assert card.xpath("//nav[@class='views']/a[@aria-current='true']/text()") == [
        "Внимание" if kind is ExplanationKind.ATTENTION else "Вклад в ответ"
    ]


@pytest.mark.api
@pytest.mark.parametrize("path", [ASSESSMENT, f"{ASSESSMENT}?view=attention"])
def test_a_frame_the_model_refused_gets_no_maps(path: str) -> None:
    # There is no truck body in the frame, so there is nothing to show a map of.
    refused = make_service(FakePredictor(ood_score=0.9), heatmaps=True)
    sign_in(refused)
    post(refused.client, request_body())

    card = page(get(refused, path))

    assert card.xpath("//nav[@class='views']") == []
    assert card.xpath("//figure[@class='photo']//img/@src") == [f"/ui/photos/{PHOTO_SHA256}"]
    assert refused.heatmaps is not None
    assert refused.heatmaps.asked == []


@pytest.mark.api
@pytest.mark.parametrize(
    ("path", "waiting"),
    [(ASSESSMENT, "Фото загружается"), (f"{ASSESSMENT}?view=attention", "Карта считается")],
)
def test_the_place_of_the_picture_says_what_is_coming(
    drawing: Service, path: str, waiting: str
) -> None:
    card = page(get(drawing, path))

    # The label waits behind the picture, so an empty box never stands there without a word.
    assert card.xpath("//div[@class='loading']/span/text()")[0].startswith(waiting)


@pytest.mark.api
def test_a_view_the_service_does_not_know_falls_back_to_the_photo(drawing: Service) -> None:
    card = page(get(drawing, f"{ASSESSMENT}?view=occlusion"))

    assert card.xpath("//figure[@class='photo']//img/@src") == [f"/ui/photos/{PHOTO_SHA256}"]


@pytest.mark.api
@pytest.mark.parametrize("kind", list(ExplanationKind))
def test_the_map_is_drawn_by_the_model_that_answers_now(
    drawing: Service, kind: ExplanationKind
) -> None:
    response = drawing.client.get(f"{MAP}?kind={kind.value}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == PICTURE
    assert drawing.heatmaps is not None
    assert drawing.heatmaps.asked == [(PHOTO_SHA256, kind, ModelVersionId("fake-1"))]


@pytest.mark.api
@pytest.mark.parametrize(
    "path",
    [f"{MAP}?kind=occlusion", f"{MAP}", f"/ui/photos/{'0' * 63}/heatmap?kind=attention"],
    ids=["unknown kind", "no kind", "malformed hash"],
)
def test_an_address_the_service_cannot_read_is_a_missing_page(drawing: Service, path: str) -> None:
    assert drawing.client.get(path).status_code == 404


@pytest.mark.api
def test_a_service_without_an_explainer_offers_no_maps(service: Service) -> None:
    sign_in(service)
    post(service.client, request_body())

    card = page(get(service, ASSESSMENT))

    assert card.xpath("//nav[@class='views']") == []
    assert service.client.get(f"{MAP}?kind=attention").status_code == 404


@pytest.mark.api
def test_a_map_waits_for_the_model() -> None:
    service = make_service(heatmaps=True)
    sign_in(service)

    response = service.client.get(f"{MAP}?kind=attention")

    assert response.status_code == 503
    assert "Модель ещё не загружена" in text_of(response)


@pytest.mark.api
def test_maps_are_shown_only_to_a_signed_in_operator(drawing: Service) -> None:
    drawing.client.cookies.clear()

    response = drawing.client.get(f"{MAP}?kind=attention", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/ui/login")
    assert drawing.heatmaps is not None
    assert drawing.heatmaps.asked == []


@pytest.mark.api
def test_a_refused_load_keeps_the_card_whole(drawing: Service) -> None:
    """The error path renders the same template, so it owes it the same context.

    A context it forgets is not an error for Jinja, it is a part of the page that silently
    disappears: the switch of the maps vanished from the card behind a refused number.
    """
    card = get(drawing, ASSESSMENT)

    refused = submit(
        drawing,
        f"{ASSESSMENT}/label",
        csrf=form_token(card),
        action="confirm",
        load_pct="abc",
    )

    assert refused.status_code == 400
    whole = "//nav[@class='views']/a | //form[@class='weigh']"
    assert len(page(refused).xpath(whole)) == len(page(card).xpath(whole))
