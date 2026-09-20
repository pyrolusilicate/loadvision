from uuid import UUID

import pytest

from loadvision.application.ports import ExplanationKind
from loadvision.domain.model_version import ModelVersionId
from tests.api.service import PHOTO_SHA256, Service, make_service, post, request_body
from tests.api.ui_pages import get, page, sign_in, text_of
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
def test_the_contribution_map_says_when_it_has_little_to_say(drawing: Service) -> None:
    # The model is at its ceiling on a full body, so the map paints the cargo as pulling down.
    full = make_service(FakePredictor(loads={PHOTO_SHA256: 100.0}), heatmaps=True)
    sign_in(full)
    post(full.client, request_body())

    caveat = "На полном кузове она говорит мало"

    assert caveat in text_of(get(full, f"{ASSESSMENT}?view=contribution"))
    assert caveat not in text_of(get(full, f"{ASSESSMENT}?view=attention"))
    assert caveat not in text_of(get(drawing, f"{ASSESSMENT}?view=contribution"))


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
