import base64
from datetime import timedelta
from uuid import UUID

import pytest

from loadvision.errors import ERROR_CATALOG, ErrorCode
from tests.api.service import PHOTO_SHA256, Service, post, request_body
from tests.api.ui_pages import get, page, sign_in, submit, text_of
from tests.image_files import png


@pytest.mark.api
def test_the_history_shows_every_photo_newest_first_with_its_label(service: Service) -> None:
    token = sign_in(service)
    assert post(service.client, request_body()).status_code == 201
    service.clock.advance(timedelta(minutes=10))
    newer = request_body(image=base64.b64encode(png()).decode())
    assert post(service.client, newer).status_code == 201
    first = f"/ui/assessments/{UUID(int=1)}"
    submit(service, f"{first}/label", csrf=token, action="confirm", load_pct="40")

    table = page(get(service, "/ui/shipments/RP-1"))

    # Columns: photo, time, prediction, confirmed load, model.
    rows = [
        [" ".join(cell.text_content().split()) for cell in row.xpath("td")]
        for row in table.xpath("//tbody/tr")
    ]
    assert [(row[1], row[3], row[4]) for row in rows] == [
        ("2026-09-16 12:10:00 UTC текущая", "нет", "fake-1"),
        ("2026-09-16 12:00:00 UTC", "40.0%", "fake-1"),
    ]
    assert table.xpath("//tbody/tr[2]//img/@src") == [f"/ui/photos/{PHOTO_SHA256}"]
    assert table.xpath("//tbody/tr[2]/td[2]/a/@href") == [first]


@pytest.mark.api
def test_a_typed_shipment_number_opens_its_page(service: Service) -> None:
    sign_in(service)

    typed = get(service, "/ui/shipments?number=%20RP-1%20")
    slashed = get(service, "/ui/shipments?number=RP%2F1%3Fx")

    assert typed.headers["location"] == "/ui/shipments/RP-1"
    # A slash or a question mark stays inside the number instead of changing the address.
    assert slashed.headers["location"] == "/ui/shipments/RP%2F1%3Fx"


@pytest.mark.api
def test_an_empty_search_shows_the_form(service: Service) -> None:
    sign_in(service)

    response = get(service, "/ui/shipments?number=%20")

    assert response.status_code == 200
    assert page(response).xpath("//input[@name='number']/@value") == [""]


@pytest.mark.api
@pytest.mark.parametrize(
    ("number", "code"),
    [("RP-404", ErrorCode.SHIPMENT_NOT_FOUND), ("RP%201", ErrorCode.INVALID_SHIPMENT_ID)],
)
def test_a_shipment_that_cannot_be_shown_says_why(
    service: Service, number: str, code: ErrorCode
) -> None:
    sign_in(service)

    response = get(service, f"/ui/shipments/{number}")

    spec = ERROR_CATALOG[code]
    assert response.status_code == spec.http_status
    assert spec.message in text_of(response)
