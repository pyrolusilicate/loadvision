from datetime import timedelta
from uuid import UUID

import pytest

from loadvision.domain.assessment import LoadPercent
from loadvision.domain.model_version import ModelVersionStatus
from loadvision.domain.shipment import ShipmentId
from tests.api.service import Service
from tests.api.ui_pages import get, page, sign_in, text_of
from tests.builders import (
    T0,
    make_assessment,
    make_image,
    make_job,
    make_label,
    make_model_version,
)
from tests.port_contracts.unit_of_work import store_assessment

TILES = 24
ROWS = 20


def assess(service: Service, count: int, shipment: str = "RP-1") -> None:
    """Stores `count` assessments a minute apart; the n-th has load n % 101."""
    with service.store.unit_of_work() as uow:
        for number in range(1, count + 1):
            raw = f"photo-{number}".encode()
            assessment = make_assessment(
                id=UUID(int=number),
                shipment_id=ShipmentId(shipment),
                image_sha256=make_image(raw).sha256,
                load_pct=LoadPercent(float(number % 101)),
                created_at=T0 + timedelta(minutes=number),
            )
            store_assessment(uow, assessment, raw)
        uow.commit()


def tiles(service: Service, path: str) -> list[str]:
    """Ids of the assessments a queue page shows, in order."""
    hrefs = page(get(service, path)).xpath("//a[@class='tile']/@href")
    return [str(href).rsplit("/", 1)[1] for href in hrefs]


def pager_links(service: Service, path: str) -> list[str]:
    return [str(href) for href in page(get(service, path)).xpath("//div[@class='pager']/a/@href")]


@pytest.mark.api
def test_the_queue_comes_in_pages_with_links_both_ways(service: Service) -> None:
    sign_in(service)
    assess(service, TILES + 1)

    first = tiles(service, "/ui/")
    second = tiles(service, "/ui/?page=2")

    assert len(first) == TILES
    assert second == [str(UUID(int=1))]
    assert pager_links(service, "/ui/") == ["/ui/?page=2"]
    assert pager_links(service, "/ui/?page=2") == ["/ui/?page=1"]
    assert f"Страница 1 из 2 · всего {TILES + 1}" in text_of(get(service, "/ui/"))


@pytest.mark.api
def test_one_page_needs_no_pager(service: Service) -> None:
    sign_in(service)
    assess(service, 3)

    assert page(get(service, "/ui/")).xpath("//div[@class='pager']") == []


@pytest.mark.api
def test_a_page_past_the_end_shows_the_last_one(service: Service) -> None:
    sign_in(service)
    assess(service, TILES + 1)

    assert tiles(service, "/ui/?page=99") == [str(UUID(int=1))]


@pytest.mark.api
@pytest.mark.parametrize(
    ("order", "first"),
    [("new", 30), ("old", 1), ("full", 30), ("empty", 1)],
)
def test_the_queue_sorts_by_time_or_by_load(service: Service, order: str, first: int) -> None:
    sign_in(service)
    assess(service, 30)

    assert tiles(service, f"/ui/?order={order}")[0] == str(UUID(int=first))


@pytest.mark.api
def test_the_order_and_the_filter_survive_every_link(service: Service) -> None:
    sign_in(service)
    assess(service, TILES + 1)
    path = "/ui/?status=all&order=full"
    queue = page(get(service, path))
    menus = queue.xpath("//details[@class='menu']")
    status_menu, order_menu = menus

    assert pager_links(service, path) == ["/ui/?status=all&order=full&page=2"]
    assert status_menu.xpath(".//a/@href") == [
        "/ui/?order=full",
        "/ui/?status=confirmed&order=full",
        "/ui/?status=unusable&order=full",
        path,
    ]
    assert order_menu.xpath(".//a/@href") == [
        "/ui/?status=all",
        "/ui/?status=all&order=old",
        path,
        "/ui/?status=all&order=empty",
    ]
    summaries = [" ".join(str(menu.xpath("string(summary)")).split()) for menu in menus]
    assert summaries == ["Статус Все", "Порядок Сначала полные"]
    assert queue.xpath("//a[@aria-current='true']/text()") == ["Все", "Сначала полные"]


@pytest.mark.api
@pytest.mark.parametrize(
    ("status", "heading", "shown"),
    [
        ("", "На рассмотрении", [3]),
        ("?status=confirmed", "Подтверждённые", [1]),
        ("?status=unusable", "Непригодные", [2]),
        ("?status=all", "Все", [3, 2, 1]),
    ],
)
def test_the_queue_filters_by_what_a_person_said(
    service: Service, status: str, heading: str, shown: list[int]
) -> None:
    sign_in(service)
    assess(service, 3)
    with service.store.unit_of_work() as uow:
        uow.labels.save(make_label(b"photo-1", 60.0))
        uow.labels.save(make_label(b"photo-2", None))
        uow.commit()

    queue = page(get(service, f"/ui/{status}"))

    assert queue.xpath("//h1/text()") == [heading]
    assert tiles(service, f"/ui/{status}") == [str(UUID(int=number)) for number in shown]


@pytest.mark.api
@pytest.mark.parametrize("path", ["/ui/?order=biggest", "/ui/?status=maybe", "/ui/?page=two"])
def test_an_address_the_queue_cannot_read_is_a_missing_page(service: Service, path: str) -> None:
    sign_in(service)

    assert get(service, path).status_code == 404


@pytest.mark.api
def test_a_long_shipment_history_comes_in_pages(service: Service) -> None:
    sign_in(service)
    assess(service, ROWS + 1)

    first = page(get(service, "/ui/shipments/RP-1"))
    second = page(get(service, "/ui/shipments/RP-1?page=2"))

    assert len(first.xpath("//tbody/tr")) == ROWS
    assert len(second.xpath("//tbody/tr")) == 1
    # Only the newest assessment of the whole history is the current one.
    assert first.xpath("//tbody/tr[1]//span[@class='badge']/text()") == ["текущая"]
    assert second.xpath("//span[@class='badge']/text()") == []
    assert pager_links(service, "/ui/shipments/RP-1") == ["/ui/shipments/RP-1?page=2"]


@pytest.mark.api
def test_model_versions_come_in_pages(service: Service) -> None:
    sign_in(service)
    with service.store.unit_of_work() as uow:
        for number in range(ROWS + 1):
            uow.model_versions.add(
                make_model_version(
                    f"v{number}",
                    status=ModelVersionStatus.CANDIDATE,
                    created_at=T0 + timedelta(hours=number),
                )
            )
        uow.commit()

    second = page(get(service, "/ui/models?page=2"))

    assert [str(cell) for cell in second.xpath("//tbody/tr/td[1]/code/text()")] == ["v0"]
    assert pager_links(service, "/ui/models?page=2") == ["/ui/models?page=1"]


@pytest.mark.api
def test_training_jobs_come_in_pages(service: Service) -> None:
    sign_in(service)
    for number in range(1, ROWS + 2):
        service.store.training_jobs[UUID(int=number)] = make_job(
            UUID(int=number), at=T0 + timedelta(minutes=number)
        )

    first = page(get(service, "/ui/jobs"))
    second = page(get(service, "/ui/jobs?page=2"))

    assert len(first.xpath("//tbody/tr")) == ROWS
    assert [" ".join(str(second.xpath("string(//tbody/tr/td[1])")).split())] == [
        "2026-09-16 12:01:00 UTC"
    ]
    assert pager_links(service, "/ui/jobs") == ["/ui/jobs?page=2"]
