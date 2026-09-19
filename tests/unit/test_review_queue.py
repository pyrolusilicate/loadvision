from datetime import timedelta
from typing import Any
from uuid import UUID

import pytest

from loadvision.application.paging import Page
from loadvision.application.ports import AssessmentOrder, ReviewStatus
from loadvision.application.review_queue import ReviewQueue
from tests.builders import T0, make_assessment, make_image, make_label
from tests.fakes.in_memory_db import InMemoryStore
from tests.port_contracts.unit_of_work import store_assessment

PHOTO_A = b"photo-a"
PHOTO_B = b"photo-b"


@pytest.fixture
def store() -> InMemoryStore:
    store = InMemoryStore()
    with store.unit_of_work() as uow:
        store_assessment(uow, make_assessment(id=UUID(int=1), created_at=T0), PHOTO_A)
        store_assessment(
            uow,
            make_assessment(
                id=UUID(int=2),
                image_sha256=make_image(PHOTO_B).sha256,
                created_at=T0 + timedelta(hours=1),
            ),
            PHOTO_B,
        )
        uow.labels.save(make_label(PHOTO_A, 70.0))
        uow.commit()
    return store


def queue(store: InMemoryStore, page: int = 1, size: int = 10, **options: Any) -> Page[Any]:
    return ReviewQueue(uow_factory=store.unit_of_work)(page=page, size=size, **options)


@pytest.mark.unit
def test_the_queue_holds_only_photos_nobody_confirmed(store: InMemoryStore) -> None:
    [waiting] = queue(store).items

    assert (waiting.assessment.id, waiting.label) == (UUID(int=2), None)


@pytest.mark.unit
def test_the_whole_list_shows_labels_newest_first(store: InMemoryStore) -> None:
    everything = queue(store, status=ReviewStatus.ALL).items

    assert [item.assessment.id for item in everything] == [UUID(int=2), UUID(int=1)]
    assert [item.label for item in everything] == [None, make_label(PHOTO_A, 70.0)]


@pytest.mark.unit
def test_the_order_is_passed_through(store: InMemoryStore) -> None:
    oldest = queue(store, status=ReviewStatus.ALL, order=AssessmentOrder.OLDEST).items

    assert [item.assessment.id for item in oldest] == [UUID(int=1), UUID(int=2)]


@pytest.mark.unit
def test_a_page_knows_its_place_in_the_list(store: InMemoryStore) -> None:
    second = queue(store, page=2, size=1, status=ReviewStatus.ALL)

    assert [item.assessment.id for item in second.items] == [UUID(int=1)]
    assert (second.number, second.pages, second.total) == (2, 2, 2)


@pytest.mark.unit
def test_a_page_past_the_end_shows_the_last_one(store: InMemoryStore) -> None:
    last = queue(store, page=9, size=1, status=ReviewStatus.ALL)

    assert (last.number, [item.assessment.id for item in last.items]) == (2, [UUID(int=1)])
