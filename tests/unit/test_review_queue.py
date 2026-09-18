from datetime import timedelta
from uuid import UUID

import pytest

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


@pytest.mark.unit
def test_the_queue_holds_only_photos_nobody_confirmed(store: InMemoryStore) -> None:
    [waiting] = ReviewQueue(uow_factory=store.unit_of_work)()

    assert (waiting.assessment.id, waiting.label) == (UUID(int=2), None)


@pytest.mark.unit
def test_the_whole_list_shows_labels_newest_first(store: InMemoryStore) -> None:
    everything = ReviewQueue(uow_factory=store.unit_of_work)(unlabeled_only=False)

    assert [item.assessment.id for item in everything] == [UUID(int=2), UUID(int=1)]
    assert [item.label for item in everything] == [None, make_label(PHOTO_A, 70.0)]


@pytest.mark.unit
def test_the_queue_is_cut_at_the_limit(store: InMemoryStore) -> None:
    assert len(ReviewQueue(uow_factory=store.unit_of_work)(1, unlabeled_only=False)) == 1
