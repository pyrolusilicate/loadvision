from datetime import timedelta
from uuid import UUID

import pytest

from loadvision.application.submit_label import SubmitLabel, SubmitLabelRequest
from loadvision.domain.assessment import LoadPercent
from loadvision.domain.confirmed_label import LabelSource
from loadvision.domain.shipment import ShipmentId
from loadvision.errors import ErrorCode, LoadvisionError
from tests.builders import T0, make_assessment, make_image
from tests.fakes.clock import FixedClock
from tests.fakes.in_memory_db import InMemoryStore
from tests.port_contracts.unit_of_work import store_assessment

FIRST_ID = UUID(int=1)
SECOND_ID = UUID(int=2)
PHOTO_SHA256 = make_image().sha256


@pytest.fixture
def store() -> InMemoryStore:
    store = InMemoryStore()
    with store.unit_of_work() as uow:
        store_assessment(uow, make_assessment(id=FIRST_ID))
        store_assessment(uow, make_assessment(id=SECOND_ID, shipment_id=ShipmentId("RP-2")))
        uow.commit()
    return store


def request(
    load_pct: float | None = 70.0, assessment_id: UUID = FIRST_ID, author: str | None = "op-17"
) -> SubmitLabelRequest:
    return SubmitLabelRequest(
        assessment_id=assessment_id,
        load_pct=None if load_pct is None else LoadPercent(load_pct),
        source=LabelSource.API,
        author=author,
    )


@pytest.mark.unit
def test_first_label_of_a_photo_is_stored(store: InMemoryStore) -> None:
    result = SubmitLabel(uow_factory=store.unit_of_work, clock=FixedClock(T0))(request())

    assert result.assessment.id == FIRST_ID
    assert result.label.image_sha256 == PHOTO_SHA256
    assert result.label.load_pct == LoadPercent(70.0)
    assert (result.label.created_at, result.label.updated_at) == (T0, T0)
    assert store.labels == {PHOTO_SHA256: result.label}


@pytest.mark.unit
def test_repeated_label_replaces_the_verdict_and_keeps_the_creation_time(
    store: InMemoryStore,
) -> None:
    clock = FixedClock(T0)
    submit = SubmitLabel(uow_factory=store.unit_of_work, clock=clock)
    submit(request(70.0))
    clock.advance(timedelta(minutes=10))

    result = submit(request(None, assessment_id=SECOND_ID))

    assert result.label.unusable is True
    assert (result.label.created_at, result.label.updated_at) == (T0, T0 + timedelta(minutes=10))
    assert store.labels == {PHOTO_SHA256: result.label}


@pytest.mark.unit
@pytest.mark.parametrize("author", [None, "", "   "])
def test_missing_or_blank_author_is_stored_as_unknown(
    store: InMemoryStore, author: str | None
) -> None:
    result = SubmitLabel(uow_factory=store.unit_of_work, clock=FixedClock(T0))(
        request(author=author)
    )
    assert result.label.author is None


@pytest.mark.unit
def test_author_is_trimmed(store: InMemoryStore) -> None:
    result = SubmitLabel(uow_factory=store.unit_of_work, clock=FixedClock(T0))(
        request(author="  op-17 ")
    )
    assert result.label.author == "op-17"


@pytest.mark.unit
def test_unknown_assessment_is_not_found_and_nothing_is_stored(store: InMemoryStore) -> None:
    submit = SubmitLabel(uow_factory=store.unit_of_work, clock=FixedClock(T0))
    with pytest.raises(LoadvisionError) as caught:
        submit(request(assessment_id=UUID(int=99)))
    assert caught.value.code is ErrorCode.ASSESSMENT_NOT_FOUND
    assert store.labels == {}
