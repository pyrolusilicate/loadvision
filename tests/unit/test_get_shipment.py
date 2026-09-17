from datetime import timedelta
from uuid import UUID

import pytest

from loadvision.application.get_shipment import GetShipment, ShipmentHistory
from loadvision.domain.shipment import ShipmentId
from loadvision.errors import ErrorCode, LoadvisionError
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
        store_assessment(uow, make_assessment(id=UUID(int=3), shipment_id=ShipmentId("RP-2")))
        uow.labels.save(make_label(PHOTO_A, 90.0))
        uow.commit()
    return store


@pytest.mark.unit
def test_history_is_newest_first_and_each_assessment_has_the_label_of_its_photo(
    store: InMemoryStore,
) -> None:
    history = GetShipment(uow_factory=store.unit_of_work)("RP-1")

    assert history.shipment_id == ShipmentId("RP-1")
    assert [item.assessment.id for item in history.assessments] == [UUID(int=2), UUID(int=1)]
    assert [item.label for item in history.assessments] == [None, make_label(PHOTO_A, 90.0)]
    assert history.current == history.assessments[0]


@pytest.mark.unit
def test_label_of_a_photo_shows_in_every_shipment_with_that_photo(store: InMemoryStore) -> None:
    history = GetShipment(uow_factory=store.unit_of_work)("RP-2")
    assert history.current.label == make_label(PHOTO_A, 90.0)


@pytest.mark.unit
def test_unknown_shipment_is_not_found(store: InMemoryStore) -> None:
    with pytest.raises(LoadvisionError) as raised:
        GetShipment(uow_factory=store.unit_of_work)("RP-404")
    assert raised.value.code is ErrorCode.SHIPMENT_NOT_FOUND


@pytest.mark.unit
def test_malformed_shipment_number_is_rejected(store: InMemoryStore) -> None:
    with pytest.raises(LoadvisionError) as raised:
        GetShipment(uow_factory=store.unit_of_work)("RP 1")
    assert raised.value.code is ErrorCode.INVALID_SHIPMENT_ID


@pytest.mark.unit
def test_history_without_assessments_is_refused() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ShipmentHistory(shipment_id=ShipmentId("RP-1"), assessments=())
