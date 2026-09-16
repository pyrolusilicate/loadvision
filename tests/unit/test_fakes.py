from datetime import timedelta
from uuid import UUID

import pytest

from loadvision.domain.assessment import LoadPercent
from loadvision.domain.model_version import ModelVersionStatus
from loadvision.domain.shipment import ShipmentId
from tests.builders import T0, make_image, make_model_version
from tests.fakes.clock import FixedClock, SequentialIds
from tests.fakes.in_memory_db import InMemoryStore
from tests.fakes.predictor import FakePredictor


@pytest.mark.unit
def test_predictor_uses_the_preset_load_and_records_batches() -> None:
    image = make_image()
    predictor = FakePredictor(loads={image.sha256: 42.0})
    assert predictor.predict([image])[0].load_pct == LoadPercent(42.0)
    assert predictor.batches == [[image.sha256]]


@pytest.mark.unit
def test_predictor_derives_different_repeatable_loads_from_hashes() -> None:
    first, second = make_image(b"photo-a"), make_image(b"photo-b")
    loads = [prediction.load_pct for prediction in FakePredictor().predict([first, second])]
    again = [prediction.load_pct for prediction in FakePredictor().predict([first, second])]
    assert loads == again
    assert loads[0] != loads[1]


@pytest.mark.unit
def test_predictor_error_fails_the_batch_after_recording_it() -> None:
    image = make_image()
    predictor = FakePredictor(error=RuntimeError("model crashed"))
    with pytest.raises(RuntimeError, match="model crashed"):
        predictor.predict([image])
    assert predictor.batches == [[image.sha256]]


@pytest.mark.unit
def test_unit_of_work_cannot_be_used_outside_its_block() -> None:
    uow = InMemoryStore().unit_of_work()
    with pytest.raises(RuntimeError, match="outside"):
        uow.shipments.get(ShipmentId("RP-1"))


@pytest.mark.unit
def test_second_active_model_version_is_rejected_like_the_database_index() -> None:
    active = ModelVersionStatus.ACTIVE
    with InMemoryStore().unit_of_work() as uow:
        uow.model_versions.add(make_model_version("v1", status=active, activated_at=T0))
        with pytest.raises(ValueError, match="already active"):
            uow.model_versions.add(make_model_version("v2", status=active, activated_at=T0))


@pytest.mark.unit
def test_model_version_repository_rejects_double_registration_and_unknown_update() -> None:
    with InMemoryStore().unit_of_work() as uow:
        uow.model_versions.add(make_model_version("v1"))
        with pytest.raises(ValueError, match="already registered"):
            uow.model_versions.add(make_model_version("v1"))
        with pytest.raises(ValueError, match="not registered"):
            uow.model_versions.update(make_model_version("v2"))


@pytest.mark.unit
def test_fixed_clock_moves_only_when_advanced() -> None:
    clock = FixedClock(T0)
    assert clock() == T0
    clock.advance(timedelta(minutes=5))
    assert clock() == T0 + timedelta(minutes=5)


@pytest.mark.unit
def test_sequential_ids_repeat_across_factories() -> None:
    first, second = SequentialIds(), SequentialIds()
    assert [first(), first()] == [UUID(int=1), UUID(int=2)]
    assert second() == UUID(int=1)
