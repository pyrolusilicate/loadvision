from datetime import timedelta

import pytest

from loadvision.application.model_versions import ActivateModel, RegisterModel, RollbackModel
from loadvision.domain.model_version import ModelVersion, ModelVersionId, ModelVersionStatus
from loadvision.errors import ErrorCode, LoadvisionError
from tests.builders import T0, make_model_version
from tests.fakes.clock import FixedClock
from tests.fakes.in_memory_db import InMemoryStore

V1 = ModelVersionId("v1")
V2 = ModelVersionId("v2")


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(T0)


def register(store: InMemoryStore, clock: FixedClock) -> RegisterModel:
    return RegisterModel(uow_factory=store.unit_of_work, clock=clock)


def activate(store: InMemoryStore, clock: FixedClock) -> ActivateModel:
    return ActivateModel(uow_factory=store.unit_of_work, clock=clock)


@pytest.mark.unit
def test_registered_version_is_a_candidate(store: InMemoryStore, clock: FixedClock) -> None:
    registered = register(store, clock)(V1)

    assert registered == ModelVersion(
        version=V1, status=ModelVersionStatus.CANDIDATE, created_at=T0
    )
    assert store.model_versions == {V1: registered}


@pytest.mark.unit
def test_registering_the_same_version_twice_is_refused(
    store: InMemoryStore, clock: FixedClock
) -> None:
    register(store, clock)(V1)

    with pytest.raises(ValueError, match="already registered"):
        register(store, clock)(V1)


@pytest.mark.unit
def test_activation_retires_the_previous_version(store: InMemoryStore, clock: FixedClock) -> None:
    register(store, clock)(V1)
    register(store, clock)(V2)
    activate(store, clock)(V1)
    clock.advance(timedelta(hours=1))

    active = activate(store, clock)(V2)

    assert active.version == V2
    assert active.activated_at == T0 + timedelta(hours=1)
    assert store.model_versions[V1].status is ModelVersionStatus.RETIRED
    assert store.model_versions[V1].activated_at == T0


@pytest.mark.unit
def test_activating_the_active_version_changes_nothing(
    store: InMemoryStore, clock: FixedClock
) -> None:
    register(store, clock)(V1)
    first = activate(store, clock)(V1)
    clock.advance(timedelta(hours=1))

    assert activate(store, clock)(V1) == first
    assert store.model_versions[V1] == first


@pytest.mark.unit
def test_unregistered_version_is_not_activated(store: InMemoryStore, clock: FixedClock) -> None:
    with pytest.raises(LoadvisionError) as raised:
        activate(store, clock)(V1)
    assert raised.value.code is ErrorCode.MODEL_VERSION_NOT_FOUND
    assert store.model_versions == {}


@pytest.mark.unit
def test_the_service_may_register_the_version_it_starts_with(
    store: InMemoryStore, clock: FixedClock
) -> None:
    active = activate(store, clock)(V1, register=True)

    assert active.status is ModelVersionStatus.ACTIVE
    assert store.model_versions[V1] == active


@pytest.mark.unit
def test_rejected_version_never_answers(store: InMemoryStore, clock: FixedClock) -> None:
    with store.unit_of_work() as uow:
        uow.model_versions.add(make_model_version("v1", status=ModelVersionStatus.REJECTED))
        uow.commit()

    with pytest.raises(LoadvisionError) as raised:
        activate(store, clock)(V1)
    assert raised.value.code is ErrorCode.MODEL_VERSION_STATUS_CONFLICT


@pytest.mark.unit
def test_rollback_brings_back_the_version_that_retired_last(
    store: InMemoryStore, clock: FixedClock
) -> None:
    for tag in ("v1", "v2", "v3"):
        register(store, clock)(ModelVersionId(tag))
        activate(store, clock)(ModelVersionId(tag))
        clock.advance(timedelta(hours=1))

    back = RollbackModel(uow_factory=store.unit_of_work, activate=activate(store, clock))()

    assert back.version == V2
    assert store.model_versions[ModelVersionId("v3")].status is ModelVersionStatus.RETIRED
    assert store.model_versions[V2].status is ModelVersionStatus.ACTIVE


@pytest.mark.unit
def test_rollback_without_a_retired_version_is_refused(
    store: InMemoryStore, clock: FixedClock
) -> None:
    register(store, clock)(V1)
    activate(store, clock)(V1)

    rollback = RollbackModel(uow_factory=store.unit_of_work, activate=activate(store, clock))
    with pytest.raises(LoadvisionError) as raised:
        rollback()
    assert raised.value.code is ErrorCode.MODEL_VERSION_NOT_FOUND
