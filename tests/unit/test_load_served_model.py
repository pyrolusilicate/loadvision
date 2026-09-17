from datetime import timedelta

import pytest

from loadvision.application.load_served_model import LoadServedModel
from loadvision.application.ports import LoadPredictor
from loadvision.domain.model_version import ModelVersion, ModelVersionId, ModelVersionStatus
from loadvision.errors import ErrorCode, LoadvisionError
from tests.builders import T0, make_model_version
from tests.fakes.clock import FixedClock
from tests.fakes.in_memory_db import InMemoryStore
from tests.fakes.predictor import FakePredictor

NOW = T0 + timedelta(days=1)
INITIAL = ModelVersionId("baseline-v0")


class Packages:
    """Package loader that builds fake predictors and records which versions it loaded."""

    def __init__(self, declared: dict[str, str] | None = None) -> None:
        self._declared = declared or {}
        self.loaded: list[ModelVersionId] = []

    def __call__(self, version: ModelVersionId) -> LoadPredictor:
        self.loaded.append(version)
        return FakePredictor(version=self._declared.get(version.value, version.value))


def broken_package(version: ModelVersionId) -> LoadPredictor:
    raise FileNotFoundError(f"storage/models/{version.value}/manifest.json")


def store_with(*versions: ModelVersion) -> InMemoryStore:
    store = InMemoryStore()
    with store.unit_of_work() as uow:
        for version in versions:
            uow.model_versions.add(version)
        uow.commit()
    return store


def load(store: InMemoryStore, packages: Packages) -> LoadPredictor:
    return LoadServedModel(
        uow_factory=store.unit_of_work,
        clock=FixedClock(NOW),
        load_package=packages,
        initial_version=INITIAL,
    )()


@pytest.mark.unit
def test_empty_database_gets_the_initial_version_registered_and_active() -> None:
    store = InMemoryStore()
    packages = Packages()

    predictor = load(store, packages)

    assert predictor.version == INITIAL
    assert packages.loaded == [INITIAL]
    assert store.model_versions == {
        INITIAL: ModelVersion(
            version=INITIAL, status=ModelVersionStatus.ACTIVE, created_at=NOW, activated_at=NOW
        )
    }


@pytest.mark.unit
def test_version_active_in_the_database_wins_over_the_initial_one() -> None:
    active = make_model_version("2026-09-20.1", status=ModelVersionStatus.ACTIVE, activated_at=T0)
    store = store_with(active)
    packages = Packages()

    predictor = load(store, packages)

    assert predictor.version == active.version
    assert packages.loaded == [active.version]
    assert store.model_versions == {active.version: active}


@pytest.mark.unit
def test_retired_initial_version_is_activated_again() -> None:
    retired = make_model_version(INITIAL.value, status=ModelVersionStatus.RETIRED, activated_at=T0)
    store = store_with(retired)

    load(store, Packages())

    assert store.model_versions[INITIAL] == retired.activate(NOW)


@pytest.mark.unit
def test_rejected_initial_version_is_not_activated() -> None:
    rejected = make_model_version(INITIAL.value, status=ModelVersionStatus.REJECTED)
    store = store_with(rejected)

    with pytest.raises(LoadvisionError) as raised:
        load(store, Packages())
    assert raised.value.code is ErrorCode.MODEL_VERSION_STATUS_CONFLICT
    assert store.model_versions == {INITIAL: rejected}


@pytest.mark.unit
def test_package_that_fails_to_load_is_not_registered() -> None:
    store = InMemoryStore()
    loader = LoadServedModel(
        uow_factory=store.unit_of_work,
        clock=FixedClock(NOW),
        load_package=broken_package,
        initial_version=INITIAL,
    )

    with pytest.raises(FileNotFoundError):
        loader()
    assert store.model_versions == {}


@pytest.mark.unit
def test_package_declaring_another_version_is_refused_and_not_registered() -> None:
    store = InMemoryStore()

    with pytest.raises(LoadvisionError) as raised:
        load(store, Packages(declared={INITIAL.value: "other-v1"}))
    assert raised.value.code is ErrorCode.MODEL_PACKAGE_INVALID
    assert store.model_versions == {}
