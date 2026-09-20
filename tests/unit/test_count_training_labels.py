from datetime import timedelta

import pytest

from loadvision.application.training_jobs import CountTrainingLabels
from loadvision.domain.model_version import ModelVersionStatus
from tests.builders import T0, make_label, make_model_version
from tests.fakes.in_memory_db import InMemoryStore

BUILT = T0 + timedelta(hours=1)


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def count(store: InMemoryStore) -> int:
    return CountTrainingLabels(uow_factory=store.unit_of_work)()


def activate(store: InMemoryStore) -> None:
    """Puts an active version built an hour after T0 into the store."""
    with store.unit_of_work() as uow:
        uow.model_versions.add(
            make_model_version(
                "v2", status=ModelVersionStatus.ACTIVE, created_at=BUILT, activated_at=BUILT
            )
        )
        uow.commit()


def label(store: InMemoryStore, raw: bytes, *, at, load_pct: float | None = 60.0) -> None:
    with store.unit_of_work() as uow:
        uow.labels.save(make_label(raw, load_pct, at=at))
        uow.commit()


@pytest.mark.unit
def test_without_an_active_version_every_confirmed_load_is_new(store: InMemoryStore) -> None:
    label(store, b"photo-a", at=T0)
    label(store, b"photo-b", at=BUILT + timedelta(minutes=1))

    assert count(store) == 2


@pytest.mark.unit
def test_loads_the_active_version_learned_from_do_not_count(store: InMemoryStore) -> None:
    label(store, b"photo-a", at=T0)
    label(store, b"photo-b", at=BUILT + timedelta(minutes=1))
    activate(store)

    assert count(store) == 1


@pytest.mark.unit
def test_a_photo_confirmed_again_counts_again(store: InMemoryStore) -> None:
    label(store, b"photo-a", at=T0)
    activate(store)
    assert count(store) == 0

    # The operator corrected the load after the version was built.
    label(store, b"photo-a", at=BUILT + timedelta(minutes=5))

    assert count(store) == 1


@pytest.mark.unit
def test_a_load_confirmed_at_the_very_moment_of_the_build_is_counted_as_seen(
    store: InMemoryStore,
) -> None:
    # The build reads the labels it starts with, so one stamped with its time went into it.
    label(store, b"photo-a", at=BUILT)
    activate(store)

    assert count(store) == 0


@pytest.mark.unit
def test_an_unusable_photo_teaches_nothing(store: InMemoryStore) -> None:
    label(store, b"photo-a", at=BUILT + timedelta(minutes=1), load_pct=None)
    activate(store)

    assert count(store) == 0
