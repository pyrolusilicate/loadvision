import threading
import time
from collections.abc import Callable
from datetime import timedelta

import pytest

from loadvision.application.load_served_model import LoadServedModel
from loadvision.application.model_versions import ActivateModel
from loadvision.application.ports import LoadPredictor, UnitOfWork
from loadvision.domain.model_version import ModelVersionId
from loadvision.ml.inference.active_model import ActiveModel
from loadvision.ml.inference.model_follower import ModelFollower
from tests.builders import T0, make_model_version
from tests.fakes.clock import FixedClock
from tests.fakes.in_memory_db import InMemoryStore
from tests.fakes.predictor import FakePredictor

INITIAL = ModelVersionId("baseline-v0")
RETRY_SECONDS = 60.0
WAIT_SECONDS = 5.0


class Seconds:
    """Monotonic clock that stands still until a test moves it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Packages:
    """Package loader: versions in `missing` fail, the rest load; every call is recorded."""

    def __init__(self, *missing: str) -> None:
        self.missing = set(missing)
        self.loaded: list[str] = []
        self.gate: threading.Event | None = None

    def __call__(self, version: ModelVersionId) -> LoadPredictor:
        self.loaded.append(version.value)
        if self.gate is not None:
            self.gate.wait(WAIT_SECONDS)
        if version.value in self.missing:
            raise FileNotFoundError(f"storage/models/{version.value}/manifest.json")
        return FakePredictor(version=version.value)


class Service:
    """A store with registered versions, the served model and its follower."""

    def __init__(self, packages: Packages, *versions: str) -> None:
        self.store = InMemoryStore()
        with self.store.unit_of_work() as uow:
            for tag in versions:
                uow.model_versions.add(make_model_version(tag))
            uow.commit()
        # The follower opens the database through this, so a test can make it fail.
        self.open: Callable[[], UnitOfWork] = self.store.unit_of_work
        self.model = ActiveModel()
        self.seconds = Seconds()
        self.follower = ModelFollower(
            self.model,
            LoadServedModel(
                uow_factory=lambda: self.open(),
                clock=FixedClock(T0),
                load_package=packages,
                initial_version=INITIAL,
            ),
            check_seconds=0.01,
            retry_seconds=RETRY_SECONDS,
            clock=self.seconds,
        )

    def activate(self, tag: str) -> None:
        """Activates a version the way a command or the worker does: in the database only."""
        clock = FixedClock(T0 + timedelta(hours=1))
        ActivateModel(uow_factory=self.store.unit_of_work, clock=clock)(ModelVersionId(tag))

    def check(self) -> None:
        loading = self.follower.check()
        if loading is not None:
            loading.join(WAIT_SECONDS)

    def serves(self) -> str | None:
        current = self.model.current()
        return None if current is None else current.version.value


@pytest.mark.unit
def test_the_first_check_loads_the_initial_version_into_an_empty_database() -> None:
    packages = Packages()
    service = Service(packages)

    service.check()

    assert service.serves() == "baseline-v0"
    assert packages.loaded == ["baseline-v0"]


@pytest.mark.unit
def test_a_version_activated_elsewhere_is_loaded_at_the_next_check() -> None:
    packages = Packages()
    service = Service(packages, "v1", "v2")
    service.activate("v1")
    service.check()

    service.activate("v2")
    service.check()

    assert service.serves() == "v2"
    assert packages.loaded == ["v1", "v2"]


@pytest.mark.unit
def test_nothing_is_loaded_while_the_active_version_answers() -> None:
    packages = Packages()
    service = Service(packages, "v1")
    service.activate("v1")
    service.check()
    # Past the retry pause, so only the answering version keeps the check from loading.
    service.seconds.now += RETRY_SECONDS

    assert service.follower.check() is None
    assert packages.loaded == ["v1"]


@pytest.mark.unit
def test_a_failed_version_is_tried_again_only_after_the_pause() -> None:
    packages = Packages("v2")
    service = Service(packages, "v1", "v2")
    service.activate("v1")
    service.check()
    service.activate("v2")
    service.check()

    service.seconds.now += RETRY_SECONDS - 1
    service.check()
    early = list(packages.loaded)
    packages.missing.clear()
    service.seconds.now += 1
    service.check()

    assert early == ["v1", "v2"]
    assert packages.loaded == ["v1", "v2", "v2"]
    assert service.serves() == "v2"


@pytest.mark.unit
def test_the_previous_version_answers_while_the_new_one_fails() -> None:
    packages = Packages("v2")
    service = Service(packages, "v1", "v2")
    service.activate("v1")
    service.check()

    service.activate("v2")
    service.check()

    assert service.serves() == "v1"
    assert service.model.failure == "FileNotFoundError: storage/models/v2/manifest.json"


@pytest.mark.unit
def test_another_activation_is_loaded_at_once_even_after_a_failure() -> None:
    packages = Packages("v2")
    service = Service(packages, "v1", "v2", "v3")
    service.activate("v1")
    service.check()
    service.activate("v2")
    service.check()

    service.activate("v3")
    service.check()

    assert service.serves() == "v3"


@pytest.mark.unit
def test_one_loading_runs_at_a_time() -> None:
    packages = Packages()
    packages.gate = threading.Event()
    service = Service(packages, "v1")
    service.activate("v1")

    first = service.follower.check()
    # Past the retry pause, so only the running loading keeps the check from starting another.
    service.seconds.now += RETRY_SECONDS
    second = service.follower.check()
    packages.gate.set()

    assert first is not None
    first.join(WAIT_SECONDS)
    assert second is None
    assert packages.loaded == ["v1"]


@pytest.mark.unit
def test_background_checks_survive_a_database_failure_and_stop_on_request() -> None:
    packages = Packages()
    service = Service(packages, "v1", "v2")
    service.activate("v1")
    calls = 0

    def flaky() -> UnitOfWork:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("the database restarts")
        return service.store.unit_of_work()

    service.open = flaky
    service.follower.start()
    deadline = time.monotonic() + WAIT_SECONDS
    while service.serves() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    service.follower.stop()
    service.activate("v2")
    time.sleep(0.05)

    assert service.serves() == "v1"
    assert calls >= 2
