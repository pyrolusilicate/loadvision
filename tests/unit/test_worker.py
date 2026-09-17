import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from loadvision.application.ports import TrainingOutcome, TrainingRequest, TrainingStatus
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.training_job import TrainingJob, TrainingMode
from loadvision.errors import ErrorCode
from loadvision.infrastructure.storage.local_storage import LocalImageStorage
from loadvision.settings import Settings
from loadvision.worker.main import ActivePackageRunner, publish_package, serve
from tests.builders import make_job
from tests.fakes.in_memory_db import InMemoryStore
from tests.model_packages import build_numpy_package
from tests.settings_files import read_settings

VERSION = "2026-09-18.1"


class Rounds:
    """Stands for the use case: returns what a test lined up, one round at a time."""

    def __init__(self, *results: TrainingJob | Exception | None) -> None:
        self._results = list(results)
        self.calls = 0

    def __call__(self) -> TrainingJob | None:
        self.calls += 1
        if not self._results:
            return None
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.unit
def test_once_runs_a_single_round() -> None:
    rounds = Rounds(make_job(), make_job(), make_job())
    stop = threading.Event()

    def stop_after_three() -> TrainingJob | None:
        # Without --once the loop would go on; the event bounds the test instead of hanging.
        job = rounds()
        if rounds.calls >= 3:
            stop.set()
        return job

    serve(stop_after_three, poll_seconds=0.01, once=True, stop=stop)

    assert rounds.calls == 1


@pytest.mark.unit
def test_the_loop_ends_when_it_is_asked_to_stop() -> None:
    stop = threading.Event()

    def one_then_stop() -> TrainingJob | None:
        stop.set()
        return make_job()

    serve(one_then_stop, poll_seconds=0.01, stop=stop)


@pytest.mark.unit
def test_a_failed_round_does_not_stop_the_worker(caplog: pytest.LogCaptureFixture) -> None:
    stop = threading.Event()
    rounds = Rounds(RuntimeError("database is down"), make_job())

    def until_the_second_round() -> TrainingJob | None:
        job = rounds()
        if rounds.calls >= 2:
            stop.set()
        return job

    serve(until_the_second_round, poll_seconds=0.01, stop=stop)

    assert rounds.calls == 2
    assert "job failed unexpectedly" in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return read_settings(
        storage={"images_dir": str(tmp_path / "images"), "models_dir": str(tmp_path / "models")},
        runs={"dir": str(tmp_path / "runs")},
        training={
            "config": "configs/experiments/ens4_apache_by_aspect.json",
            "data_dir": str(tmp_path / "data"),
            "device": "mps",
            "poll_seconds": 1,
        },
    )


@pytest.mark.unit
def test_a_finished_package_lands_in_the_models_folder_and_the_database(
    settings: Settings, tmp_path: Path
) -> None:
    store = InMemoryStore()
    package = build_numpy_package(tmp_path / "run", load=55.0, version=VERSION)

    version = publish_package(settings, store.unit_of_work, package)

    assert version == ModelVersionId(VERSION)
    assert (settings.storage.models_dir / VERSION / "manifest.json").is_file()
    assert store.model_versions[version].status.value == "candidate"


@pytest.mark.unit
def test_the_pipeline_is_pointed_at_the_package_it_has_to_beat(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: list[dict[str, Any]] = []

    class RecordingRunner:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

        def run(self, request: TrainingRequest) -> TrainingOutcome:
            return TrainingOutcome(
                run_id="run-1",
                status=TrainingStatus.FAILED,
                error_code=ErrorCode.INTERNAL_ERROR,
            )

    monkeypatch.setattr("loadvision.worker.main.PipelineTrainingRunner", RecordingRunner)
    runner = ActivePackageRunner(
        settings=settings,
        storage=LocalImageStorage(settings.storage.images_dir),
        config={"members": []},
    )

    for base in (None, ModelVersionId("baseline-v0")):
        runner.run(TrainingRequest(mode=TrainingMode.FAST, base_version=base))

    assert built[0]["active_package"] is None
    assert built[1]["active_package"] == settings.storage.models_dir / "baseline-v0"
    assert built[0]["device"] == "mps"
    assert built[0]["data_dir"] == settings.training.data_dir
    assert built[0]["cache_dir"] == settings.runs.dir / "cache"
