"""Composition root of the worker process: it runs training jobs, one at a time.

Run with ``loadvision-worker``; ``--once`` runs a single job and exits, which is what a cron
entry or a test needs.
"""

import argparse
import functools
import json
import logging
import signal
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType

from sqlalchemy import Engine

from loadvision.application.model_versions import ActivateModel, RegisterModel
from loadvision.application.ports import (
    TrainingOutcome,
    TrainingRequest,
    UnitOfWorkFactory,
)
from loadvision.application.run_training_job import RunTrainingJob
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.timestamps import utc_now
from loadvision.domain.training_job import TrainingJob
from loadvision.infrastructure.db.activity import WORKER_APPLICATION
from loadvision.infrastructure.db.session import PostgresUnitOfWork, create_database_engine
from loadvision.infrastructure.storage.local_storage import LocalImageStorage
from loadvision.ml.devices import resolve_device
from loadvision.ml.packaging.model_store import place_package
from loadvision.ml.training.runner import PipelineTrainingRunner
from loadvision.observability.logging import configure_logging
from loadvision.settings import Settings, database_url, load_settings

_logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the worker until it is asked to stop.

    Returns:
        Exit code 0.

    Raises:
        RuntimeError: If `DATABASE_URL` is not set.
        OSError: If the settings file cannot be read.
    """
    parser = argparse.ArgumentParser(
        prog="loadvision-worker", description="Runs queued retrainings, one at a time."
    )
    parser.add_argument("--once", action="store_true", help="run one job, then exit")
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.logging.format, settings.logging.level)
    engine = create_database_engine(database_url(), application_name=WORKER_APPLICATION)
    try:
        run_job = build_worker(settings, engine)
        serve(run_job, poll_seconds=settings.training.poll_seconds, once=args.once)
    finally:
        engine.dispose()
    return 0


def build_worker(settings: Settings, engine: Engine) -> Callable[[], TrainingJob | None]:
    """Wires the use case that runs one job to Postgres, the photo storage and the pipeline.

    Args:
        settings: Service settings.
        engine: Engine of the service database.

    Returns:
        A callable that runs the next job, or returns None when there is nothing to run.

    Raises:
        OSError: If the experiment config cannot be read.
        json.JSONDecodeError: If it is not valid JSON.
    """
    uow_factory: UnitOfWorkFactory = functools.partial(PostgresUnitOfWork, engine)
    config = json.loads(settings.training.config.read_text(encoding="utf-8"))
    runner = ActivePackageRunner(
        settings=settings,
        storage=LocalImageStorage(settings.storage.images_dir),
        config=config,
    )
    return RunTrainingJob(
        uow_factory=uow_factory,
        clock=utc_now,
        runner=runner,
        publish=functools.partial(publish_package, settings, uow_factory),
        activate=ActivateModel(uow_factory=uow_factory, clock=utc_now),
    )


def serve(
    run_job: Callable[[], TrainingJob | None],
    *,
    poll_seconds: float,
    once: bool = False,
    stop: threading.Event | None = None,
) -> None:
    """Runs jobs until asked to stop, waiting between empty rounds.

    A failed job is recorded by the use case, so the loop only logs what it could not handle
    and goes on: a worker that dies on one bad job stops every later one.

    Args:
        run_job: Runs the next job, or returns None when the queue is empty.
        poll_seconds: Wait between rounds that found nothing.
        once: Stop after the first round, whether it found a job or not.
        stop: Event that ends the loop; by default SIGINT and SIGTERM set it.
    """
    stop = _stop_on_signals() if stop is None else stop
    while not stop.is_set():
        try:
            job = run_job()
        except Exception:
            _logger.exception("job failed unexpectedly")
            job = None
        if once:
            return
        if job is None:
            stop.wait(poll_seconds)


def publish_package(
    settings: Settings, uow_factory: UnitOfWorkFactory, package_dir: Path
) -> ModelVersionId:
    """Puts a package the pipeline produced into the models folder and registers its version.

    Args:
        settings: Service settings, for the models folder.
        uow_factory: Opens database transactions.
        package_dir: Package of the finished run.

    Returns:
        The registered version.

    Raises:
        ModelPackageError: If the package is invalid.
        FileExistsError: If the models folder already holds this version.
        OSError: If the package cannot be copied.
    """
    package = place_package(package_dir, settings.storage.models_dir)
    RegisterModel(uow_factory=uow_factory, clock=utc_now)(package.version)
    _logger.info("registered %s from %s", package.version.value, package_dir)
    return package.version


class ActivePackageRunner:
    """`TrainingRunner` that points the pipeline at the package of the active version.

    The active version changes between jobs, so the pipeline runner is built for every
    request, with the package the candidate has to beat.

    Args:
        settings: Service settings.
        storage: Where the bytes of confirmed photos live.
        config: Experiment config to retrain with.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        storage: LocalImageStorage,
        config: dict[str, object],
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._config = config

    def run(self, request: TrainingRequest) -> TrainingOutcome:
        """Retrains and compares the candidate with the active package."""
        training = self._settings.training
        active = (
            None
            if request.base_version is None
            else self._settings.storage.models_dir / request.base_version.value
        )
        runner = PipelineTrainingRunner(
            config=self._config,
            data_dir=training.data_dir,
            runs_dir=self._settings.runs.dir,
            cache_dir=self._settings.runs.cache_dir,
            storage=self._storage,
            clock=utc_now,
            active_package=active,
            device=resolve_device(training.device),
            foreign_dir=training.foreign_dir,
        )
        return runner.run(request)


def _stop_on_signals() -> threading.Event:
    stop = threading.Event()

    def ask_to_stop(number: int, _frame: FrameType | None) -> None:
        _logger.info("signal %s: finishing the current job and stopping", number)
        stop.set()

    signal.signal(signal.SIGINT, ask_to_stop)
    signal.signal(signal.SIGTERM, ask_to_stop)
    return stop
