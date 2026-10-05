"""Composition root of the API process: settings, Postgres, photo storage and model packages.

Run with ``uvicorn --factory loadvision.api.main:create_service_app``.
"""

import functools
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

from fastapi import FastAPI
from sqlalchemy import Engine

from loadvision.api.app import create_app
from loadvision.api.services import ApiServices
from loadvision.api.ui.services import UiServices
from loadvision.application.assess_photo import AssessPhoto
from loadvision.application.explain_photo import ExplainPhoto
from loadvision.application.get_assessment import GetAssessment
from loadvision.application.get_shipment import GetShipment
from loadvision.application.load_served_model import LoadServedModel
from loadvision.application.model_versions import (
    ActivateModel,
    ListModelVersions,
    RollbackModel,
)
from loadvision.application.operators import Authenticate, SignIn, SignOut
from loadvision.application.ports import ExplanationKind, UnitOfWorkFactory
from loadvision.application.review_queue import ReviewQueue
from loadvision.application.submit_label import MarkReadable, SubmitLabel
from loadvision.application.training_jobs import (
    CountTrainingLabels,
    GetTrainingJob,
    ListTrainingJobPage,
    ListTrainingJobs,
    ReleaseTrainingJob,
    StartTraining,
)
from loadvision.application.weigh_cargo import WeighCargo
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.timestamps import utc_now
from loadvision.imaging.validator import prepare_image
from loadvision.infrastructure.db.activity import worker_connected
from loadvision.infrastructure.db.session import PostgresUnitOfWork, create_database_engine
from loadvision.infrastructure.storage.heatmap_store import HeatmapStore
from loadvision.infrastructure.storage.local_storage import LocalImageStorage
from loadvision.ml.devices import resolve_device
from loadvision.ml.inference.active_explainer import ActiveExplainer
from loadvision.ml.inference.active_model import ActiveModel
from loadvision.ml.inference.explainer import load_explainer
from loadvision.ml.inference.model_follower import ModelFollower
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.inference.stack import ActiveStackMeter, load_stack_meter
from loadvision.ml.packaging.model_package import ModelPackage
from loadvision.observability.logging import configure_logging
from loadvision.settings import Settings, database_url, load_settings

_logger = logging.getLogger(__name__)

# What a package's measured scale is kept in, next to the package folder.
SCALE_SUFFIX: Final = ".scale.json"


def create_service_app() -> FastAPI:
    """Builds the service from the settings file and `DATABASE_URL`.

    Returns:
        The application.

    Raises:
        RuntimeError: If `DATABASE_URL` is not set.
        OSError: If the settings file cannot be read.
        yaml.YAMLError: If the settings file is not valid YAML.
        pydantic.ValidationError: If a setting is missing, unknown or out of range.
        ValueError: If `model.version` is not a valid version tag.
    """
    settings = load_settings()
    configure_logging(settings.logging.format, settings.logging.level)
    return build_service_app(settings, create_database_engine(database_url()))


def build_service_app(settings: Settings, engine: Engine) -> FastAPI:
    """Wires the use cases to Postgres, the photo folder and the model packages.

    The service answers as soon as it starts. The model loads in the background, and until it
    is ready assessments and `/health` answer 503. Afterwards the service follows the version
    the database names active: a newly activated one loads in the background while the
    previous one keeps answering. The engine is closed when the service stops.

    Args:
        settings: Service settings.
        engine: Engine of the service database; the application disposes it on shutdown.

    Returns:
        The application.

    Raises:
        ValueError: If `model.version` is not a valid version tag.
    """
    uow_factory: UnitOfWorkFactory = functools.partial(PostgresUnitOfWork, engine)
    device = resolve_device(settings.model.device)
    _logger.info("the model runs on %s", device)
    model = ActiveModel()
    follower = ModelFollower(
        model,
        LoadServedModel(
            uow_factory=uow_factory,
            clock=utc_now,
            load_package=functools.partial(
                load_package, settings.storage.models_dir, device, settings.model.bf16
            ),
            initial_version=ModelVersionId(settings.model.version),
        ),
        check_seconds=settings.model.check_seconds,
        retry_seconds=settings.model.retry_seconds,
    )
    storage = LocalImageStorage(settings.storage.images_dir)
    heatmaps = HeatmapStore(settings.storage.heatmaps_dir)
    explainers = ActiveExplainer(
        functools.partial(load_explainer, settings.storage.models_dir, device)
    )
    explain_photo = ExplainPhoto(storage=storage, limits=settings.images.limits())

    stack_meters = ActiveStackMeter(
        functools.partial(load_stack_meter, settings.storage.models_dir, device)
    )

    def measure_stack(sha256: str, version: ModelVersionId) -> float | None:
        """Returns how high the cargo is stacked, or None when it cannot be read off the photo."""
        raw = storage.load(sha256)
        if raw is None:
            return None
        return stack_meters.measure(prepare_image(raw, settings.images.limits()), version)

    def draw_heatmap(sha256: str, kind: ExplanationKind, version: ModelVersionId) -> bytes:
        """Returns the map of what the model saw, drawing it only when it is not stored yet."""
        # The store answers before the explainer is built, so a map drawn before a restart
        # costs a file read and not another copy of the encoder.
        drawn = heatmaps.read(sha256, version, kind)
        if drawn is not None:
            return drawn
        png = explain_photo(sha256, kind, explainers.of(version))
        heatmaps.write(sha256, version, kind, png)
        return png

    services = ApiServices(
        assess_photo=AssessPhoto(
            uow_factory=uow_factory,
            storage=storage,
            limits=settings.images.limits(),
            min_brightness=settings.review.min_brightness,
            unusual_band_share=settings.review.unusual_band_share,
            unreadable_band_share=settings.review.unreadable_band_share,
            clock=utc_now,
            new_id=uuid.uuid4,
        ),
        get_assessment=GetAssessment(uow_factory=uow_factory),
        submit_label=SubmitLabel(uow_factory=uow_factory, clock=utc_now),
        get_shipment=GetShipment(uow_factory=uow_factory),
        list_models=ListModelVersions(uow_factory=uow_factory),
        activate_model=ActivateModel(uow_factory=uow_factory, clock=utc_now),
        start_training=StartTraining(uow_factory=uow_factory, clock=utc_now, new_id=uuid.uuid4),
        list_training_jobs=ListTrainingJobs(uow_factory=uow_factory),
        get_training_job=GetTrainingJob(uow_factory=uow_factory),
        model=model,
        reload_model=follower.check,
        request_id_header=settings.api.request_id_header,
        max_body_bytes=settings.api.max_body_bytes,
    )
    ui = UiServices(
        sign_in=SignIn(uow_factory=uow_factory, clock=utc_now),
        authenticate=Authenticate(uow_factory=uow_factory, clock=utc_now),
        sign_out=SignOut(uow_factory=uow_factory),
        assess_photo=services.assess_photo,
        review_queue=ReviewQueue(uow_factory=uow_factory),
        get_assessment=services.get_assessment,
        submit_label=services.submit_label,
        weigh_cargo=WeighCargo(uow_factory=uow_factory, clock=utc_now),
        mark_readable=MarkReadable(uow_factory=uow_factory),
        get_shipment=services.get_shipment,
        list_models=services.list_models,
        activate_model=services.activate_model,
        rollback_model=RollbackModel(uow_factory=uow_factory, activate=services.activate_model),
        reload_model=follower.check,
        start_training=services.start_training,
        release_training_job=ReleaseTrainingJob(uow_factory=uow_factory, clock=utc_now),
        training_job_page=ListTrainingJobPage(uow_factory=uow_factory),
        count_training_labels=CountTrainingLabels(uow_factory=uow_factory),
        worker_online=functools.partial(worker_connected, engine),
        storage=storage,
        model=model,
        draw_heatmap=draw_heatmap,
        measure_stack=measure_stack,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        follower.start()
        try:
            yield
        finally:
            follower.stop()
            engine.dispose()

    return create_app(services, lifespan, ui)


def load_package(
    models_dir: Path, device: str, bf16: bool, version: ModelVersionId
) -> PackagePredictor:
    """Loads the package of a version from its folder in the models storage.

    Args:
        models_dir: Folder with one package folder per version.
        device: Torch device the model runs on.
        bf16: Whether the encoder runs in bfloat16; it acts only on a card.
        version: Version to load; its tag format keeps the path inside `models_dir`.

    Raises:
        ModelPackageError: If the package is missing, invalid or not supported.
    """
    return PackagePredictor(
        ModelPackage.load(models_dir / version.value),
        device,
        bf16=bf16,
        # Beside the package, not inside it: every file inside is listed in the manifest with
        # its hash, and this one is ours, not the model's.
        scale_cache=models_dir / f"{version.value}{SCALE_SUFFIX}",
    )
