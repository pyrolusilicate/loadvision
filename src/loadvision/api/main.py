"""Composition root of the API process: settings, Postgres, photo storage and model packages.

Run with ``uvicorn --factory loadvision.api.main:create_service_app``.
"""

import functools
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import Engine

from loadvision.api.app import create_app
from loadvision.api.services import ApiServices
from loadvision.application.assess_photo import AssessPhoto
from loadvision.application.get_assessment import GetAssessment
from loadvision.application.get_shipment import GetShipment
from loadvision.application.load_served_model import LoadServedModel
from loadvision.application.ports import UnitOfWorkFactory
from loadvision.application.submit_label import SubmitLabel
from loadvision.domain.model_version import ModelVersionId
from loadvision.infrastructure.db.session import PostgresUnitOfWork, create_database_engine
from loadvision.infrastructure.storage.local_storage import LocalImageStorage
from loadvision.ml.inference.active_model import ActiveModel
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.model_package import ModelPackage
from loadvision.settings import Settings, database_url, load_settings


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
    return build_service_app(load_settings(), create_database_engine(database_url()))


def build_service_app(settings: Settings, engine: Engine) -> FastAPI:
    """Wires the use cases to Postgres, the photo folder and the model packages.

    The service answers as soon as it starts. The model loads in the background, and until it
    is ready assessments and `/health` answer 503. The engine is closed when the service stops.

    Args:
        settings: Service settings.
        engine: Engine of the service database; the application disposes it on shutdown.

    Returns:
        The application.

    Raises:
        ValueError: If `model.version` is not a valid version tag.
    """
    uow_factory: UnitOfWorkFactory = functools.partial(PostgresUnitOfWork, engine)
    model = ActiveModel()
    services = ApiServices(
        assess_photo=AssessPhoto(
            uow_factory=uow_factory,
            storage=LocalImageStorage(settings.storage.images_dir),
            limits=settings.images.limits(),
            clock=utc_now,
            new_id=uuid.uuid4,
        ),
        get_assessment=GetAssessment(uow_factory=uow_factory),
        submit_label=SubmitLabel(uow_factory=uow_factory, clock=utc_now),
        get_shipment=GetShipment(uow_factory=uow_factory),
        model=model,
        request_id_header=settings.api.request_id_header,
        max_body_bytes=settings.api.max_body_bytes,
    )
    load_model = LoadServedModel(
        uow_factory=uow_factory,
        clock=utc_now,
        load_package=functools.partial(load_package, settings.storage.models_dir),
        initial_version=ModelVersionId(settings.model.version),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model.load_in_background(load_model)
        try:
            yield
        finally:
            engine.dispose()

    return create_app(services, lifespan)


def load_package(models_dir: Path, version: ModelVersionId) -> PackagePredictor:
    """Loads the package of a version from its folder in the models storage.

    Args:
        models_dir: Folder with one package folder per version.
        version: Version to load; its tag format keeps the path inside `models_dir`.

    Raises:
        ModelPackageError: If the package is missing, invalid or not supported.
    """
    return PackagePredictor(ModelPackage.load(models_dir / version.value))


def utc_now() -> datetime:
    """Returns the current time in UTC."""
    return datetime.now(UTC)
