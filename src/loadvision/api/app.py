"""FastAPI application factory: routes, error handlers and middleware around ready services."""

from fastapi import FastAPI
from starlette.types import Lifespan

from loadvision.api.errors import install_error_handlers
from loadvision.api.middleware import RequestContextMiddleware
from loadvision.api.routers import (
    assessments,
    health,
    labels,
    models,
    schema,
    shipments,
    training_jobs,
)
from loadvision.api.services import ApiServices


def create_app(services: ApiServices, lifespan: Lifespan[FastAPI] | None = None) -> FastAPI:
    """Builds the application around ready services.

    The service process wires the real database, storage and model package in `api/main.py`;
    tests pass fakes here.

    Args:
        services: Use cases, the active model and HTTP settings.
        lifespan: What runs when the application starts and stops.

    Returns:
        The application.
    """
    app = FastAPI(title="Loadvision API", version="1", lifespan=lifespan)
    app.state.services = services
    install_error_handlers(app)
    app.include_router(assessments.router)
    app.include_router(labels.router)
    app.include_router(shipments.router)
    app.include_router(models.router)
    app.include_router(training_jobs.router)
    app.include_router(schema.router)
    app.include_router(health.router)
    app.add_middleware(
        RequestContextMiddleware,
        header=services.request_id_header,
        max_body_bytes=services.max_body_bytes,
    )
    return app
