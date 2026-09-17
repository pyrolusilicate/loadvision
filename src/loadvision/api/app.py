"""FastAPI application factory and composition root."""

from fastapi import FastAPI

from loadvision.api.errors import install_error_handlers
from loadvision.api.middleware import RequestContextMiddleware
from loadvision.api.routers import assessments, health, schema
from loadvision.api.services import ApiServices


def create_app(services: ApiServices) -> FastAPI:
    """Builds the application around ready services.

    Wiring with the real database, storage and model package lives in the composition root
    that comes with the Postgres repositories (task 6.2); tests pass fakes here.

    Args:
        services: Use cases, the active model and HTTP settings.

    Returns:
        The application.
    """
    app = FastAPI(title="Loadvision API", version="1")
    app.state.services = services
    install_error_handlers(app)
    app.include_router(assessments.router)
    app.include_router(schema.router)
    app.include_router(health.router)
    app.add_middleware(
        RequestContextMiddleware,
        header=services.request_id_header,
        max_body_bytes=services.max_body_bytes,
    )
    return app
