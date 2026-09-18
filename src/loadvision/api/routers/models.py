"""Routes for model versions."""

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response

from loadvision.api.path_params import version_from_path
from loadvision.api.services import services_of
from loadvision.api.xml.response import xml_response
from loadvision.api.xml.training import model_document, models_document

router = APIRouter()


@router.get("/api/v1/models")
async def list_models(request: Request) -> Response:
    """Answers 200 with every registered version, newest registration first."""
    versions = await run_in_threadpool(services_of(request).list_models)
    return xml_response(models_document(versions))


@router.post("/api/v1/models/{version}/activate")
async def activate_model(request: Request, version: str) -> Response:
    """Makes a registered version the one the service answers with.

    Answers 200 with the active version. Errors: 404 `MODEL_VERSION_NOT_FOUND` for a version
    that is not registered, 409 `MODEL_VERSION_STATUS_CONFLICT` for one the gate rejected.
    """
    active = await run_in_threadpool(
        services_of(request).activate_model, version_from_path(version)
    )
    return xml_response(model_document(active))
