"""Model versions and retraining: what answers now, what can replace it, what is training."""

from typing import Final

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from loadvision.api.path_params import version_from_path
from loadvision.api.ui.pages import page
from loadvision.api.ui.services import ui_services_of
from loadvision.api.ui.session import SignedIn, checked_form
from loadvision.api.ui.templates import UI_PREFIX
from loadvision.domain.training_job import TrainingMode

router = APIRouter()

_MODELS: Final = f"{UI_PREFIX}/models"
_JOBS: Final = f"{UI_PREFIX}/jobs"


@router.get("/models")
async def models(request: Request, session: SignedIn) -> Response:
    """Shows every registered version with its status."""
    versions = await run_in_threadpool(ui_services_of(request).list_models)
    return page(request, "models.html", session, versions=versions)


@router.post("/models/{version}/activate")
async def activate(request: Request, session: SignedIn, version: str) -> Response:
    """Makes a version active and starts loading it; until then the previous one answers."""
    await checked_form(request, session)
    services = ui_services_of(request)
    await run_in_threadpool(services.activate_model, version_from_path(version))
    await run_in_threadpool(services.reload_model)
    return RedirectResponse(_MODELS, status_code=303)


@router.post("/models/rollback")
async def rollback(request: Request, session: SignedIn) -> Response:
    """Brings back the version that answered before the current one."""
    await checked_form(request, session)
    services = ui_services_of(request)
    await run_in_threadpool(services.rollback_model)
    await run_in_threadpool(services.reload_model)
    return RedirectResponse(_MODELS, status_code=303)


@router.get("/jobs")
async def jobs(request: Request, session: SignedIn) -> Response:
    """Shows the recent retrainings and how the gate judged them."""
    recent = await run_in_threadpool(ui_services_of(request).list_training_jobs)
    return page(request, "jobs.html", session, jobs=recent)


@router.post("/jobs")
async def start_job(request: Request, session: SignedIn) -> Response:
    """Queues a retraining on the confirmed labels; the worker picks it up."""
    await checked_form(request, session)
    await run_in_threadpool(ui_services_of(request).start_training, TrainingMode.FAST)
    return RedirectResponse(_JOBS, status_code=303)
