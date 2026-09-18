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
from loadvision.domain.operator import OperatorSession
from loadvision.domain.training_job import TrainingMode
from loadvision.ml.training.gate import DEFAULT_RULES

router = APIRouter()

_MODELS: Final = f"{UI_PREFIX}/models"
_JOBS: Final = f"{UI_PREFIX}/jobs"
_NOTHING_TO_LEARN: Final = (
    "Дообучать не на чем.",
    "Подтвердите загрузку хотя бы на одном фото в очереди проверки.",
)


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
    """Shows the recent retrainings, how the gate judged them and what there is to learn from."""
    return await _jobs_page(request, session)


@router.post("/jobs")
async def start_job(request: Request, session: SignedIn) -> Response:
    """Queues a retraining on the confirmed labels; the worker picks it up.

    Without a single confirmed load the page refuses: the run would take the worker minutes
    and end with the gate rejecting the candidate. The XML API leaves that choice to its caller.
    """
    await checked_form(request, session)
    services = ui_services_of(request)
    if await run_in_threadpool(services.count_training_labels) == 0:
        return await _jobs_page(request, session, status_code=400, error=_NOTHING_TO_LEARN)
    await run_in_threadpool(services.start_training, TrainingMode.FAST)
    return RedirectResponse(_JOBS, status_code=303)


async def _jobs_page(
    request: Request,
    session: OperatorSession,
    *,
    status_code: int = 200,
    error: tuple[str, str] | None = None,
) -> Response:
    services = ui_services_of(request)
    recent = await run_in_threadpool(services.list_training_jobs)
    labels = await run_in_threadpool(services.count_training_labels)
    return page(
        request,
        "jobs.html",
        session,
        status_code=status_code,
        jobs=recent,
        labels=labels,
        # The gate of the worker: fewer labels than this and no candidate can pass.
        needed=DEFAULT_RULES.min_new_labels,
        error=error,
    )
