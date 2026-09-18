"""Model versions and retraining: what answers now, what can replace it, what is training."""

import functools
from typing import Annotated, Final

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from loadvision.api.path_params import version_from_path
from loadvision.api.ui.pages import page
from loadvision.api.ui.pagination import ROWS_PER_PAGE, pager
from loadvision.api.ui.services import ui_services_of
from loadvision.api.ui.session import SignedIn, checked_form
from loadvision.api.ui.templates import UI_PREFIX
from loadvision.application.paging import slice_page
from loadvision.domain.operator import OperatorSession
from loadvision.domain.training_job import TrainingMode
from loadvision.ml.training.gate import DEFAULT_RULES

router = APIRouter()

_MODELS: Final = f"{UI_PREFIX}/models"
_JOBS: Final = f"{UI_PREFIX}/jobs"
# The gate of the worker: with fewer confirmed loads no candidate can pass, so a run would
# only take the worker's time.
_NEEDED_LABELS: Final = DEFAULT_RULES.min_new_labels
_TOO_FEW_LABELS: Final = (
    "Подтверждённых загрузок пока мало.",
    f"Дообучение запускается от {_NEEDED_LABELS}: подтвердите загрузку в очереди проверки.",
)
_NO_WORKER: Final = ("Воркер не запущен.", "Запустите воркер и нажмите ещё раз.")


@router.get("/models")
async def models(
    request: Request, session: SignedIn, number: Annotated[int, Query(alias="page")] = 1
) -> Response:
    """Shows the registered versions with their status, newest first."""
    # Versions come a few a day, so the list is paged in memory.
    versions = slice_page(
        await run_in_threadpool(ui_services_of(request).list_models), number, ROWS_PER_PAGE
    )
    return page(request, "models.html", session, versions=versions, pager=pager(versions, _MODELS))


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
async def jobs(
    request: Request, session: SignedIn, number: Annotated[int, Query(alias="page")] = 1
) -> Response:
    """Shows the retrainings, how the gate judged them and what there is to learn from."""
    return await _jobs_page(request, session, number=number)


@router.post("/jobs")
async def start_job(request: Request, session: SignedIn) -> Response:
    """Queues a retraining on the confirmed labels; the worker picks it up.

    The page refuses with fewer confirmed loads than the gate needs, where the run would end
    with the gate rejecting the candidate, and without a running worker, where the job would
    wait in the queue with nobody to take it. The XML API leaves both to its caller.
    """
    await checked_form(request, session)
    services = ui_services_of(request)
    if await run_in_threadpool(services.count_training_labels) < _NEEDED_LABELS:
        return await _jobs_page(request, session, status_code=400, error=_TOO_FEW_LABELS)
    if not await run_in_threadpool(services.worker_online):
        return await _jobs_page(request, session, status_code=409, error=_NO_WORKER)
    await run_in_threadpool(services.start_training, TrainingMode.FAST)
    return RedirectResponse(_JOBS, status_code=303)


async def _jobs_page(
    request: Request,
    session: OperatorSession,
    *,
    number: int = 1,
    status_code: int = 200,
    error: tuple[str, str] | None = None,
) -> Response:
    services = ui_services_of(request)
    recent = await run_in_threadpool(
        functools.partial(services.training_job_page, page=number, size=ROWS_PER_PAGE)
    )
    labels = await run_in_threadpool(services.count_training_labels)
    worker_online = await run_in_threadpool(services.worker_online)
    return page(
        request,
        "jobs.html",
        session,
        status_code=status_code,
        jobs=recent,
        pager=pager(recent, _JOBS),
        labels=labels,
        needed=_NEEDED_LABELS,
        worker_online=worker_online,
        error=error,
    )
