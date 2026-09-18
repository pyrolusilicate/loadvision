"""The review queue: photos waiting for a confirmed load, one assessment, and its photo."""

import re
from typing import Annotated, Final

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from loadvision.api.path_params import uuid_from_path
from loadvision.api.ui.assess_routes import NEW_RESULT, REUSED_RESULT
from loadvision.api.ui.pages import page
from loadvision.api.ui.services import ui_services_of
from loadvision.api.ui.session import SignedIn, checked_form
from loadvision.api.ui.templates import UI_PREFIX
from loadvision.application.submit_label import SubmitLabelRequest
from loadvision.domain.assessment import SHA256_HEX_PATTERN, LoadPercent
from loadvision.domain.confirmed_label import LabelSource
from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.imaging.validator import detect_media_type

router = APIRouter()

_SHA256_RE: Final = re.compile(SHA256_HEX_PATTERN)
# A photo never changes under its hash, so a browser may keep it for a day.
_PHOTO_CACHE: Final = "private, max-age=86400, immutable"
_UNUSABLE_ACTION: Final = "unusable"


@router.get("/")
async def queue(
    request: Request, session: SignedIn, show_all: Annotated[bool, Query(alias="all")] = False
) -> Response:
    """Shows the newest assessments, by default only those whose photo nobody confirmed."""
    items = await run_in_threadpool(
        ui_services_of(request).review_queue, unlabeled_only=not show_all
    )
    return page(request, "queue.html", session, items=items, show_all=show_all)


@router.get("/assessments/{assessment_id}")
async def assessment(
    request: Request, session: SignedIn, assessment_id: str, result: str = ""
) -> Response:
    """Shows one assessment: the photo, the prediction and the form for the verdict.

    `result` comes from the photo form and says whether the photo was just assessed or had
    been assessed for this shipment before.
    """
    details = await run_in_threadpool(
        ui_services_of(request).get_assessment,
        uuid_from_path(assessment_id, ErrorCode.ASSESSMENT_NOT_FOUND),
    )
    return page(
        request,
        "assessment.html",
        session,
        details=details,
        created=result == NEW_RESULT,
        reused=result == REUSED_RESULT,
    )


@router.post("/assessments/{assessment_id}/label")
async def label(request: Request, session: SignedIn, assessment_id: str) -> Response:
    """Stores the verdict of the operator and returns to the queue.

    The operator either types the load or marks the photo unusable. A load outside 0-100 or
    not a number sends the page back with the reason, and nothing is stored.
    """
    target = uuid_from_path(assessment_id, ErrorCode.ASSESSMENT_NOT_FOUND)
    form = await checked_form(request, session)
    services = ui_services_of(request)
    if form.get("action") == _UNUSABLE_ACTION:
        load_pct = None
    else:
        load_pct = _load(str(form.get("load_pct", "")))
        if load_pct is None:
            details = await run_in_threadpool(services.get_assessment, target)
            return page(
                request,
                "assessment.html",
                session,
                status_code=400,
                details=details,
                typed=form.get("load_pct", ""),
                error="Загрузка — число от 0 до 100, например 72.5.",
            )
    await run_in_threadpool(
        services.submit_label,
        SubmitLabelRequest(
            assessment_id=target,
            load_pct=load_pct,
            source=LabelSource.UI,
            author=session.login,
        ),
    )
    return RedirectResponse(f"{UI_PREFIX}/", status_code=303)


@router.get("/photos/{sha256}")
async def photo(request: Request, _session: SignedIn, sha256: str) -> Response:
    """Returns the original photo bytes, addressed by their hash."""
    if _SHA256_RE.fullmatch(sha256) is None:
        raise LoadvisionError(ErrorCode.ROUTE_NOT_FOUND, "malformed photo hash")
    raw = await run_in_threadpool(ui_services_of(request).storage.load, sha256)
    media_type = None if raw is None else detect_media_type(raw)
    if raw is None or media_type is None:
        raise LoadvisionError(ErrorCode.ROUTE_NOT_FOUND, f"photo {sha256} is not stored")
    return Response(raw, media_type=media_type.value, headers={"Cache-Control": _PHOTO_CACHE})


def _load(text: str) -> LoadPercent | None:
    # A comma is what a Russian keyboard types for a decimal point.
    try:
        value = float(text.strip().replace(",", "."))
    except ValueError:
        return None
    # False for nan and infinity as well.
    if not 0 <= value <= 100:
        return None
    # Stored as typed, like every load: rounding belongs to the pages. Adding 0.0 turns "-0"
    # into 0.0, which is shown as "0.0".
    return LoadPercent(value + 0.0)
