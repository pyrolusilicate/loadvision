"""The review queue: photos waiting for a confirmed load, one assessment, and its photo."""

import functools
import logging
import re
from typing import Annotated, Final
from urllib.parse import urlencode
from uuid import UUID

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from loadvision.api.path_params import uuid_from_path
from loadvision.api.ui.assess_routes import (
    AGAIN_RESULT,
    NEW_RESULT,
    PREVIOUS_PARAM,
    REUSED_RESULT,
    TAKEN_FROM_PARAM,
)
from loadvision.api.ui.cargo_form import chosen_from, stored_block
from loadvision.api.ui.pages import page
from loadvision.api.ui.pagination import TILES_PER_PAGE, pager
from loadvision.api.ui.services import UiServices, ui_services_of
from loadvision.api.ui.session import SignedIn, checked_form
from loadvision.api.ui.templates import UI_PREFIX
from loadvision.application.ports import AssessmentOrder, ExplanationKind, ReviewStatus
from loadvision.application.submit_label import SubmitLabelRequest
from loadvision.application.weigh_cargo import WeighCargoRequest
from loadvision.domain.assessment import SHA256_HEX_PATTERN, LoadPercent
from loadvision.domain.cargo_weight import CargoKind
from loadvision.domain.confirmed_label import LabelSource
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.shipment import ShipmentId
from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.imaging.validator import detect_media_type

_logger = logging.getLogger(__name__)

router = APIRouter()

_SHA256_RE: Final = re.compile(SHA256_HEX_PATTERN)
# A photo never changes under its hash, so a browser may keep it for a day.
_PHOTO_CACHE: Final = "private, max-age=86400, immutable"
# A map is redrawn when the model changes, so a browser keeps it for a minute, not a day.
_MAP_CACHE: Final = "private, max-age=60"
_MAP_TYPE: Final = "image/jpeg"
_UNUSABLE_ACTION: Final = "unusable"
# What the address says when the operator has opened the weight block under the photo.
_WEIGH_OPEN: Final = "1"
# What the address says when the kind of cargo of the weighing was read off the photo.
_RECOGNIZED: Final = "1"
_QUEUE: Final = f"{UI_PREFIX}/"
# Filters and orders of the queue in the words of the page.
_STATUSES: Final = (
    (ReviewStatus.WAITING, "На рассмотрении"),
    (ReviewStatus.CONFIRMED, "Подтверждённые"),
    (ReviewStatus.UNUSABLE, "Непригодные"),
    (ReviewStatus.ALL, "Все"),
)
_ORDERS: Final = (
    (AssessmentOrder.NEWEST, "Сначала новые"),
    (AssessmentOrder.OLDEST, "Сначала старые"),
    (AssessmentOrder.FULLEST, "Сначала полные"),
    (AssessmentOrder.EMPTIEST, "Сначала пустые"),
)


@router.get("/")
async def queue(
    request: Request,
    session: SignedIn,
    status: ReviewStatus = ReviewStatus.WAITING,
    order: AssessmentOrder = AssessmentOrder.NEWEST,
    number: Annotated[int, Query(alias="page")] = 1,
) -> Response:
    """Shows one page of assessments, by default those whose photo nobody labelled yet."""
    shown = await run_in_threadpool(
        functools.partial(
            ui_services_of(request).review_queue,
            page=number,
            size=TILES_PER_PAGE,
            order=order,
            status=status,
        )
    )
    return page(
        request,
        "queue.html",
        session,
        items=shown.items,
        status=status.value,
        status_label=dict(_STATUSES)[status],
        order_label=dict(_ORDERS)[order],
        statuses=[
            (label, _queue_address(status=choice, order=order), choice is status)
            for choice, label in _STATUSES
        ],
        orders=[
            (label, _queue_address(status=status, order=choice), choice is order)
            for choice, label in _ORDERS
        ],
        pager=pager(shown, _QUEUE, **_queue_params(status=status, order=order)),
    )


@router.get("/assessments/{assessment_id}")
async def assessment(
    request: Request,
    session: SignedIn,
    assessment_id: str,
    result: str = "",
    previous: Annotated[str, Query(alias=PREVIOUS_PARAM)] = "",
    same_as: Annotated[str, Query(alias=TAKEN_FROM_PARAM)] = "",
    view: str = "",
    weigh: str = "",
    recognized: str = "",
) -> Response:
    """Shows one assessment: the photo, the prediction and the form for the verdict.

    The weight block under the photo shows what was weighed and stored for this assessment, if
    anything; weighing itself is a POST of its own.

    `result` comes from the photo form and says whether the photo was just assessed, had been
    assessed for this shipment by the model that answers now, or was assessed again in place
    of the estimate of the model named by `previous`. `same_as` names the shipment whose
    stored estimate of this same photo was used instead of running the model again.
    """
    services = ui_services_of(request)
    details = await run_in_threadpool(
        services.get_assessment,
        uuid_from_path(assessment_id, ErrorCode.ASSESSMENT_NOT_FOUND),
    )
    # The model found nothing of a truck body in this frame and marked the photo unusable
    # itself, so the page shows the measurements instead of a prediction nobody should act on.
    label = details.label
    unreadable = label is not None and label.source is LabelSource.MODEL
    return page(
        request,
        "assessment.html",
        session,
        details=details,
        unreadable=unreadable,
        heatmaps=services.draw_heatmap is not None,
        view=_view_or_photo(view),
        created=result == NEW_RESULT,
        reused=result == REUSED_RESULT,
        again=result == AGAIN_RESULT,
        previous=_version_or_blank(previous),
        same_as=_shipment_or_blank(same_as),
        weight=stored_block(details.weighing),
        weigh_open=weigh == _WEIGH_OPEN,
        recognize=services.recognize_cargo is not None,
        recognized=recognized == _RECOGNIZED,
    )


def _shipment_or_blank(value: str) -> str:
    """Returns the shipment number, or nothing when the address carries a malformed one."""
    try:
        return ShipmentId(value).value
    except LoadvisionError:
        return ""


def _view_or_photo(value: str) -> str:
    """The map the address asks for, or nothing at all, which means the photo itself."""
    return value if value in {kind.value for kind in ExplanationKind} else ""


def _version_or_blank(value: str) -> str:
    """The model version the address names, or nothing if it is not a version at all."""
    try:
        return ModelVersionId(value).value
    except ValueError:
        return ""


@router.post("/assessments/{assessment_id}/weight")
async def weigh(request: Request, session: SignedIn, assessment_id: str) -> Response:
    """Weighs the cargo of this assessment and keeps the numbers beside it.

    A weighing is stored rather than computed on every page, because the table of densities is
    ours to change and a number an operator acted on must not change under him afterwards.
    """
    target = uuid_from_path(assessment_id, ErrorCode.ASSESSMENT_NOT_FOUND)
    form = await checked_form(request, session)
    asked = chosen_from(
        str(form.get("body", "")),
        str(form.get("cargo", "")),
        str(form.get("length", "")),
        str(form.get("width", "")),
        str(form.get("height", "")),
    )
    services = ui_services_of(request)
    kind = await _kind_of(services, target) if asked.auto else asked.kind
    if kind is not None:
        await run_in_threadpool(
            services.weigh_cargo,
            WeighCargoRequest(
                assessment_id=target,
                body=asked.body,
                size=asked.size,
                kind=kind,
                stack_share=await _stack_of(services, target),
                author=session.login,
            ),
        )
    # Back to the card with the block still open, so the operator sees what he asked for.
    read = f"&recognized={_RECOGNIZED}" if asked.auto else ""
    return RedirectResponse(
        f"{UI_PREFIX}/assessments/{target}?weigh={_WEIGH_OPEN}{read}", status_code=303
    )


async def _kind_of(services: UiServices, assessment_id: UUID) -> CargoKind:
    """The kind of cargo read off the photo of this assessment.

    Mixed is the answer whenever nothing better can be read: its density is the widest, so it is
    the least wrong. A photo the model calls empty is weighed as mixed too, because how empty
    the body is, the load already says, and the weight of "nothing" would be a weight nobody
    asked for.
    """
    if services.recognize_cargo is None:
        return CargoKind.MIXED
    try:
        details = await run_in_threadpool(services.get_assessment, assessment_id)
        kind = await run_in_threadpool(services.recognize_cargo, details.assessment.image_sha256)
    except Exception:
        _logger.exception("the kind of cargo of %s could not be read", assessment_id)
        return CargoKind.MIXED
    return CargoKind.MIXED if kind is None or kind is CargoKind.EMPTY else kind


async def _stack_of(services: UiServices, assessment_id: UUID) -> float | None:
    """How high the cargo is stacked on the photo of this assessment.

    Returns None when the service cannot measure it at all, when the assessment is gone, or
    when the depth model fails: a weight is an estimate either way, and a page that refuses to
    weigh because one of its assumptions could not be sharpened helps nobody.
    """
    if services.measure_stack is None:
        return None
    try:
        details = await run_in_threadpool(services.get_assessment, assessment_id)
        return await run_in_threadpool(
            services.measure_stack,
            details.assessment.image_sha256,
            details.assessment.model_version,
        )
    except Exception:
        _logger.exception("the stack of %s could not be measured", assessment_id)
        return None


@router.post("/assessments/{assessment_id}/readable")
async def readable(request: Request, session: SignedIn, assessment_id: str) -> Response:
    """Cancels the verdict of the model, so the photo can be judged like any other."""
    target = uuid_from_path(assessment_id, ErrorCode.ASSESSMENT_NOT_FOUND)
    await checked_form(request, session)
    await run_in_threadpool(ui_services_of(request).mark_readable, target)
    return RedirectResponse(f"{UI_PREFIX}/assessments/{target}", status_code=303)


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
            label_of = details.label
            return page(
                request,
                "assessment.html",
                session,
                status_code=400,
                details=details,
                typed=form.get("load_pct", ""),
                error="Загрузка — число от 0 до 100, например 72.5.",
                # The card is the same card whatever brought it up, so it gets the same
                # context. A missing one is not an error for Jinja, it is a quietly different
                # page: without `weight` a refused number turned a 400 into a 500, and without
                # `heatmaps` the switch of the maps disappears.
                weight=stored_block(details.weighing),
                unreadable=label_of is not None and label_of.source is LabelSource.MODEL,
                heatmaps=services.draw_heatmap is not None,
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


@router.get("/photos/{sha256}/heatmap")
async def heatmap(
    request: Request, _session: SignedIn, sha256: str, kind: ExplanationKind
) -> Response:
    """Returns the map of what the model saw on the photo, as PNG.

    The map is drawn here and not when the photo is assessed: an assessment costs exactly as
    much as it did before, and a map only when somebody opens one.
    """
    services = ui_services_of(request)
    if services.draw_heatmap is None:
        raise LoadvisionError(ErrorCode.ROUTE_NOT_FOUND, "this service draws no maps")
    if _SHA256_RE.fullmatch(sha256) is None:
        raise LoadvisionError(ErrorCode.ROUTE_NOT_FOUND, "malformed photo hash")
    predictor = services.model.current()
    if predictor is None:
        raise LoadvisionError(ErrorCode.MODEL_NOT_READY, services.model.failure)
    picture = await run_in_threadpool(services.draw_heatmap, sha256, kind, predictor.version)
    # Not the long cache of a photo: a map belongs to a model version, and the address says
    # nothing about the version.
    return Response(picture, media_type=_MAP_TYPE, headers={"Cache-Control": _MAP_CACHE})


def _queue_params(*, status: ReviewStatus, order: AssessmentOrder) -> dict[str, str]:
    # Defaults stay out of the address, so the plain queue keeps its plain address.
    params: dict[str, str] = {}
    if status is not ReviewStatus.WAITING:
        params["status"] = status.value
    if order is not AssessmentOrder.NEWEST:
        params["order"] = order.value
    return params


def _queue_address(*, status: ReviewStatus, order: AssessmentOrder) -> str:
    params = _queue_params(status=status, order=order)
    return f"{_QUEUE}?{urlencode(params)}" if params else _QUEUE


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
