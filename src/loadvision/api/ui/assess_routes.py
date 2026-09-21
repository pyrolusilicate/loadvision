"""Assessing a photo from the browser: a shipment number and a file in, the result page out."""

import logging
from typing import Final
from urllib.parse import urlencode

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from loadvision.api.services import request_id_of
from loadvision.api.ui.pages import catalog_error, page
from loadvision.api.ui.services import ui_services_of
from loadvision.api.ui.session import SignedIn, checked_form
from loadvision.api.ui.templates import UI_PREFIX
from loadvision.application.assess_photo import AssessPhotoRequest, AssessPhotoResult
from loadvision.domain.assessment import LoadPercent
from loadvision.domain.operator import OperatorSession
from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.settings import MEBIBYTE

router = APIRouter()

# What the result page says about the assessment it opens with.
NEW_RESULT: Final = "new"
REUSED_RESULT: Final = "reused"
AGAIN_RESULT: Final = "again"
# Query parameter that names the model version whose estimate was replaced.
PREVIOUS_PARAM: Final = "from"
# Query parameter that names the shipment whose stored estimate of this photo was used.
TAKEN_FROM_PARAM: Final = "same_as"
_NO_PHOTO: Final = ("Фото не выбрано.", "Выберите файл с фото грузового отсека.")
_BAD_DECLARED: Final = (
    "Загрузка по документам указана неверно.",
    "Введите число от 0 до 100 или оставьте поле пустым.",
)

_logger = logging.getLogger(__name__)


@router.get("/assess")
def assess_form(request: Request, session: SignedIn, number: str = "") -> Response:
    """Shows the form; `number` fills in the shipment, such as when a shipment page links here."""
    return _form(request, session, number=number.strip(), loading=_still_loading(request))


@router.post("/assess")
async def assess(request: Request, session: SignedIn) -> Response:
    """Assesses the photo, stores it with the shipment and opens the result.

    A photo the service does not take sends the form back with the reason and hint from the
    error catalog, the same words the XML API answers with; the number stays filled in.
    """
    form = await checked_form(request, session)
    number = str(form.get("shipment_id", "")).strip()
    typed = str(form.get("declared_load_pct", "")).strip()
    try:
        declared = None if not typed else LoadPercent(float(typed))
    except ValueError:
        return _form(
            request, session, number=number, status_code=400, error=_BAD_DECLARED, declared=typed
        )
    photo = form.get("photo")
    # A browser sends an empty file part when nothing was chosen.
    if not isinstance(photo, UploadFile) or not photo.filename:
        return _form(
            request, session, number=number, status_code=400, error=_NO_PHOTO, declared=typed
        )
    raw = await photo.read()
    services = ui_services_of(request)
    try:
        predictor = services.model.current()
        if predictor is None:
            raise LoadvisionError(ErrorCode.MODEL_NOT_READY, services.model.failure)
        result = await run_in_threadpool(
            services.assess_photo,
            AssessPhotoRequest(
                shipment_id=number,
                image=raw,
                request_id=request_id_of(request),
                declared_load_pct=declared,
            ),
            predictor,
        )
    except LoadvisionError as error:
        _logger.info(
            "photo refused: %s %s",
            error.code.value,
            error.detail,
            extra={"request_id": request_id_of(request)},
        )
        status, message, hint = catalog_error(error.code)
        # A model that is still loading is not a rejected photo: the page says what is going on
        # instead of blaming the file the operator just picked.
        if error.code is ErrorCode.MODEL_NOT_READY:
            return _form(
                request, session, number=number, status_code=status, declared=typed, loading=True
            )
        return _form(
            request,
            session,
            number=number,
            status_code=status,
            error=(message, hint),
            declared=typed,
        )
    query = _result_query(result)
    return RedirectResponse(
        f"{UI_PREFIX}/assessments/{result.assessment.id}?{query}", status_code=303
    )


def _result_query(result: AssessPhotoResult) -> str:
    """Tells the result page what happened to the photo, with values encoded for an address."""
    if result.reused:
        return urlencode({"result": REUSED_RESULT})
    fields = {"result": NEW_RESULT if result.replaced is None else AGAIN_RESULT}
    if result.replaced is not None:
        fields[PREVIOUS_PARAM] = result.replaced.value
    # The same photo in another shipment costs no prediction; the page says so, or the two
    # cards look like the model was asked twice and answered the same to the decimal.
    if result.taken_from is not None:
        fields[TAKEN_FROM_PARAM] = result.taken_from.value
    return urlencode(fields)


def _still_loading(request: Request) -> bool:
    """Whether the service has no model yet and is not stuck: it is coming, not broken."""
    model = ui_services_of(request).model
    return model.current() is None and model.failure is None


def _form(
    request: Request,
    session: OperatorSession,
    *,
    number: str,
    status_code: int = 200,
    error: tuple[str, str] | None = None,
    declared: str = "",
    loading: bool = False,
) -> Response:
    limits = ui_services_of(request).assess_photo.limits
    return page(
        request,
        "assess.html",
        session,
        status_code=status_code,
        number=number,
        error=error,
        declared=declared,
        loading=loading,
        accept=",".join(sorted(media.value for media in limits.media_types)),
        formats=", ".join(sorted(media.name for media in limits.media_types)),
        max_mb=f"{limits.max_bytes / MEBIBYTE:g}",
        min_side=limits.min_side,
    )
