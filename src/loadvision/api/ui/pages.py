"""Rendering pages: what every page shows, and the words of a catalog error."""

from types import MappingProxyType
from typing import Any, Final

from starlette.requests import Request
from starlette.responses import Response

from loadvision.api.errors import answered_code
from loadvision.api.ui.services import ui_services_of
from loadvision.api.ui.templates import templates
from loadvision.domain.operator import OperatorSession
from loadvision.errors import ERROR_CATALOG, ErrorCode

# The catalog hints speak to an API client: methods, the schema, requestId. A person on a page
# gets these instead; the messages stay the catalog's own.
_PAGE_HINTS: Final = MappingProxyType(
    {
        # The words of SHIPMENT_ID_PATTERN in domain/shipment.py.
        ErrorCode.INVALID_SHIPMENT_ID: (
            "Номер состоит из латинских букв, цифр, дефиса и подчёркивания, от 1 до 64 знаков, "
            "например RP-2026-000123."
        ),
        ErrorCode.PAYLOAD_TOO_LARGE: "Уменьшите разрешение или пересожмите фото в JPEG.",
        ErrorCode.CORRUPTED_IMAGE: "Выберите исходный файл фото, без обрезки и пересохранения.",
        ErrorCode.SHIPMENT_NOT_FOUND: (
            "Проверьте номер: перевозка появляется после первой оценки её фото."
        ),
        ErrorCode.ASSESSMENT_NOT_FOUND: "Откройте оценку из очереди или из истории перевозки.",
        ErrorCode.MODEL_VERSION_NOT_FOUND: "Выберите версию из списка на странице «Модели».",
        ErrorCode.MODEL_VERSION_STATUS_CONFLICT: (
            "Версию, которую отвергла проверка, включить нельзя."
        ),
        ErrorCode.MODEL_NOT_READY: "Подождите несколько секунд и повторите.",
        ErrorCode.MODEL_INFERENCE_FAILED: "Повторите: повтор безопасен.",
        ErrorCode.INTERNAL_ERROR: "Повторите позже. Если не пройдёт, сообщите номер запроса.",
    }
)


def catalog_error(code: ErrorCode) -> tuple[int, str, str]:
    """Returns the HTTP status, message and hint a page shows for a catalog error.

    The code is the one `answered_code` gives, the same as in the XML answers.
    """
    code = answered_code(code)
    spec = ERROR_CATALOG[code]
    return spec.http_status or 500, spec.message, _PAGE_HINTS.get(code, spec.hint)


def page(
    request: Request,
    name: str,
    session: OperatorSession | None,
    *,
    status_code: int = 200,
    **context: Any,
) -> Response:
    """Renders a template with the header data.

    Args:
        request: The request.
        name: Template file, such as ``queue.html``.
        session: Signed-in operator, or None on the sign-in and error pages.
        status_code: HTTP status of the page.
        **context: Values the template shows.
    """
    model = ui_services_of(request).model
    predictor = model.current()
    return templates.TemplateResponse(
        request,
        name,
        {
            "session": session,
            "model_version": None if predictor is None else predictor.version.value,
            "model_failure": model.failure,
            **context,
        },
        status_code=status_code,
    )
