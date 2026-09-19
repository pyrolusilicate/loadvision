"""The operator pages: a sub-application under `/ui` with its own HTML error pages."""

import logging
from pathlib import Path
from typing import Final
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.staticfiles import StaticFiles

from loadvision.api.services import request_id_of
from loadvision.api.ui import (
    assess_routes,
    auth_routes,
    model_routes,
    review_routes,
    shipment_routes,
)
from loadvision.api.ui.pages import catalog_error, page
from loadvision.api.ui.services import UiServices
from loadvision.api.ui.session import FormRejectedError, SignInRequiredError
from loadvision.api.ui.templates import UI_PREFIX
from loadvision.errors import ErrorCode, LoadvisionError

_STATIC: Final = Path(__file__).parent / "static"
_FORM_REJECTED: Final = (
    "Форма устарела или открыта с другого сайта.",
    "Обновите страницу и повторите действие.",
)
# The catalog words routing errors for API clients; a person on a page needs a way back.
_NO_PAGE: Final = ("Такой страницы нет.", "Вернитесь на главную страницу.")

_logger = logging.getLogger(__name__)


def create_ui_app(services: UiServices) -> FastAPI:
    """Builds the operator pages around ready services.

    The pages call the same use cases as the XML API; only the way they answer differs. Errors
    come back as pages with the message and hint of the catalog, not as XML.

    Args:
        services: Use cases, photo storage and the active model.

    Returns:
        The sub-application; the API mounts it at `UI_PREFIX`.
    """
    app = FastAPI(
        title="Loadvision operator pages", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.ui_services = services
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    for routes in (auth_routes, assess_routes, review_routes, shipment_routes, model_routes):
        app.include_router(routes.router)
    app.add_exception_handler(SignInRequiredError, _to_sign_in)
    app.add_exception_handler(FormRejectedError, _form_rejected)
    app.add_exception_handler(LoadvisionError, _catalog_error)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _malformed_address)
    app.add_exception_handler(Exception, _unexpected_error)
    return app


async def _to_sign_in(request: Request, _error: Exception) -> Response:
    back = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    return RedirectResponse(f"{UI_PREFIX}/login?next={quote(back, safe='/')}", status_code=303)


async def _form_rejected(request: Request, _error: Exception) -> Response:
    message, hint = _FORM_REJECTED
    return _error_page(request, 403, message, hint)


async def _catalog_error(request: Request, error: Exception) -> Response:
    assert isinstance(error, LoadvisionError)
    _logger.info(
        "page failed: %s %s",
        error.code.value,
        error.detail,
        extra={"request_id": request_id_of(request)},
    )
    status, message, hint = catalog_error(error.code)
    return _error_page(request, status, message, hint)


async def _http_error(request: Request, error: Exception) -> Response:
    # No such page, or the address of a form opened as a link.
    assert isinstance(error, HTTPException)
    message, hint = _NO_PAGE
    response = _error_page(request, error.status_code, message, hint)
    # 405 lists the allowed methods in the Allow header.
    response.headers.update(error.headers or {})
    return response


async def _malformed_address(request: Request, _error: Exception) -> Response:
    # A query value the page cannot read, such as ?all=maybe typed by hand: no such page, just
    # as a malformed id in the path.
    message, hint = _NO_PAGE
    return _error_page(request, 404, message, hint)


async def _unexpected_error(request: Request, _error: Exception) -> Response:
    # Not logged here: after the page is sent, Starlette raises the error again, and the handler
    # of the service application logs it with the request id.
    status, message, hint = catalog_error(ErrorCode.INTERNAL_ERROR)
    return _error_page(request, status, message, hint)


def _error_page(request: Request, status: int, message: str, hint: str) -> Response:
    return page(
        request,
        "error.html",
        None,
        status_code=status,
        message=message,
        hint=hint,
        request_id=request_id_of(request),
    )
