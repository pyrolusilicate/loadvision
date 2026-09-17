"""Mapping of error codes to HTTP statuses and XML error responses."""

import logging
from typing import Final

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from loadvision.api.services import request_id_of, services_of
from loadvision.api.xml.response import XmlResponse, xml_response
from loadvision.api.xml.schemas import error_document
from loadvision.errors import ERROR_CATALOG, ErrorCode, LoadvisionError

_logger = logging.getLogger(__name__)

# Routing failures of the framework, answered in XML like every other error.
_HTTP_CODES: Final = {404: ErrorCode.ROUTE_NOT_FOUND, 405: ErrorCode.METHOD_NOT_ALLOWED}


def error_response(code: ErrorCode, request_id: str) -> XmlResponse:
    """Builds the XML error response of a catalog code.

    A code without an HTTP status, such as a pipeline failure, never reaches a client as is: it
    is answered as `INTERNAL_ERROR`.

    Args:
        code: Catalog code.
        request_id: Id of the request, echoed in the body.

    Returns:
        The response with the status of the code.
    """
    status = ERROR_CATALOG[code].http_status
    if status is None:
        code = ErrorCode.INTERNAL_ERROR
        status = ERROR_CATALOG[code].http_status or 500
    return xml_response(error_document(code, request_id), status)


def install_error_handlers(app: FastAPI) -> None:
    """Makes every error of the application an XML `Error` response.

    Args:
        app: Application to configure.
    """
    app.add_exception_handler(LoadvisionError, _on_loadvision_error)
    app.add_exception_handler(HTTPException, _on_http_error)
    app.add_exception_handler(Exception, _on_unexpected_error)


async def _on_loadvision_error(request: Request, error: Exception) -> Response:
    assert isinstance(error, LoadvisionError)
    request_id = request_id_of(request)
    _logger.info(
        "request failed: %s %s", error.code.value, error.detail, extra={"request_id": request_id}
    )
    return error_response(error.code, request_id)


async def _on_http_error(request: Request, error: Exception) -> Response:
    assert isinstance(error, HTTPException)
    code = _HTTP_CODES.get(error.status_code, ErrorCode.INTERNAL_ERROR)
    response = error_response(code, request_id_of(request))
    # 405 lists the allowed methods in the Allow header.
    response.headers.update(error.headers or {})
    return response


async def _on_unexpected_error(request: Request, error: Exception) -> Response:
    # This handler runs outside the request middleware, so the request id header is set here.
    request_id = request_id_of(request)
    _logger.exception("unexpected error", extra={"request_id": request_id})
    response = error_response(ErrorCode.INTERNAL_ERROR, request_id)
    response.headers[services_of(request).request_id_header] = request_id
    return response
