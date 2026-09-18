"""Request id, body size limit and one log line per request."""

import logging
import re
import time
import uuid
from typing import Final

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from loadvision.api.errors import error_response
from loadvision.errors import ErrorCode
from loadvision.observability.events import Event

# Repeated in the XSD as RequestIdType; tests/contract/test_xsd.py checks both are the same.
REQUEST_ID_PATTERN: Final = "[A-Za-z0-9._-]{1,128}"

_REQUEST_ID_RE: Final = re.compile(REQUEST_ID_PATTERN)
# Status of a request whose exception is answered by the server error handler outside.
_UNHANDLED_STATUS: Final = 500

_logger = logging.getLogger(__name__)


class RequestContextMiddleware:
    """Gives every request an id, refuses a body over the limit and logs how the request went.

    A client id in the header is kept if it matches `REQUEST_ID_PATTERN`, otherwise a new one
    is issued; either way the response carries it in the same header. The body is read into
    memory up to the limit, so a route never sees a partial body. Every request ends with one
    `request.finished` line: id, method, path, status and time. The body itself never reaches
    the log, because a photo in base64 would fill it.

    Args:
        app: The wrapped application.
        header: Request id header, such as ``X-Request-ID``.
        max_body_bytes: Largest accepted body.
    """

    def __init__(self, app: ASGIApp, *, header: str, max_body_bytes: int) -> None:
        self._app = app
        self._header = header
        self._max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        request_id = self._request_id(headers.get(self._header))
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status: int | None = None

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[self._header] = request_id
                status = message["status"]
            await send(message)

        try:
            await self._handle(scope, receive, send_with_id, request_id, headers)
        except Exception:
            status = _UNHANDLED_STATUS
            raise
        finally:
            _logger.info(
                "%s %s %s",
                scope["method"],
                scope["path"],
                status,
                extra={
                    "event": Event.REQUEST_FINISHED.value,
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    # None when the client went away before an answer was sent.
                    "status": status,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                },
            )

    async def _handle(
        self,
        scope: Scope,
        receive: Receive,
        send_with_id: Send,
        request_id: str,
        headers: Headers,
    ) -> None:
        if self._declared_too_large(headers.get("content-length")):
            await error_response(ErrorCode.PAYLOAD_TOO_LARGE, request_id)(
                scope, receive, send_with_id
            )
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self._max_body_bytes:
                await error_response(ErrorCode.PAYLOAD_TOO_LARGE, request_id)(
                    scope, receive, send_with_id
                )
                return
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self._app(scope, replay, send_with_id)

    def _request_id(self, sent: str | None) -> str:
        if sent is not None and _REQUEST_ID_RE.fullmatch(sent):
            return sent
        return uuid.uuid4().hex

    def _declared_too_large(self, content_length: str | None) -> bool:
        return (
            content_length is not None
            and content_length.isdigit()
            and int(content_length) > self._max_body_bytes
        )
