import asyncio

import pytest
from starlette.types import Message, Receive, Scope, Send

from loadvision.api.middleware import RequestContextMiddleware


@pytest.mark.api
def test_declared_length_over_the_limit_is_refused_without_reading_the_body() -> None:
    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        raise AssertionError("the application must not run")

    async def receive() -> Message:
        raise AssertionError("the body must not be read")

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestContextMiddleware(app, header="X-Request-ID", max_body_bytes=10)
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/assessments",
        "headers": [(b"content-length", b"11")],
    }

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
