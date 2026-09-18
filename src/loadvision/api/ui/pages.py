"""Rendering a page with what every page shows: the operator, the form token and the model."""

from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from loadvision.api.ui.services import ui_services_of
from loadvision.api.ui.templates import templates
from loadvision.domain.operator import OperatorSession


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
