"""Who is signed in, and whether a form came from one of our own pages."""

from typing import Annotated, Final

from fastapi import Depends
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData
from starlette.requests import Request

from loadvision.api.ui.services import ui_services_of
from loadvision.application.operators import csrf_matches
from loadvision.domain.operator import OperatorSession

SESSION_COOKIE: Final = "loadvision_session"
CSRF_FIELD: Final = "csrf"


class SignInRequiredError(Exception):
    """The browser has no live session; the page sends it to sign in and back."""


class FormRejectedError(Exception):
    """A form came without the token of its session, so it may have come from another site."""


async def operator_session(request: Request) -> OperatorSession:
    """Returns the session of the browser; the dependency of every page but the sign-in one.

    Raises:
        SignInRequiredError: If the browser has no live session.
    """
    token = request.cookies.get(SESSION_COOKIE)
    session = await run_in_threadpool(ui_services_of(request).authenticate, token)
    if session is None:
        raise SignInRequiredError
    return session


async def checked_form(request: Request, session: OperatorSession) -> FormData:
    """Reads a posted form and checks that it carries the token of the session.

    Raises:
        FormRejectedError: If the token is missing or belongs to another session.
    """
    form = await request.form()
    token = form.get(CSRF_FIELD)
    if not isinstance(token, str) or not csrf_matches(session, token):
        raise FormRejectedError
    return form


# The signed-in operator, as a route parameter: a page without a live session sends the
# browser to sign in.
type SignedIn = Annotated[OperatorSession, Depends(operator_session)]
