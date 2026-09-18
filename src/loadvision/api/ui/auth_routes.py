"""Signing in and out of the operator pages."""

from typing import Annotated

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from loadvision.api.ui.pages import page
from loadvision.api.ui.services import ui_services_of
from loadvision.api.ui.session import SESSION_COOKIE, SignedIn, checked_form
from loadvision.api.ui.templates import HOME_PAGE, UI_PREFIX, human_time
from loadvision.application.operators import SignInLockedError, SignInRefusedError
from loadvision.domain.operator import SESSION_LIFETIME

router = APIRouter()


@router.get("/login")
def login_form(
    request: Request, target: Annotated[str, Query(alias="next")] = HOME_PAGE
) -> Response:
    """Shows the sign-in form; `next` is the page to return to."""
    return page(request, "login.html", None, next=_safe_next(target))


@router.post("/login")
async def log_in(request: Request) -> Response:
    """Signs the operator in and sends them where they were going.

    A wrong login and a wrong password get the same answer, so the page does not tell which
    logins exist.
    """
    form = await request.form()
    login = str(form.get("login", "")).strip()
    password = str(form.get("password", ""))
    target = _safe_next(str(form.get("next", HOME_PAGE)))
    try:
        signed = await run_in_threadpool(ui_services_of(request).sign_in, login, password)
    except SignInRefusedError:
        return page(
            request,
            "login.html",
            None,
            status_code=401,
            next=target,
            login=login,
            error="Неверный логин или пароль.",
        )
    except SignInLockedError as locked:
        return page(
            request,
            "login.html",
            None,
            status_code=429,
            next=target,
            login=login,
            error=f"Слишком много неверных паролей. Вход откроется в {human_time(locked.until)}.",
        )
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        signed.token,
        max_age=int(SESSION_LIFETIME.total_seconds()),
        path=UI_PREFIX,
        httponly=True,
        samesite="lax",
        # Over HTTPS the cookie never travels in the clear; a local demo runs on plain HTTP.
        secure=request.url.scheme == "https",
    )
    return response


@router.post("/logout")
async def log_out(request: Request, session: SignedIn) -> Response:
    """Ends the session and returns to the sign-in form."""
    await checked_form(request, session)
    token = request.cookies.get(SESSION_COOKIE, "")
    await run_in_threadpool(ui_services_of(request).sign_out, token)
    response = RedirectResponse(f"{UI_PREFIX}/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path=UI_PREFIX)
    return response


def _safe_next(target: str) -> str:
    # Only a page of our own: a link from a letter must not send a signed-in operator away.
    # A target under the prefix starts with one slash, so it never names another host.
    if target.startswith(f"{UI_PREFIX}/"):
        return target
    return HOME_PAGE
