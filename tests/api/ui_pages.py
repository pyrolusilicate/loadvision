"""Signing in to the operator pages and reading what the pages show."""

from typing import Any

from httpx2 import Response
from lxml import html

from loadvision.application.operators import AddOperator
from tests.api.service import Service

LOGIN = "operator-17"
PASSWORD = "correct horse battery"


def add_operator(service: Service, login: str = LOGIN, password: str = PASSWORD) -> None:
    AddOperator(uow_factory=service.store.unit_of_work, clock=service.clock)(login, password)


def log_in(service: Service, login: str = LOGIN, password: str = PASSWORD, **form: str) -> Response:
    return service.client.post(
        "/ui/login", data={"login": login, "password": password, **form}, follow_redirects=False
    )


def sign_in(service: Service) -> str:
    """Adds the operator, signs in and returns the form token of the session."""
    add_operator(service)
    assert log_in(service).status_code == 303
    return form_token(service.client.get("/ui/"))


def get(service: Service, path: str) -> Response:
    return service.client.get(path, follow_redirects=False)


def submit(service: Service, path: str, **form: Any) -> Response:
    return service.client.post(path, data=form, follow_redirects=False)


def page(response: Response) -> html.HtmlElement:
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    return html.fromstring(response.text)


def text_of(response: Response) -> str:
    """The visible text of a page with runs of spaces collapsed."""
    return " ".join(page(response).text_content().split())


def form_token(response: Response) -> str:
    """The token every form of the page carries; all forms of one session carry the same."""
    [token] = set(page(response).xpath("//input[@name='csrf']/@value"))
    return str(token)
