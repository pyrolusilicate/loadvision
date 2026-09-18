import hashlib
from dataclasses import replace

import pytest

from loadvision.domain.operator import MAX_FAILED_SIGN_INS, SESSION_LIFETIME
from tests.api.service import Service
from tests.api.ui_pages import (
    LOGIN,
    add_operator,
    form_token,
    get,
    log_in,
    page,
    sign_in,
    submit,
    text_of,
)
from tests.builders import T0


@pytest.mark.api
def test_the_bare_address_opens_the_pages(service: Service) -> None:
    response = get(service, "/")

    assert (response.status_code, response.headers["location"]) == (303, "/ui/")


@pytest.mark.api
def test_a_page_without_a_session_sends_to_sign_in_and_back(service: Service) -> None:
    response = get(service, "/ui/?all=true")

    assert response.status_code == 303
    assert response.headers["location"] == "/ui/login?next=/ui/%3Fall%3Dtrue"
    form = page(get(service, response.headers["location"]))
    assert form.xpath("//input[@name='next']/@value") == ["/ui/?all=true"]


@pytest.mark.api
def test_signing_in_returns_to_the_page_with_a_cookie_only_the_pages_get(
    service: Service,
) -> None:
    add_operator(service)

    response = log_in(service, next="/ui/?all=true")

    assert (response.status_code, response.headers["location"]) == (303, "/ui/?all=true")
    cookie = response.headers["set-cookie"]
    assert cookie.startswith("loadvision_session=")
    for attribute in ("HttpOnly", "Path=/ui", "SameSite=lax", "Max-Age=43200"):
        assert attribute in cookie
    # A plain HTTP address, like a local demo, gets a cookie without Secure.
    assert "Secure" not in cookie
    assert get(service, "/ui/").status_code == 200


@pytest.mark.api
def test_spaces_around_the_login_are_ignored(service: Service) -> None:
    add_operator(service)

    assert log_in(service, f" {LOGIN} ").status_code == 303


@pytest.mark.api
def test_the_stylesheet_address_changes_with_its_content(service: Service) -> None:
    [href] = page(get(service, "/ui/login")).xpath("//link[@rel='stylesheet']/@href")

    response = get(service, href)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    path, version = href.split("?v=")
    assert path == "/ui/static/panel.css"
    assert hashlib.sha256(response.content).hexdigest().startswith(version)


@pytest.mark.api
@pytest.mark.parametrize(
    "target", ["https://example.com/", "//example.com/ui/", "/api/v1/models", "/uix"]
)
def test_signing_in_never_leaves_the_pages(service: Service, target: str) -> None:
    add_operator(service)

    response = log_in(service, next=target)

    assert response.headers["location"] == "/ui/"


@pytest.mark.api
@pytest.mark.parametrize(("login", "password"), [(LOGIN, "wrong password"), ("nobody", "x")])
def test_a_wrong_login_or_password_gets_one_answer(
    service: Service, login: str, password: str
) -> None:
    add_operator(service)

    response = log_in(service, login, password)

    assert response.status_code == 401
    assert "Неверный логин или пароль." in text_of(response)
    assert page(response).xpath("//input[@name='login']/@value") == [login]
    assert "set-cookie" not in response.headers


@pytest.mark.api
def test_a_locked_login_says_when_sign_in_opens(service: Service) -> None:
    add_operator(service)
    locked = replace(
        service.store.operators[LOGIN], failed_sign_ins=MAX_FAILED_SIGN_INS, first_failure_at=T0
    )
    service.store.operators[LOGIN] = locked

    response = log_in(service)

    assert response.status_code == 429
    assert "Вход откроется в 2026-09-16 12:15:00 UTC." in text_of(response)


@pytest.mark.api
def test_signing_out_ends_the_session(service: Service) -> None:
    token = sign_in(service)

    response = submit(service, "/ui/logout", csrf=token)

    assert (response.status_code, response.headers["location"]) == (303, "/ui/login")
    assert 'loadvision_session=""' in response.headers["set-cookie"]
    assert list(service.store.operator_sessions.values()) == [None]
    assert get(service, "/ui/").status_code == 303


@pytest.mark.api
@pytest.mark.parametrize("form", [{}, {"csrf": "forged"}])
def test_a_form_without_the_session_token_is_refused(
    service: Service, form: dict[str, str]
) -> None:
    sign_in(service)

    response = submit(service, "/ui/logout", **form)

    assert response.status_code == 403
    assert "Форма устарела или открыта с другого сайта." in text_of(response)
    assert get(service, "/ui/").status_code == 200


@pytest.mark.api
def test_a_token_sent_as_a_file_is_refused(service: Service) -> None:
    token = sign_in(service)

    response = service.client.post(
        "/ui/logout", files={"csrf": ("csrf.txt", token.encode())}, follow_redirects=False
    )

    assert response.status_code == 403


@pytest.mark.api
def test_a_session_ends_after_its_lifetime(service: Service) -> None:
    sign_in(service)

    service.clock.advance(SESSION_LIFETIME)

    assert get(service, "/ui/").status_code == 303


@pytest.mark.api
def test_every_form_of_a_page_carries_the_session_token(service: Service) -> None:
    token = sign_in(service)

    forms = page(get(service, "/ui/models")).xpath("//form[@method='post']")

    assert forms
    for form in forms:
        assert form.xpath(".//input[@name='csrf']/@value") == [token]
    assert form_token(get(service, "/ui/jobs")) == token
