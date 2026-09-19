from itertools import count

import pytest

from loadvision.application.operators import (
    AddOperator,
    Authenticate,
    DisableOperator,
    ListOperators,
    SetOperatorPassword,
    SignIn,
    SignInLockedError,
    SignInRefusedError,
    SignOut,
    csrf_matches,
    token_hash,
)
from loadvision.domain.operator import FAILURE_WINDOW, MAX_FAILED_SIGN_INS, SESSION_LIFETIME
from tests.builders import T0
from tests.fakes.clock import FixedClock
from tests.fakes.in_memory_db import InMemoryStore

LOGIN = "anna"
PASSWORD = "correct horse battery"


class Tokens:
    """Token factory that returns token-1, token-2 and so on."""

    def __init__(self) -> None:
        self._numbers = count(1)

    def __call__(self) -> str:
        return f"token-{next(self._numbers)}"


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(T0)


@pytest.fixture
def store(clock: FixedClock) -> InMemoryStore:
    store = InMemoryStore()
    AddOperator(uow_factory=store.unit_of_work, clock=clock)(LOGIN, PASSWORD)
    return store


def sign_in(store: InMemoryStore, clock: FixedClock) -> SignIn:
    return SignIn(uow_factory=store.unit_of_work, clock=clock, new_token=Tokens())


@pytest.mark.unit
def test_a_new_account_keeps_a_hash_and_not_the_password(store: InMemoryStore) -> None:
    [operator] = ListOperators(uow_factory=store.unit_of_work)()

    assert (operator.login, operator.active, operator.created_at) == (LOGIN, True, T0)
    assert PASSWORD not in operator.password_hash


@pytest.mark.unit
@pytest.mark.parametrize(
    ("login", "password", "message"),
    [
        (LOGIN, PASSWORD, "already exists"),
        ("Anna", PASSWORD, "login must match"),
        ("petr", "short", "at least"),
    ],
    ids=["taken", "malformed", "short-password"],
)
def test_an_account_is_refused(
    store: InMemoryStore, clock: FixedClock, login: str, password: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AddOperator(uow_factory=store.unit_of_work, clock=clock)(login, password)


@pytest.mark.unit
def test_signing_in_opens_a_session_the_browser_can_use(
    store: InMemoryStore, clock: FixedClock
) -> None:
    signed = sign_in(store, clock)(LOGIN, PASSWORD)

    assert signed.token == "token-1"
    assert signed.session.token_hash == token_hash("token-1")
    assert signed.session.csrf == "token-2"
    assert signed.session.expires_at == T0 + SESSION_LIFETIME
    # Only the hash is stored.
    assert "token-1" not in store.operator_sessions
    assert Authenticate(uow_factory=store.unit_of_work, clock=clock)("token-1") == signed.session


@pytest.mark.unit
def test_a_wrong_password_is_refused_and_counted(store: InMemoryStore, clock: FixedClock) -> None:
    with pytest.raises(SignInRefusedError):
        sign_in(store, clock)(LOGIN, "wrong password")

    assert store.operators[LOGIN].failed_sign_ins == 1
    assert store.operator_sessions == {}


@pytest.mark.unit
@pytest.mark.parametrize("login", ["nobody", "Not A Login"], ids=["unknown", "malformed"])
def test_an_unknown_login_is_refused_like_a_wrong_password(
    store: InMemoryStore, clock: FixedClock, login: str
) -> None:
    with pytest.raises(SignInRefusedError):
        sign_in(store, clock)(login, PASSWORD)
    assert store.operator_sessions == {}


@pytest.mark.unit
def test_a_disabled_account_cannot_sign_in(store: InMemoryStore, clock: FixedClock) -> None:
    DisableOperator(uow_factory=store.unit_of_work)(LOGIN)

    with pytest.raises(SignInRefusedError):
        sign_in(store, clock)(LOGIN, PASSWORD)


@pytest.mark.unit
def test_a_locked_login_is_refused_even_with_the_right_password(
    store: InMemoryStore, clock: FixedClock
) -> None:
    operator = store.operators[LOGIN]
    for _ in range(MAX_FAILED_SIGN_INS):
        operator = operator.failed(T0)
    store.operators[LOGIN] = operator

    with pytest.raises(SignInLockedError) as raised:
        sign_in(store, clock)(LOGIN, PASSWORD)
    assert raised.value.until == T0 + FAILURE_WINDOW

    clock.advance(FAILURE_WINDOW)
    assert sign_in(store, clock)(LOGIN, PASSWORD).session.login == LOGIN
    assert store.operators[LOGIN].failed_sign_ins == 0


@pytest.mark.unit
def test_an_expired_session_ends(store: InMemoryStore, clock: FixedClock) -> None:
    signed = sign_in(store, clock)(LOGIN, PASSWORD)
    clock.advance(SESSION_LIFETIME)

    assert Authenticate(uow_factory=store.unit_of_work, clock=clock)(signed.token) is None
    assert store.operator_sessions[signed.session.token_hash] is None


@pytest.mark.unit
@pytest.mark.parametrize("token", [None, "", "token-404"], ids=["none", "empty", "unknown"])
def test_a_browser_without_a_session_is_not_signed_in(
    store: InMemoryStore, clock: FixedClock, token: str | None
) -> None:
    assert Authenticate(uow_factory=store.unit_of_work, clock=clock)(token) is None


@pytest.mark.unit
def test_disabling_an_account_ends_its_sessions(store: InMemoryStore, clock: FixedClock) -> None:
    signed = sign_in(store, clock)(LOGIN, PASSWORD)

    DisableOperator(uow_factory=store.unit_of_work)(LOGIN)

    assert store.operators[LOGIN].active is False
    assert Authenticate(uow_factory=store.unit_of_work, clock=clock)(signed.token) is None


@pytest.mark.unit
def test_a_new_password_replaces_the_old_one_and_ends_the_sessions(
    store: InMemoryStore, clock: FixedClock
) -> None:
    signed = sign_in(store, clock)(LOGIN, PASSWORD)

    SetOperatorPassword(uow_factory=store.unit_of_work)(LOGIN, "a brand new password")

    assert Authenticate(uow_factory=store.unit_of_work, clock=clock)(signed.token) is None
    with pytest.raises(SignInRefusedError):
        sign_in(store, clock)(LOGIN, PASSWORD)
    assert sign_in(store, clock)(LOGIN, "a brand new password").session.login == LOGIN


@pytest.mark.unit
@pytest.mark.parametrize(
    "command", [DisableOperator, SetOperatorPassword], ids=["disable", "password"]
)
def test_an_unknown_account_cannot_be_changed(store: InMemoryStore, command: type) -> None:
    arguments = ("nobody",) if command is DisableOperator else ("nobody", PASSWORD)
    with pytest.raises(ValueError, match="no operator"):
        command(uow_factory=store.unit_of_work)(*arguments)


@pytest.mark.unit
def test_signing_out_ends_the_session(store: InMemoryStore, clock: FixedClock) -> None:
    signed = sign_in(store, clock)(LOGIN, PASSWORD)

    SignOut(uow_factory=store.unit_of_work)(signed.token)

    assert Authenticate(uow_factory=store.unit_of_work, clock=clock)(signed.token) is None


@pytest.mark.unit
def test_a_form_must_carry_the_token_of_its_session(
    store: InMemoryStore, clock: FixedClock
) -> None:
    session = sign_in(store, clock)(LOGIN, PASSWORD).session

    assert csrf_matches(session, session.csrf) is True
    assert csrf_matches(session, "forged") is False
    assert csrf_matches(session, None) is False
