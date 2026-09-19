from datetime import datetime, timedelta

import pytest

from loadvision.domain.operator import (
    FAILURE_WINDOW,
    MAX_FAILED_SIGN_INS,
    Operator,
    OperatorSession,
    check_login,
)
from tests.builders import T0, make_operator, make_session


@pytest.mark.unit
@pytest.mark.parametrize("login", ["anna", "operator-17", "i.ivanov", "a", "a" * 64])
def test_well_formed_logins_are_accepted(login: str) -> None:
    check_login(login)


@pytest.mark.unit
@pytest.mark.parametrize("login", ["Anna", ".hidden", "-anna", "", "a" * 65, "ann a", "анна"])
def test_malformed_logins_are_refused(login: str) -> None:
    with pytest.raises(ValueError, match="login must match"):
        check_login(login)


@pytest.mark.unit
def test_a_login_locks_after_too_many_failures_within_the_window() -> None:
    operator = make_operator()
    for step in range(MAX_FAILED_SIGN_INS):
        assert operator.locked_until(T0 + timedelta(seconds=step)) is None
        operator = operator.failed(T0 + timedelta(seconds=step))

    assert operator.locked_until(T0 + timedelta(minutes=1)) == T0 + FAILURE_WINDOW
    assert operator.locked_until(T0 + FAILURE_WINDOW) is None


@pytest.mark.unit
def test_a_failure_after_the_window_starts_a_new_one() -> None:
    operator = make_operator().failed(T0).failed(T0 + timedelta(minutes=1))

    later = operator.failed(T0 + FAILURE_WINDOW)

    assert (later.failed_sign_ins, later.first_failure_at) == (1, T0 + FAILURE_WINDOW)


@pytest.mark.unit
def test_a_successful_sign_in_forgets_the_failures() -> None:
    operator = make_operator().failed(T0).signed_in()
    assert (operator.failed_sign_ins, operator.first_failure_at) == (0, None)


@pytest.mark.unit
def test_a_new_password_enables_the_account_and_clears_the_failures() -> None:
    operator = make_operator().failed(T0).disabled().with_password("scrypt$11$22")

    assert operator.active is True
    assert operator.password_hash == "scrypt$11$22"
    assert operator.failed_sign_ins == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failed", "first"),
    [(-1, None), (2, None), (0, T0)],
    ids=["negative", "count-without-window", "window-without-count"],
)
def test_a_counter_and_window_that_disagree_are_refused(
    failed: int, first: datetime | None
) -> None:
    with pytest.raises(ValueError):
        Operator(
            login="anna",
            password_hash="scrypt$00$00",
            active=True,
            created_at=T0,
            failed_sign_ins=failed,
            first_failure_at=first,
        )


@pytest.mark.unit
def test_a_session_expires_at_its_end() -> None:
    session = make_session()
    assert session.expired(session.expires_at - timedelta(seconds=1)) is False
    assert session.expired(session.expires_at) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("token_hash", "csrf", "hours"),
    [("A" * 64, "c", 1), ("a" * 63, "c", 1), ("a" * 64, "", 1), ("a" * 64, "c", 0)],
    ids=["upper-case", "short-hash", "no-csrf", "ends-when-it-starts"],
)
def test_an_impossible_session_is_refused(token_hash: str, csrf: str, hours: int) -> None:
    with pytest.raises(ValueError):
        OperatorSession(
            token_hash=token_hash,
            login="anna",
            csrf=csrf,
            created_at=T0,
            expires_at=T0 + timedelta(hours=hours),
        )
