"""Operator account and sign-in session for the operator pages."""

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Final, Self

from loadvision.domain.assessment import SHA256_HEX_PATTERN
from loadvision.domain.timestamps import ensure_aware

# Lowercase so that "Ivan" and "ivan" are one person; the login also becomes a label author.
OPERATOR_LOGIN_PATTERN: Final = "[a-z0-9][a-z0-9._-]{0,63}"
# Failed sign-ins allowed within one window before the login locks until the window ends.
MAX_FAILED_SIGN_INS: Final = 30
FAILURE_WINDOW: Final = timedelta(minutes=15)
SESSION_LIFETIME: Final = timedelta(hours=12)

_LOGIN_RE: Final = re.compile(OPERATOR_LOGIN_PATTERN)
_SHA256_RE: Final = re.compile(SHA256_HEX_PATTERN)


def check_login(login: str) -> None:
    """Validates an operator login.

    Raises:
        ValueError: If `login` does not match `OPERATOR_LOGIN_PATTERN`.
    """
    if _LOGIN_RE.fullmatch(login) is None:
        raise ValueError(f"login must match {OPERATOR_LOGIN_PATTERN}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Operator:
    """A person who confirms loads on the operator pages.

    Accounts are made from the command line, never from the pages, so the pages have no way
    to grant themselves access.

    Attributes:
        login: Account name; also written as the author of the labels this person confirms.
        password_hash: Password hash, never the password.
        active: Whether the account may sign in.
        created_at: When the account was made, timezone-aware.
        failed_sign_ins: Failed sign-ins in the current window.
        first_failure_at: When the current window of failures began, or None without one.
    """

    login: str
    password_hash: str
    active: bool
    created_at: datetime
    failed_sign_ins: int = 0
    first_failure_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validates the login, the timestamps and the failure counter.

        Raises:
            ValueError: If the login is malformed, a timestamp has no UTC offset, the counter
                is negative, or the counter and the window start disagree.
        """
        check_login(self.login)
        ensure_aware(self.created_at, "created_at")
        if self.first_failure_at is not None:
            ensure_aware(self.first_failure_at, "first_failure_at")
        if self.failed_sign_ins < 0:
            raise ValueError("failed_sign_ins must not be negative")
        if (self.failed_sign_ins == 0) != (self.first_failure_at is None):
            raise ValueError("a failure window starts with the first failure")

    def locked_until(self, now: datetime) -> datetime | None:
        """Returns when sign-in opens again, or None when it is open now."""
        if self.first_failure_at is None or self.failed_sign_ins < MAX_FAILED_SIGN_INS:
            return None
        until = self.first_failure_at + FAILURE_WINDOW
        return until if now < until else None

    def failed(self, now: datetime) -> Self:
        """Returns the account after a wrong password; a new window starts once one is over."""
        if self.first_failure_at is None or now - self.first_failure_at >= FAILURE_WINDOW:
            return replace(self, failed_sign_ins=1, first_failure_at=now)
        return replace(self, failed_sign_ins=self.failed_sign_ins + 1)

    def signed_in(self) -> Self:
        """Returns the account after a successful sign-in, with its failures forgotten."""
        return replace(self, failed_sign_ins=0, first_failure_at=None)

    def disabled(self) -> Self:
        """Returns the account that may no longer sign in."""
        return replace(self, active=False)

    def with_password(self, password_hash: str) -> Self:
        """Returns the account with a new password, active again and with no failures."""
        return replace(
            self, password_hash=password_hash, active=True, failed_sign_ins=0, first_failure_at=None
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class OperatorSession:
    """A signed-in browser.

    The browser holds a random token; the database holds only its hash, so a copy of the
    database gives nobody a way in.

    Attributes:
        token_hash: SHA-256 of the session token, lowercase hex.
        login: Operator who signed in.
        csrf: Token every form of this session sends back, so another site cannot post on the
            operator's behalf.
        created_at: When the operator signed in, timezone-aware.
        expires_at: When the session stops working, timezone-aware.
    """

    token_hash: str
    login: str
    csrf: str
    created_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        """Validates the hash, the login and the lifetime.

        Raises:
            ValueError: If the hash is not 64 lowercase hex characters, the login is malformed,
                the CSRF token is empty, a timestamp has no UTC offset or the session ends
                before it starts.
        """
        if _SHA256_RE.fullmatch(self.token_hash) is None:
            raise ValueError("token_hash must be 64 lowercase hex characters")
        check_login(self.login)
        if not self.csrf:
            raise ValueError("csrf must not be empty")
        ensure_aware(self.created_at, "created_at")
        ensure_aware(self.expires_at, "expires_at")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")

    def expired(self, now: datetime) -> bool:
        """Whether the session no longer lets the browser in."""
        return now >= self.expires_at
