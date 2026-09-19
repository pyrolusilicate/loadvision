"""Use cases for operator accounts and signing in to the operator pages."""

import hashlib
import hmac
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import cache

from loadvision.application.passwords import check_password, hash_password, verify_password
from loadvision.application.ports import Clock, UnitOfWorkFactory
from loadvision.domain.operator import (
    SESSION_LIFETIME,
    Operator,
    OperatorSession,
    check_login,
)

type TokenFactory = Callable[[], str]


class SignInRefusedError(Exception):
    """Wrong login or password; the page never says which, so logins cannot be guessed."""


class SignInLockedError(Exception):
    """Too many wrong passwords for this login; sign-in opens again at `until`.

    Args:
        until: When sign-in opens again.
    """

    def __init__(self, until: datetime) -> None:
        super().__init__(f"sign-in is locked until {until.isoformat()}")
        self.until = until


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedIn:
    """A new session and the token the browser keeps.

    Attributes:
        token: Session token for the cookie; only its hash is stored.
        session: The stored session.
    """

    token: str
    session: OperatorSession


def new_token() -> str:
    """Returns a random token for a session or a form, 32 bytes, URL-safe."""
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    """Returns what the database stores for a session token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def csrf_matches(session: OperatorSession, submitted: str | None) -> bool:
    """Whether a form carries the token of its session, compared in constant time."""
    return submitted is not None and hmac.compare_digest(session.csrf, submitted)


@dataclass(frozen=True, slots=True, kw_only=True)
class AddOperator:
    """Creates an account; accounts are made from the command line only.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock

    def __call__(self, login: str, password: str) -> Operator:
        """Creates the account.

        Raises:
            ValueError: If the login is malformed or taken, or the password is too short.
        """
        check_login(login)
        check_password(password)
        with self.uow_factory() as uow:
            if uow.operators.get(login) is not None:
                raise ValueError(f"operator {login} already exists")
            operator = Operator(
                login=login,
                password_hash=hash_password(password),
                active=True,
                created_at=self.clock(),
            )
            uow.operators.add(operator)
            uow.commit()
        return operator


@dataclass(frozen=True, slots=True, kw_only=True)
class SetOperatorPassword:
    """Gives an account a new password; it is enabled again and every session of it ends.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self, login: str, password: str) -> Operator:
        """Sets the password.

        Raises:
            ValueError: If there is no such account or the password is too short.
        """
        check_password(password)
        with self.uow_factory() as uow:
            operator = _existing(uow.operators.get(login), login)
            changed = operator.with_password(hash_password(password))
            uow.operators.update(changed)
            uow.operator_sessions.delete_for(login)
            uow.commit()
        return changed


@dataclass(frozen=True, slots=True, kw_only=True)
class DisableOperator:
    """Stops an account from signing in and ends the sessions it has open.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self, login: str) -> Operator:
        """Disables the account.

        Raises:
            ValueError: If there is no such account.
        """
        with self.uow_factory() as uow:
            disabled = _existing(uow.operators.get(login), login).disabled()
            uow.operators.update(disabled)
            uow.operator_sessions.delete_for(login)
            uow.commit()
        return disabled


@dataclass(frozen=True, slots=True, kw_only=True)
class ListOperators:
    """Reads every account, by login.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self) -> list[Operator]:
        """Returns the accounts."""
        with self.uow_factory() as uow:
            return uow.operators.list_all()


@dataclass(frozen=True, slots=True, kw_only=True)
class SignIn:
    """Checks a login and a password and opens a session.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
        new_token: Random tokens for the session and its forms.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock
    new_token: TokenFactory = new_token

    def __call__(self, login: str, password: str) -> SignedIn:
        """Signs the operator in.

        Returns:
            The session and the token for the browser.

        Raises:
            SignInRefusedError: If the login is unknown or disabled, or the password is wrong.
            SignInLockedError: If the login had too many wrong passwords recently.
        """
        now = self.clock()
        with self.uow_factory() as uow:
            operator = uow.operators.get(login) if _well_formed(login) else None
            if operator is None or not operator.active:
                # The same slow hash runs as for a real account, so the time of the answer
                # does not tell whether the login exists.
                verify_password(password, _stand_in_hash())
                raise SignInRefusedError
            locked_until = operator.locked_until(now)
            if locked_until is not None:
                raise SignInLockedError(locked_until)
            if not verify_password(password, operator.password_hash):
                uow.operators.update(operator.failed(now))
                uow.commit()
                raise SignInRefusedError
            token = self.new_token()
            session = OperatorSession(
                token_hash=token_hash(token),
                login=operator.login,
                csrf=self.new_token(),
                created_at=now,
                expires_at=now + SESSION_LIFETIME,
            )
            uow.operator_sessions.add(session)
            uow.operators.update(operator.signed_in())
            uow.commit()
        return SignedIn(token=token, session=session)


@dataclass(frozen=True, slots=True, kw_only=True)
class Authenticate:
    """Finds the session of a browser, ending it if it expired or its account was disabled.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock

    def __call__(self, token: str | None) -> OperatorSession | None:
        """Returns the live session of the token, or None."""
        if not token:
            return None
        now = self.clock()
        with self.uow_factory() as uow:
            session = uow.operator_sessions.get(token_hash(token))
            if session is None:
                return None
            operator = uow.operators.get(session.login)
            if session.expired(now) or operator is None or not operator.active:
                uow.operator_sessions.delete(session.token_hash)
                uow.commit()
                return None
        return session


@dataclass(frozen=True, slots=True, kw_only=True)
class SignOut:
    """Ends the session of a browser.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self, token: str) -> None:
        """Ends the session; an unknown token changes nothing."""
        with self.uow_factory() as uow:
            uow.operator_sessions.delete(token_hash(token))
            uow.commit()


def _existing(operator: Operator | None, login: str) -> Operator:
    if operator is None:
        raise ValueError(f"there is no operator {login}")
    return operator


def _well_formed(login: str) -> bool:
    try:
        check_login(login)
    except ValueError:
        return False
    return True


@cache
def _stand_in_hash() -> str:
    return hash_password(new_token())
