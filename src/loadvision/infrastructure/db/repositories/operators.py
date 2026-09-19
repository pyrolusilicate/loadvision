"""Postgres repositories for operator accounts and their sessions."""

from typing import Any

from sqlalchemy import Connection, Row, delete, insert, select, update

from loadvision.domain.operator import Operator, OperatorSession
from loadvision.infrastructure.db.tables import operator_sessions, operators


class PostgresOperatorRepository:
    """`OperatorRepository` on the connection of one unit of work."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def get(self, login: str) -> Operator | None:
        row = self._connection.execute(
            select(operators).where(operators.c.login == login)
        ).one_or_none()
        return None if row is None else _operator(row)

    def add(self, operator: Operator) -> None:
        self._connection.execute(insert(operators).values(_operator_values(operator)))

    def update(self, operator: Operator) -> None:
        """Saves the account.

        Raises:
            ValueError: If the account was never stored.
        """
        result = self._connection.execute(
            update(operators)
            .where(operators.c.login == operator.login)
            .values(_operator_values(operator))
        )
        if result.rowcount == 0:
            raise ValueError(f"operator {operator.login} is not stored")

    def list_all(self) -> list[Operator]:
        rows = self._connection.execute(select(operators).order_by(operators.c.login))
        return [_operator(row) for row in rows]


class PostgresOperatorSessionRepository:
    """`OperatorSessionRepository` on the connection of one unit of work."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def add(self, session: OperatorSession) -> None:
        self._connection.execute(
            insert(operator_sessions).values(
                token_hash=session.token_hash,
                login=session.login,
                csrf=session.csrf,
                created_at=session.created_at,
                expires_at=session.expires_at,
            )
        )

    def get(self, token_hash: str) -> OperatorSession | None:
        row = self._connection.execute(
            select(operator_sessions).where(operator_sessions.c.token_hash == token_hash)
        ).one_or_none()
        if row is None:
            return None
        return OperatorSession(
            token_hash=row.token_hash,
            login=row.login,
            csrf=row.csrf,
            created_at=row.created_at,
            expires_at=row.expires_at,
        )

    def delete(self, token_hash: str) -> None:
        self._connection.execute(
            delete(operator_sessions).where(operator_sessions.c.token_hash == token_hash)
        )

    def delete_for(self, login: str) -> None:
        self._connection.execute(
            delete(operator_sessions).where(operator_sessions.c.login == login)
        )


def _operator_values(operator: Operator) -> dict[str, object]:
    return {
        "login": operator.login,
        "password_hash": operator.password_hash,
        "active": operator.active,
        "created_at": operator.created_at,
        "failed_sign_ins": operator.failed_sign_ins,
        "first_failure_at": operator.first_failure_at,
    }


def _operator(row: Row[Any]) -> Operator:
    return Operator(
        login=row.login,
        password_hash=row.password_hash,
        active=row.active,
        created_at=row.created_at,
        failed_sign_ins=row.failed_sign_ins,
        first_failure_at=row.first_failure_at,
    )
