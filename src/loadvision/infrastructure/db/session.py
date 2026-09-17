"""Database engine and the unit of work on top of it."""

from types import TracebackType
from typing import Self

from sqlalchemy import Connection, Engine, create_engine

from loadvision.infrastructure.db.repositories.assessments import PostgresAssessmentRepository
from loadvision.infrastructure.db.repositories.images import PostgresImageRepository
from loadvision.infrastructure.db.repositories.labels import PostgresLabelRepository
from loadvision.infrastructure.db.repositories.model_versions import (
    PostgresModelVersionRepository,
)
from loadvision.infrastructure.db.repositories.shipments import PostgresShipmentRepository


def create_database_engine(url: str) -> Engine:
    """Creates the engine of the service database.

    Args:
        url: Connection string, such as ``postgresql+psycopg://user:password@host/db``.

    Returns:
        Engine with a connection pool. A pooled connection is checked before use, so a
        database restart does not fail the next request.
    """
    return create_engine(url, pool_pre_ping=True)


class PostgresUnitOfWork:
    """`UnitOfWork` on one pooled connection and its transaction.

    Args:
        engine: Engine of the service database.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._connection: Connection | None = None

    @property
    def shipments(self) -> PostgresShipmentRepository:
        return PostgresShipmentRepository(self._open())

    @property
    def assessments(self) -> PostgresAssessmentRepository:
        return PostgresAssessmentRepository(self._open())

    @property
    def images(self) -> PostgresImageRepository:
        return PostgresImageRepository(self._open())

    @property
    def model_versions(self) -> PostgresModelVersionRepository:
        return PostgresModelVersionRepository(self._open())

    @property
    def labels(self) -> PostgresLabelRepository:
        return PostgresLabelRepository(self._open())

    def __enter__(self) -> Self:
        self._connection = self._engine.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            # Closing rolls back what was not committed and returns the connection to the pool.
            connection.close()

    def commit(self) -> None:
        """Commits the transaction; the next statement starts a new one."""
        self._open().commit()

    def _open(self) -> Connection:
        if self._connection is None:
            raise RuntimeError("unit of work is used outside its with block")
        return self._connection
