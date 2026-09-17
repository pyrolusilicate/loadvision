"""The service database, opened for the time of one command."""

import functools
from collections.abc import Iterator
from contextlib import contextmanager

from loadvision.application.ports import UnitOfWorkFactory
from loadvision.infrastructure.db.session import PostgresUnitOfWork, create_database_engine
from loadvision.settings import database_url


@contextmanager
def unit_of_work_factory() -> Iterator[UnitOfWorkFactory]:
    """Opens the database from `DATABASE_URL` and closes it when the command is done.

    Raises:
        RuntimeError: If `DATABASE_URL` is not set.
    """
    engine = create_database_engine(database_url())
    try:
        yield functools.partial(PostgresUnitOfWork, engine)
    finally:
        engine.dispose()
