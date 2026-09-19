"""The service database, opened for the time of one command."""

import functools
from collections.abc import Iterator
from contextlib import contextmanager

from loadvision.application.ports import UnitOfWorkFactory
from loadvision.cli.errors import CommandError
from loadvision.infrastructure.db.session import PostgresUnitOfWork, create_database_engine
from loadvision.settings import DATABASE_URL_ENV, database_url


@contextmanager
def unit_of_work_factory() -> Iterator[UnitOfWorkFactory]:
    """Opens the database from `DATABASE_URL` and closes it when the command is done.

    Raises:
        CommandError: If `DATABASE_URL` is not set.
    """
    try:
        url = database_url()
    except RuntimeError as error:
        # uv does not read .env on its own, so the usual cause is a command run without it.
        raise CommandError(
            f"{DATABASE_URL_ENV} is not set; create .env from .env.example and run "
            "uv run --env-file .env loadvision <command>"
        ) from error
    engine = create_database_engine(url)
    try:
        yield functools.partial(PostgresUnitOfWork, engine)
    finally:
        engine.dispose()
