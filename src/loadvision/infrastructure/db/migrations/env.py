"""Alembic environment: applies migrations to the database from DATABASE_URL."""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from loadvision.infrastructure.db.tables import metadata
from loadvision.settings import DATABASE_URL_ENV


def _database_url() -> str:
    # Tests pass the URL of a throwaway database through the Alembic config; the command line
    # takes it from the environment, so credentials never land in alembic.ini.
    url = context.config.get_main_option("sqlalchemy.url") or os.environ.get(DATABASE_URL_ENV)
    if not url:
        raise RuntimeError(
            f"{DATABASE_URL_ENV} is not set; run `just db-migrate`, which reads it from .env"
        )
    return url


def _run_offline() -> None:
    context.configure(url=_database_url(), target_metadata=metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def _run_online() -> None:
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=metadata)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name, disable_existing_loggers=False)

if context.is_offline_mode():
    _run_offline()
else:
    _run_online()
