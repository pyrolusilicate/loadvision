"""Fixtures with a migrated throwaway Postgres database; see tests/integration/database.py."""

import os
from collections.abc import Iterator

import pytest
from alembic import command
from sqlalchemy import URL, Connection, Engine, make_url

from loadvision.settings import DATABASE_URL_ENV
from tests.integration.database import alembic_config, engine_for, throwaway_database


@pytest.fixture(scope="session")
def server_url() -> URL:
    raw = os.environ.get(DATABASE_URL_ENV)
    if not raw:
        # Like the data tests without data/: a plain `pytest` run skips them. With the variable
        # set and Postgres down, the tests fail loudly on connect.
        pytest.skip(f"{DATABASE_URL_ENV} is not set; run `just db-up`, then `just integration`")
    return make_url(raw)


@pytest.fixture(scope="session")
def migrated_engine(server_url: URL) -> Iterator[Engine]:
    with throwaway_database(server_url) as url, engine_for(url) as engine:
        command.upgrade(alembic_config(url), "head")
        yield engine


@pytest.fixture
def connection(migrated_engine: Engine) -> Iterator[Connection]:
    """Connection whose changes are rolled back after the test."""
    with migrated_engine.connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()
