"""Throwaway Postgres databases and Alembic configs for integration tests.

Tests never touch the database named in DATABASE_URL: they use its server to create a database
with a random name and drop it afterwards.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from alembic.config import Config
from sqlalchemy import URL, Engine, create_engine, pool, text

ALEMBIC_INI = Path(__file__).parents[2] / "alembic.ini"


@contextmanager
def throwaway_database(server_url: URL) -> Iterator[URL]:
    name = f"loadvision_test_{uuid4().hex[:12]}"
    admin = create_engine(server_url, isolation_level="AUTOCOMMIT", poolclass=pool.NullPool)
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))
        try:
            yield server_url.set(database=name)
        finally:
            with admin.connect() as connection:
                connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    finally:
        admin.dispose()


def alembic_config(url: URL) -> Config:
    # A config without the ini file: env.py would otherwise reconfigure logging inside pytest.
    config = Config()
    config.set_main_option(
        "script_location", Config(str(ALEMBIC_INI)).get_main_option("script_location") or ""
    )
    config.set_main_option("path_separator", "os")
    # ConfigParser treats "%" as interpolation, and a password may contain it.
    config.set_main_option(
        "sqlalchemy.url", url.render_as_string(hide_password=False).replace("%", "%%")
    )
    return config


@contextmanager
def engine_for(url: URL) -> Iterator[Engine]:
    engine = create_engine(url, poolclass=pool.NullPool)
    try:
        yield engine
    finally:
        engine.dispose()
