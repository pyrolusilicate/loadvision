"""Typed application settings loaded from YAML and environment variables."""

from typing import Final

# Connection string of the service database, such as postgresql+psycopg://user:password@host/db;
# kept in the environment, never in a committed file.
DATABASE_URL_ENV: Final = "DATABASE_URL"
