"""Who is connected to the service database: the pages ask whether a worker is running."""

from typing import Final

from sqlalchemy import Engine, text

# The name a worker gives its connections, so the service can see it in pg_stat_activity.
WORKER_APPLICATION: Final = "loadvision-worker"

_WORKER_CONNECTION: Final = text(
    "SELECT 1 FROM pg_stat_activity"
    " WHERE datname = current_database() AND application_name = :name LIMIT 1"
)


def worker_connected(engine: Engine) -> bool:
    """Whether a worker process holds a connection to the database of `engine`.

    A running worker keeps its pooled connection open between polls, and the connection goes
    away with the process, so this answers without a heartbeat table.
    """
    with engine.connect() as connection:
        found = connection.execute(_WORKER_CONNECTION, {"name": WORKER_APPLICATION}).first()
    return found is not None
