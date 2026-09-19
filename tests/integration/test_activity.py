import time

import pytest
from sqlalchemy import Engine

from loadvision.infrastructure.db.activity import WORKER_APPLICATION, worker_connected
from loadvision.infrastructure.db.session import create_database_engine

GONE_SECONDS = 5.0


@pytest.mark.integration
def test_a_worker_is_seen_while_it_holds_a_connection(engine: Engine) -> None:
    url = engine.url.render_as_string(hide_password=False)
    worker = create_database_engine(url, application_name=WORKER_APPLICATION)
    before = worker_connected(engine)

    with worker.connect():
        pass
    # The connection went back to the pool and stays open, as between two polls of a worker.
    while_running = worker_connected(engine)
    worker.dispose()
    deadline = time.monotonic() + GONE_SECONDS
    while worker_connected(engine) and time.monotonic() < deadline:
        time.sleep(0.05)

    assert (before, while_running, worker_connected(engine)) == (False, True, False)


@pytest.mark.integration
def test_the_service_itself_is_not_taken_for_a_worker(engine: Engine) -> None:
    with engine.connect():
        assert worker_connected(engine) is False
