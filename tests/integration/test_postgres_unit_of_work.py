import threading
from uuid import UUID

import pytest
from sqlalchemy import Connection, Engine, QueuePool, event

from loadvision.application.ports import AssessmentAlreadyExistsError, UnitOfWorkFactory
from loadvision.infrastructure.db.session import PostgresUnitOfWork, create_database_engine
from tests.builders import make_assessment, make_image_record, make_model_version, make_shipment
from tests.port_contracts.unit_of_work import UnitOfWorkContract

# Long enough for a thread that is not blocked to finish its insert.
BLOCKED_SECONDS = 0.5


@pytest.mark.integration
class TestPostgresUnitOfWork(UnitOfWorkContract):
    @pytest.fixture
    def uow_factory(self, engine: Engine) -> UnitOfWorkFactory:
        return lambda: PostgresUnitOfWork(engine)


@pytest.mark.integration
def test_parallel_duplicate_waits_for_the_first_commit_and_raises(engine: Engine) -> None:
    first = make_assessment(id=UUID(int=1))
    with PostgresUnitOfWork(engine) as uow:
        uow.shipments.add(make_shipment())
        uow.images.add(make_image_record())
        uow.model_versions.add(make_model_version(first.model_version.value))
        uow.commit()
    errors: list[BaseException] = []

    def add_duplicate() -> None:
        with PostgresUnitOfWork(engine) as uow:
            try:
                uow.assessments.add(make_assessment(id=UUID(int=2)))
            except AssessmentAlreadyExistsError as error:
                errors.append(error)

    with PostgresUnitOfWork(engine) as uow:
        uow.assessments.add(first)
        duplicate = threading.Thread(target=add_duplicate)
        duplicate.start()
        duplicate.join(BLOCKED_SECONDS)
        assert duplicate.is_alive(), "the insert must wait for the first transaction"
        uow.commit()
    duplicate.join()
    assert len(errors) == 1
    with PostgresUnitOfWork(engine) as uow:
        assert uow.assessments.get(first.id) == first
        assert uow.assessments.get(UUID(int=2)) is None


@pytest.mark.integration
def test_unit_of_work_closes_its_connection(migrated_engine: Engine) -> None:
    pooled = create_database_engine(migrated_engine.url.render_as_string(hide_password=False))
    # Keeping the connections here stops the garbage collector from returning a forgotten one
    # to the pool, which would hide the leak.
    opened: list[Connection] = []
    event.listen(pooled, "engine_connect", opened.append)
    pool = pooled.pool
    assert isinstance(pool, QueuePool)
    try:
        # No statement runs, so a leaked connection holds no locks that would block the cleanup
        # of later tests.
        with PostgresUnitOfWork(pooled):
            assert pool.checkedout() == 1
        assert [connection.closed for connection in opened] == [True]
        assert pool.checkedout() == 0
    finally:
        pooled.dispose()
