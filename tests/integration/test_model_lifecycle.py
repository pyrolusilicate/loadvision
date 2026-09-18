import functools
from datetime import timedelta

import pytest
from sqlalchemy import Engine

from loadvision.application.model_versions import ActivateModel, RegisterModel, RollbackModel
from loadvision.application.ports import UnitOfWorkFactory
from loadvision.domain.model_version import ModelVersionId, ModelVersionStatus
from loadvision.infrastructure.db.session import PostgresUnitOfWork
from tests.builders import T0
from tests.fakes.clock import FixedClock

TAGS = ("v1", "v2", "v3")


@pytest.mark.integration
def test_versions_are_registered_activated_and_rolled_back(engine: Engine) -> None:
    factory: UnitOfWorkFactory = functools.partial(PostgresUnitOfWork, engine)
    clock = FixedClock(T0)
    register = RegisterModel(uow_factory=factory, clock=clock)
    activate = ActivateModel(uow_factory=factory, clock=clock)

    for tag in TAGS:
        register(ModelVersionId(tag))
        activate(ModelVersionId(tag))
        clock.advance(timedelta(hours=1))

    with factory() as uow:
        active = uow.model_versions.get_active()
        assert active is not None and active.version == ModelVersionId("v3")

    back = RollbackModel(uow_factory=factory, activate=activate)()

    assert back.version == ModelVersionId("v2")
    with factory() as uow:
        stored = {
            version.version.value: version.status for version in uow.model_versions.list_all()
        }
    assert stored == {
        "v1": ModelVersionStatus.RETIRED,
        "v2": ModelVersionStatus.ACTIVE,
        "v3": ModelVersionStatus.RETIRED,
    }


@pytest.mark.integration
def test_registering_a_known_version_again_is_refused(engine: Engine) -> None:
    factory: UnitOfWorkFactory = functools.partial(PostgresUnitOfWork, engine)
    register = RegisterModel(uow_factory=factory, clock=FixedClock(T0))
    register(ModelVersionId("v1"))

    with pytest.raises(ValueError, match="already registered"):
        register(ModelVersionId("v1"))

    with factory() as uow:
        assert len(uow.model_versions.list_all()) == 1
