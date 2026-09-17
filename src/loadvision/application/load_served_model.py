"""Use case: load the model version the service answers with."""

from collections.abc import Callable
from dataclasses import dataclass

from loadvision.application.ports import Clock, LoadPredictor, UnitOfWorkFactory
from loadvision.domain.model_version import ModelVersion, ModelVersionId, ModelVersionStatus
from loadvision.errors import ErrorCode, LoadvisionError


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadServedModel:
    """Loads the active model version, or activates the initial one in an empty database.

    The database decides which version is active, so a restart keeps a version activated
    there. Only while no version is active does the service take `initial_version`: it is
    registered and activated after its package has loaded, so a broken package never becomes
    active. Every assessment refers to a registered version, so a model must not answer before
    its version is in the database.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
        load_package: Loads the package of a version into a predictor.
        initial_version: Version to start with while no version is active.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock
    load_package: Callable[[ModelVersionId], LoadPredictor]
    initial_version: ModelVersionId

    def __call__(self) -> LoadPredictor:
        """Loads the model.

        Returns:
            Predictor of the active version.

        Raises:
            LoadvisionError: `MODEL_PACKAGE_INVALID` if the package declares another version;
                `MODEL_VERSION_STATUS_CONFLICT` if the initial version was rejected.
            Exception: Whatever `load_package` raises for a missing or broken package.
        """
        with self.uow_factory() as uow:
            active = uow.model_versions.get_active()
        version = self.initial_version if active is None else active.version
        predictor = self.load_package(version)
        if predictor.version != version:
            raise LoadvisionError(
                ErrorCode.MODEL_PACKAGE_INVALID,
                f"package of {version.value} declares version {predictor.version.value}",
            )
        if active is None:
            self._activate(version)
        return predictor

    def _activate(self, version: ModelVersionId) -> None:
        now = self.clock()
        with self.uow_factory() as uow:
            stored = uow.model_versions.get(version)
            if stored is None:
                stored = ModelVersion(
                    version=version, status=ModelVersionStatus.CANDIDATE, created_at=now
                )
                uow.model_versions.add(stored)
            uow.model_versions.update(stored.activate(now))
            uow.commit()
