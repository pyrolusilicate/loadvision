"""Use case: load the model version the service answers with."""

from collections.abc import Callable
from dataclasses import dataclass

from loadvision.application.model_versions import ActivateModel
from loadvision.application.ports import Clock, LoadPredictor, UnitOfWorkFactory
from loadvision.domain.model_version import ModelVersion, ModelVersionId
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

    @property
    def _activate(self) -> ActivateModel:
        return ActivateModel(uow_factory=self.uow_factory, clock=self.clock)

    def wanted_version(self) -> ModelVersionId:
        """Returns the version the service should answer with, without loading it.

        Returns:
            The active version, or `initial_version` while none is active.
        """
        active = self._active()
        return self.initial_version if active is None else active.version

    def __call__(self) -> LoadPredictor:
        """Loads the model.

        Returns:
            Predictor of the active version.

        Raises:
            LoadvisionError: `MODEL_PACKAGE_INVALID` if the package declares another version;
                `MODEL_VERSION_STATUS_CONFLICT` if the initial version was rejected.
            Exception: Whatever `load_package` raises for a missing or broken package.
        """
        active = self._active()
        version = self.initial_version if active is None else active.version
        predictor = self.load_package(version)
        if predictor.version != version:
            raise LoadvisionError(
                ErrorCode.MODEL_PACKAGE_INVALID,
                f"package of {version.value} declares version {predictor.version.value}",
            )
        if active is None:
            # The version the service starts with may be brand new, so it is registered here.
            self._activate(version, register=True)
        return predictor

    def _active(self) -> ModelVersion | None:
        with self.uow_factory() as uow:
            return uow.model_versions.get_active()
