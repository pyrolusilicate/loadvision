"""Use cases for the lifecycle of model versions: register, activate, roll back."""

from dataclasses import dataclass
from datetime import datetime

from loadvision.application.ports import Clock, UnitOfWork, UnitOfWorkFactory
from loadvision.domain.model_version import ModelVersion, ModelVersionId, ModelVersionStatus


@dataclass(frozen=True, slots=True, kw_only=True)
class RegisterModel:
    """Puts a model version into the database as a candidate.

    Registration and activation are separate on purpose: a version is registered when its
    package exists, and activated when someone decides it should answer.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock

    def __call__(self, version: ModelVersionId) -> ModelVersion:
        """Registers the version.

        Returns:
            The registered version.

        Raises:
            ValueError: If the version is already registered; a second registration would hide
                which package the first one meant.
        """
        with self.uow_factory() as uow:
            if uow.model_versions.get(version) is not None:
                raise ValueError(f"model version {version.value} is already registered")
            registered = ModelVersion(
                version=version, status=ModelVersionStatus.CANDIDATE, created_at=self.clock()
            )
            uow.model_versions.add(registered)
            uow.commit()
        return registered


@dataclass(frozen=True, slots=True, kw_only=True)
class ActivateModel:
    """Makes a version the one the service answers with; the previous one retires.

    Retiring and activating happen in one transaction, and the database keeps at most one
    active version, so two activations at once cannot both succeed.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock

    def __call__(self, version: ModelVersionId, *, register: bool = False) -> ModelVersion:
        """Activates the version; activating the already active one changes nothing.

        Args:
            version: Version to activate.
            register: Register the version as a candidate first if the database does not know
                it. The service does that for the version it starts with; a person activates
                only versions that were registered with their package.

        Returns:
            The active version.

        Raises:
            ValueError: If the version is not registered and `register` is False.
            LoadvisionError: `MODEL_VERSION_STATUS_CONFLICT` if the version was rejected by the
                gate; a rejected version never answers.
        """
        now = self.clock()
        with self.uow_factory() as uow:
            stored = _stored(uow, version, register=register, now=now)
            if stored.status is ModelVersionStatus.ACTIVE:
                return stored
            activated = stored.activate(now)
            current = uow.model_versions.get_active()
            if current is not None:
                uow.model_versions.update(current.retire())
            uow.model_versions.update(activated)
            uow.commit()
        return activated


@dataclass(frozen=True, slots=True, kw_only=True)
class RollbackModel:
    """Brings back the version that answered before the current one.

    Attributes:
        uow_factory: Opens database transactions.
        activate: Activation use case; the rollback only chooses which version to activate.
    """

    uow_factory: UnitOfWorkFactory
    activate: ActivateModel

    def __call__(self) -> ModelVersion:
        """Activates the version that was active most recently before the current one.

        Returns:
            The version brought back.

        Raises:
            ValueError: If no version has been retired yet, so there is nothing to go back to.
            LoadvisionError: `MODEL_VERSION_STATUS_CONFLICT` if that version cannot be
                activated again.
        """
        with self.uow_factory() as uow:
            retired = [
                (version.activated_at, version)
                for version in uow.model_versions.list_all()
                if version.status is ModelVersionStatus.RETIRED and version.activated_at
            ]
        if not retired:
            raise ValueError("no retired version to roll back to")
        _, previous = max(retired, key=_activation_time)
        return self.activate(previous.version)


def _stored(
    uow: UnitOfWork, version: ModelVersionId, *, register: bool, now: datetime
) -> ModelVersion:
    stored = uow.model_versions.get(version)
    if stored is not None:
        return stored
    if not register:
        raise ValueError(f"model version {version.value} is not registered")
    registered = ModelVersion(version=version, status=ModelVersionStatus.CANDIDATE, created_at=now)
    uow.model_versions.add(registered)
    return registered


def _activation_time(pair: tuple[datetime | None, ModelVersion]) -> datetime:
    at, _ = pair
    assert at is not None, "retired versions are filtered by their activation time"
    return at
