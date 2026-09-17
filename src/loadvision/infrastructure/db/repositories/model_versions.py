"""Postgres repository for model versions."""

from typing import Any

from sqlalchemy import Connection, Row, insert, select, update

from loadvision.domain.model_version import ModelVersion, ModelVersionId, ModelVersionStatus
from loadvision.infrastructure.db.tables import model_versions


class PostgresModelVersionRepository:
    """`ModelVersionRepository` on the connection of one unit of work.

    The single active version is kept by a partial unique index: activating a second version
    before retiring the first fails in the database, whatever the caller does.
    """

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def get(self, version: ModelVersionId) -> ModelVersion | None:
        row = self._connection.execute(
            select(model_versions).where(model_versions.c.version == version.value)
        ).one_or_none()
        return None if row is None else _model_version(row)

    def get_active(self) -> ModelVersion | None:
        row = self._connection.execute(
            select(model_versions).where(model_versions.c.status == ModelVersionStatus.ACTIVE.value)
        ).one_or_none()
        return None if row is None else _model_version(row)

    def list_all(self) -> list[ModelVersion]:
        rows = self._connection.execute(
            select(model_versions).order_by(model_versions.c.created_at.desc())
        )
        return [_model_version(row) for row in rows]

    def add(self, model_version: ModelVersion) -> None:
        self._connection.execute(
            insert(model_versions).values(
                version=model_version.version.value,
                status=model_version.status.value,
                created_at=model_version.created_at,
                activated_at=model_version.activated_at,
            )
        )

    def update(self, model_version: ModelVersion) -> None:
        """Saves the status and activation time.

        Raises:
            ValueError: If the version is not registered; an update of no rows would otherwise
                pass silently.
        """
        result = self._connection.execute(
            update(model_versions)
            .where(model_versions.c.version == model_version.version.value)
            .values(status=model_version.status.value, activated_at=model_version.activated_at)
        )
        if result.rowcount == 0:
            raise ValueError(f"model version {model_version.version.value} is not registered")


def _model_version(row: Row[Any]) -> ModelVersion:
    return ModelVersion(
        version=ModelVersionId(row.version),
        status=ModelVersionStatus(row.status),
        created_at=row.created_at,
        activated_at=row.activated_at,
    )
