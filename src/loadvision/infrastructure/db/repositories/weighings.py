"""Postgres repository for weighings."""

from typing import Any
from uuid import UUID

from sqlalchemy import Connection, Row, select
from sqlalchemy.dialects.postgresql import insert

from loadvision.domain.cargo_weight import BodySize, CargoKind, WeightEstimate
from loadvision.domain.weighing import Weighing
from loadvision.infrastructure.db.tables import weighings


class PostgresWeighingRepository:
    """`WeighingRepository` on the connection of one unit of work."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def get(self, assessment_id: UUID) -> Weighing | None:
        row = self._connection.execute(
            select(weighings).where(weighings.c.assessment_id == assessment_id)
        ).one_or_none()
        return None if row is None else _weighing(row)

    def save(self, weighing: Weighing) -> None:
        values = {
            "assessment_id": weighing.assessment_id,
            "body": weighing.body,
            "length_m": weighing.size.length,
            "width_m": weighing.size.width,
            "height_m": weighing.size.height,
            "cargo_kind": weighing.kind.value,
            "kilograms": weighing.estimate.kilograms,
            "low_kg": weighing.estimate.low,
            "high_kg": weighing.estimate.high,
            "cubic_metres": weighing.estimate.cubic_metres,
            "author": weighing.author,
            "created_at": weighing.created_at,
        }
        statement = insert(weighings).values(values)
        # Weighing again replaces the whole row, as the port says.
        self._connection.execute(
            statement.on_conflict_do_update(
                index_elements=[weighings.c.assessment_id],
                set_={name: statement.excluded[name] for name in values if name != "assessment_id"},
            )
        )


def _weighing(row: Row[Any]) -> Weighing:
    return Weighing(
        assessment_id=row.assessment_id,
        body=row.body,
        size=BodySize(length=row.length_m, width=row.width_m, height=row.height_m),
        kind=CargoKind(row.cargo_kind),
        estimate=WeightEstimate(
            kilograms=row.kilograms,
            low=row.low_kg,
            high=row.high_kg,
            cubic_metres=row.cubic_metres,
        ),
        author=row.author,
        created_at=row.created_at,
    )
