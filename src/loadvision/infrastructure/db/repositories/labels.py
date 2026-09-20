"""Postgres repository for confirmed labels."""

from typing import Any

from sqlalchemy import Connection, Row, delete, select
from sqlalchemy.dialects.postgresql import insert

from loadvision.domain.assessment import LoadPercent
from loadvision.domain.confirmed_label import ConfirmedLabel, LabelSource
from loadvision.infrastructure.db.tables import confirmed_labels


class PostgresLabelRepository:
    """`LabelRepository` on the connection of one unit of work."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def get(self, image_sha256: str) -> ConfirmedLabel | None:
        row = self._connection.execute(
            select(confirmed_labels).where(confirmed_labels.c.image_sha256 == image_sha256)
        ).one_or_none()
        return None if row is None else _label(row)

    def save(self, label: ConfirmedLabel) -> None:
        values = {
            "image_sha256": label.image_sha256,
            "load_pct": None if label.load_pct is None else label.load_pct.value,
            "unusable": label.unusable,
            "source": label.source.value,
            "author": label.author,
            "created_at": label.created_at,
            "updated_at": label.updated_at,
        }
        statement = insert(confirmed_labels).values(values)
        # The whole row is replaced, as the port says; the domain already kept `created_at`.
        self._connection.execute(
            statement.on_conflict_do_update(
                index_elements=[confirmed_labels.c.image_sha256],
                set_={name: statement.excluded[name] for name in values if name != "image_sha256"},
            )
        )

    def delete(self, image_sha256: str) -> None:
        self._connection.execute(
            delete(confirmed_labels).where(confirmed_labels.c.image_sha256 == image_sha256)
        )

    def list_for_training(self) -> list[ConfirmedLabel]:
        rows = self._connection.execute(
            select(confirmed_labels)
            .where(confirmed_labels.c.unusable.is_(False))
            .order_by(confirmed_labels.c.image_sha256)
        )
        return [_label(row) for row in rows]


def _label(row: Row[Any]) -> ConfirmedLabel:
    return ConfirmedLabel(
        image_sha256=row.image_sha256,
        load_pct=None if row.load_pct is None else LoadPercent(row.load_pct),
        source=LabelSource(row.source),
        author=row.author,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
