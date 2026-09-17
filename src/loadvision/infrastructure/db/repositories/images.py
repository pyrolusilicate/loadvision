"""Postgres repository for photo metadata."""

from sqlalchemy import Connection, select
from sqlalchemy.dialects.postgresql import insert

from loadvision.application.ports import ImageRecord
from loadvision.imaging.prepared_image import MediaType
from loadvision.infrastructure.db.tables import images


class PostgresImageRepository:
    """`ImageRepository` on the connection of one unit of work."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def add(self, record: ImageRecord) -> None:
        self._connection.execute(
            insert(images)
            .values(
                sha256=record.sha256,
                media_type=record.media_type.value,
                width=record.width,
                height=record.height,
                size_bytes=record.size_bytes,
                created_at=record.created_at,
            )
            .on_conflict_do_nothing(index_elements=[images.c.sha256])
        )

    def get(self, sha256: str) -> ImageRecord | None:
        row = self._connection.execute(
            select(images).where(images.c.sha256 == sha256)
        ).one_or_none()
        if row is None:
            return None
        return ImageRecord(
            sha256=row.sha256,
            media_type=MediaType(row.media_type),
            width=row.width,
            height=row.height,
            size_bytes=row.size_bytes,
            created_at=row.created_at,
        )
