"""Postgres repository for shipments."""

from sqlalchemy import Connection, select
from sqlalchemy.dialects.postgresql import insert

from loadvision.domain.shipment import Shipment, ShipmentId
from loadvision.infrastructure.db.tables import shipments


class PostgresShipmentRepository:
    """`ShipmentRepository` on the connection of one unit of work."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def get(self, shipment_id: ShipmentId) -> Shipment | None:
        row = self._connection.execute(
            select(shipments).where(shipments.c.id == shipment_id.value)
        ).one_or_none()
        return None if row is None else Shipment(ShipmentId(row.id), row.created_at)

    def add(self, shipment: Shipment) -> None:
        # A parallel transaction adding the same shipment makes this insert wait for its commit
        # and then do nothing, so the first stored shipment stays.
        self._connection.execute(
            insert(shipments)
            .values(id=shipment.id.value, created_at=shipment.created_at)
            .on_conflict_do_nothing(index_elements=[shipments.c.id])
        )
