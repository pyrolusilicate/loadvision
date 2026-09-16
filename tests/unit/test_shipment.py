from datetime import UTC, datetime

import pytest

from loadvision.domain.shipment import Shipment, ShipmentId


@pytest.mark.unit
def test_shipment_keeps_id_and_time() -> None:
    created_at = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
    shipment = Shipment(ShipmentId("RP-1"), created_at)
    assert shipment.id == ShipmentId("RP-1")
    assert shipment.created_at == created_at


@pytest.mark.unit
def test_shipment_rejects_naive_time() -> None:
    with pytest.raises(ValueError, match="created_at"):
        Shipment(ShipmentId("RP-1"), datetime(2026, 9, 16, 12, 0))
