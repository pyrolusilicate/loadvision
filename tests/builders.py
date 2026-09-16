"""Valid domain objects for tests; override only the fields a test is about."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import numpy as np

from loadvision.application.ports import ImageRecord
from loadvision.domain.assessment import Assessment, LoadPercent
from loadvision.domain.model_version import ModelVersion, ModelVersionId, ModelVersionStatus
from loadvision.domain.shipment import Shipment, ShipmentId
from loadvision.imaging.prepared_image import MediaType, PreparedImage

T0 = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def make_shipment(shipment_id: str = "RP-1", created_at: datetime = T0) -> Shipment:
    return Shipment(ShipmentId(shipment_id), created_at)


def make_image(raw: bytes = b"photo-a") -> PreparedImage:
    return PreparedImage(raw, MediaType.JPEG, np.zeros((4, 3, 3), dtype=np.uint8))


def make_image_record(raw: bytes = b"photo-a", created_at: datetime = T0) -> ImageRecord:
    return ImageRecord.from_prepared(make_image(raw), created_at)


def make_assessment(**overrides: Any) -> Assessment:
    values: dict[str, Any] = {
        "id": UUID(int=1),
        "shipment_id": ShipmentId("RP-1"),
        "image_sha256": make_image().sha256,
        "load_pct": LoadPercent(50.0),
        "model_version": ModelVersionId("fake-1"),
        "processing_ms": 10,
        "created_at": T0,
    }
    values.update(overrides)
    return Assessment(**values)


def make_model_version(
    version: str = "v1",
    *,
    status: ModelVersionStatus = ModelVersionStatus.CANDIDATE,
    created_at: datetime = T0,
    activated_at: datetime | None = None,
) -> ModelVersion:
    return ModelVersion(
        version=ModelVersionId(version),
        status=status,
        created_at=created_at,
        activated_at=activated_at,
    )
