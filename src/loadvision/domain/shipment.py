"""Shipment entity and ShipmentId value object."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from loadvision.domain.timestamps import ensure_aware
from loadvision.errors import ErrorCode, LoadvisionError

# No ^ and $ anchors: XSD patterns are anchored implicitly and the XSD repeats this exact string.
# Python checks it with fullmatch, which also rejects a trailing newline that "$" would let through.
SHIPMENT_ID_PATTERN: Final = "[A-Za-z0-9_-]{1,64}"

_SHIPMENT_ID_RE: Final = re.compile(SHIPMENT_ID_PATTERN)


@dataclass(frozen=True, slots=True)
class ShipmentId:
    """Identifier of a shipment in the customer's internal systems.

    Attributes:
        value: 1 to 64 Latin letters, digits, underscores or hyphens.
    """

    value: str

    def __post_init__(self) -> None:
        """Validates the format.

        Raises:
            LoadvisionError: With code `INVALID_SHIPMENT_ID` if `value` does not match
                `SHIPMENT_ID_PATTERN`.
        """
        if _SHIPMENT_ID_RE.fullmatch(self.value) is None:
            raise LoadvisionError(ErrorCode.INVALID_SHIPMENT_ID)


@dataclass(frozen=True, slots=True)
class Shipment:
    """Shipment known to the service; its assessments are stored separately.

    Attributes:
        id: Shipment identifier.
        created_at: When the first assessment created the shipment, timezone-aware.
    """

    id: ShipmentId
    created_at: datetime

    def __post_init__(self) -> None:
        """Validates the timestamp.

        Raises:
            ValueError: If `created_at` has no UTC offset.
        """
        ensure_aware(self.created_at, "created_at")
