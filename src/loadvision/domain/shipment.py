"""Shipment entity and ShipmentId value object."""

import re
from dataclasses import dataclass
from typing import Final

# No ^ and $ anchors: XSD patterns are anchored implicitly and the XSD repeats this exact string.
# Python checks it with fullmatch, which also rejects a trailing newline that "$" would let through.
SHIPMENT_ID_PATTERN: Final = "[A-Za-z0-9_-]{1,64}"

_SHIPMENT_ID_RE: Final = re.compile(SHIPMENT_ID_PATTERN)


class InvalidShipmentIdError(ValueError):
    """Shipment id does not match `SHIPMENT_ID_PATTERN`."""


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
            InvalidShipmentIdError: If `value` does not match `SHIPMENT_ID_PATTERN`.
        """
        if _SHIPMENT_ID_RE.fullmatch(self.value) is None:
            raise InvalidShipmentIdError(f"shipment id must match {SHIPMENT_ID_PATTERN}")
