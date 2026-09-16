"""Rules for timestamps kept in domain entities."""

from datetime import datetime


def ensure_aware(value: datetime, field: str) -> None:
    """Rejects a timestamp without a UTC offset.

    A naive timestamp is ambiguous once stored in a timestamptz column or rendered with a
    UTC suffix in XML, so every entity timestamp carries its offset.

    Args:
        value: Timestamp to check.
        field: Field name used in the error message.

    Raises:
        ValueError: If `value` has no UTC offset.
    """
    if value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
