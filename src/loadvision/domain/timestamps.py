"""Rules for timestamps: every one carries its offset and is written as text in one format."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Returns the current time in UTC.

    Use cases take the clock as a parameter and never call this themselves: a test moves time
    by passing its own clock. The real one is given to them where the service is assembled.
    """
    return datetime.now(UTC)


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


def format_utc(value: datetime) -> str:
    """Writes a timestamp as ISO 8601 text in UTC with a ``Z`` suffix.

    Args:
        value: Timezone-aware timestamp.

    Returns:
        Text such as ``2026-09-16T12:00:00Z``; microseconds are kept when present.

    Raises:
        ValueError: If `value` has no UTC offset.
    """
    ensure_aware(value, "value")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def format_utc_seconds(value: datetime) -> str:
    """Writes a timestamp like `format_utc`, but to the second: for tables and pages.

    Raises:
        ValueError: If `value` has no UTC offset.
    """
    return format_utc(value.replace(microsecond=0))
