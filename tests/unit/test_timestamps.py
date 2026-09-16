from datetime import UTC, datetime, timedelta, timezone

import pytest

from loadvision.domain.timestamps import ensure_aware, format_utc


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
        datetime(2026, 9, 16, 15, 0, tzinfo=timezone(timedelta(hours=3))),
    ],
)
def test_accepts_timestamp_with_offset(value: datetime) -> None:
    ensure_aware(value, "created_at")


@pytest.mark.unit
def test_rejects_naive_timestamp_and_names_the_field() -> None:
    with pytest.raises(ValueError, match="created_at"):
        ensure_aware(datetime(2026, 9, 16, 12, 0), "created_at")


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
        datetime(2026, 9, 16, 15, 0, tzinfo=timezone(timedelta(hours=3))),
    ],
)
def test_format_utc_writes_utc_with_z_suffix(value: datetime) -> None:
    assert format_utc(value) == "2026-09-16T12:00:00Z"


@pytest.mark.unit
def test_format_utc_keeps_microseconds() -> None:
    value = datetime(2026, 9, 16, 12, 0, 0, 123, tzinfo=UTC)
    assert format_utc(value) == "2026-09-16T12:00:00.000123Z"


@pytest.mark.unit
def test_format_utc_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        format_utc(datetime(2026, 9, 16, 12, 0))
