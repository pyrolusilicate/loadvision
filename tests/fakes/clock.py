"""Deterministic replacements for the clock and the id factory given to use cases."""

from datetime import datetime, timedelta
from uuid import UUID

from loadvision.domain.timestamps import ensure_aware


class FixedClock:
    """Clock that returns the same time until a test moves it."""

    def __init__(self, now: datetime) -> None:
        ensure_aware(now, "now")
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


class SequentialIds:
    """Id factory that returns UUID(int=1), UUID(int=2) and so on."""

    def __init__(self) -> None:
        self._last = 0

    def __call__(self) -> UUID:
        self._last += 1
        return UUID(int=self._last)
