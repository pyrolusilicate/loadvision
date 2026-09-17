"""Confirmed label: a person's verdict about the load on a photo."""

import re
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Final, Self

from loadvision.domain.assessment import SHA256_HEX_PATTERN, LoadPercent
from loadvision.domain.timestamps import ensure_aware

_SHA256_HEX_RE: Final = re.compile(SHA256_HEX_PATTERN)


class LabelSource(StrEnum):
    """Where a label came from."""

    API = "api"
    UI = "ui"
    IMPORT = "import"


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfirmedLabel:
    """A person's verdict about one photo: the confirmed load, or that the photo is unusable.

    A label belongs to the photo, not to an assessment, so every assessment of the same photo
    shows it. An unusable photo, such as a covered lens or a black frame, has no load and never
    goes into training.

    Attributes:
        image_sha256: Lowercase hex SHA-256 of the photo.
        load_pct: Confirmed load, or None for an unusable photo.
        source: Where the label came from.
        author: Who confirmed it, or None if unknown.
        created_at: When the photo was first labeled, timezone-aware.
        updated_at: When the label was last replaced, timezone-aware.
    """

    image_sha256: str
    load_pct: LoadPercent | None
    source: LabelSource
    author: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        """Validates the hash, the author and the timestamps.

        Raises:
            ValueError: If `image_sha256` is not 64 lowercase hex characters, `author` is blank,
                a timestamp has no UTC offset or `updated_at` is before `created_at`.
        """
        if _SHA256_HEX_RE.fullmatch(self.image_sha256) is None:
            raise ValueError("image_sha256 must be 64 lowercase hex characters")
        if self.author is not None and not self.author.strip():
            raise ValueError("author must not be blank; use None for an unknown author")
        ensure_aware(self.created_at, "created_at")
        ensure_aware(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be before created_at")

    @property
    def unusable(self) -> bool:
        """Whether the photo was judged unusable and has no load."""
        return self.load_pct is None

    @classmethod
    def first(
        cls,
        *,
        image_sha256: str,
        load_pct: LoadPercent | None,
        source: LabelSource,
        author: str | None,
        at: datetime,
    ) -> Self:
        """Creates the first label of a photo.

        Args:
            image_sha256: Photo hash.
            load_pct: Confirmed load, or None for an unusable photo.
            source: Where the label came from.
            author: Who confirmed it.
            at: Time of the confirmation, timezone-aware.
        """
        return cls(
            image_sha256=image_sha256,
            load_pct=load_pct,
            source=source,
            author=author,
            created_at=at,
            updated_at=at,
        )

    def replaced(
        self,
        *,
        load_pct: LoadPercent | None,
        source: LabelSource,
        author: str | None,
        at: datetime,
    ) -> Self:
        """Returns the label after a new confirmation of the same photo; the creation time stays.

        Args:
            load_pct: Confirmed load, or None for an unusable photo.
            source: Where the new label came from.
            author: Who confirmed it.
            at: Time of the new confirmation, timezone-aware.
        """
        return replace(self, load_pct=load_pct, source=source, author=author, updated_at=at)
