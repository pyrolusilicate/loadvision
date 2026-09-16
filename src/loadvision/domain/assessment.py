"""Assessment entity with its load percent, trust signals and review reasons."""

import math
import re
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import StrEnum
from typing import Final
from uuid import UUID

from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.shipment import ShipmentId
from loadvision.domain.timestamps import ensure_aware

_SHA256_HEX_RE: Final = re.compile("[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class LoadPercent:
    """Cargo load of a truck body in percent.

    The raw value is kept: rounding belongs to presentation, so stored predictions stay exact.

    Attributes:
        value: Finite number from 0 to 100.
    """

    value: float

    def __post_init__(self) -> None:
        """Validates the range.

        Raises:
            ValueError: If `value` is NaN, infinite or outside [0, 100]. The meaning depends on
                the caller, so the caller picks the error code: a model output becomes
                MODEL_INFERENCE_FAILED.
        """
        # isfinite goes first: NaN fails every comparison, so "< 0 or > 100" alone lets it in.
        if not math.isfinite(self.value) or self.value < 0 or self.value > 100:
            raise ValueError(f"load percent must be finite and within [0, 100], got {self.value!r}")


class ReviewReason(StrEnum):
    """Why an assessment needs a manual check."""

    LOW_CONFIDENCE = "low_confidence"
    UNUSUAL_SCENE = "unusual_scene"
    TOO_DARK = "too_dark"
    BLURRY = "blurry"


@dataclass(frozen=True, slots=True, kw_only=True)
class Signals:
    """Trust signals of one assessment; a signal is None until it is computed.

    Attributes:
        uncertainty: Spread of ensemble member predictions, in percentage points.
        ood_score: Distance from the photo to the nearest training photos in embedding space.
        brightness: Mean brightness of the photo after EXIF rotation.
        sharpness: Sharpness of the downscaled photo.
    """

    uncertainty: float | None = None
    ood_score: float | None = None
    brightness: float | None = None
    sharpness: float | None = None

    def __post_init__(self) -> None:
        """Validates the signals that are present.

        Raises:
            ValueError: If a present signal is NaN, infinite or negative.
        """
        for signal in fields(self):
            value: float | None = getattr(self, signal.name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{signal.name} must be finite and non-negative, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Assessment:
    """Load assessment of one photo attached to a shipment.

    Attributes:
        id: Assessment identifier.
        shipment_id: Shipment the photo belongs to.
        image_sha256: Lowercase hex SHA-256 of the original photo bytes.
        load_pct: Predicted load.
        model_version: Model version that produced the prediction.
        processing_ms: Time spent on the request, in milliseconds.
        created_at: When the assessment was made, timezone-aware.
        signals: Trust signals known at assessment time.
        review_reasons: Reasons that require a manual check; empty when none apply.
        request_id: Id of the API request, or None for assessments made outside the API.
    """

    id: UUID
    shipment_id: ShipmentId
    image_sha256: str
    load_pct: LoadPercent
    model_version: ModelVersionId
    processing_ms: int
    created_at: datetime
    signals: Signals = field(default_factory=Signals)
    review_reasons: frozenset[ReviewReason] = field(default_factory=frozenset[ReviewReason])
    request_id: str | None = None

    def __post_init__(self) -> None:
        """Validates the photo hash, the processing time and the timestamp.

        Raises:
            ValueError: If `image_sha256` is not 64 lowercase hex characters, `processing_ms`
                is negative or `created_at` has no UTC offset.
        """
        if _SHA256_HEX_RE.fullmatch(self.image_sha256) is None:
            raise ValueError("image_sha256 must be 64 lowercase hex characters")
        if self.processing_ms < 0:
            raise ValueError(f"processing_ms must be non-negative, got {self.processing_ms}")
        ensure_aware(self.created_at, "created_at")

    @property
    def needs_review(self) -> bool:
        """Whether a person should check the assessment; derived from `review_reasons`."""
        return bool(self.review_reasons)
