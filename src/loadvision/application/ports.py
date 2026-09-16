"""Protocols the application layer depends on: repositories, storage, predictor."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Protocol, Self
from uuid import UUID

from loadvision.domain.assessment import Assessment, LoadPercent
from loadvision.domain.model_version import ModelVersion, ModelVersionId
from loadvision.domain.shipment import Shipment, ShipmentId
from loadvision.domain.timestamps import ensure_aware
from loadvision.imaging.prepared_image import MediaType, PreparedImage

# Use cases receive time and ids as functions: the domain never reads the clock itself and tests
# pass fixed values.
type Clock = Callable[[], datetime]
type IdFactory = Callable[[], UUID]


@dataclass(frozen=True, slots=True)
class Prediction:
    """Model output for one photo.

    Uncertainty and the OOD score are validated when the use case turns them into `Signals`,
    so their rules stay in one place.

    Attributes:
        load_pct: Predicted load.
        uncertainty: Spread of ensemble member predictions in percentage points, or None if the
            model does not estimate it.
        ood_score: Distance to the nearest training photos in embedding space, or None if the
            model does not compute it.
    """

    load_pct: LoadPercent
    uncertainty: float | None = None
    ood_score: float | None = None


class LoadPredictor(Protocol):
    """Model that estimates cargo load from photos."""

    @property
    def version(self) -> ModelVersionId:
        """Version of the model package that makes the predictions."""
        ...

    def predict(self, images: Sequence[PreparedImage]) -> list[Prediction]:
        """Predicts load for a batch of photos.

        Args:
            images: Decoded photos.

        Returns:
            One prediction per photo, in the same order.

        Raises:
            Exception: Any model failure fails the whole batch; the use case reports it as
                MODEL_INFERENCE_FAILED.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageRecord:
    """Stored metadata of a photo; the bytes themselves live in `ImageStorage`.

    Attributes:
        sha256: Lowercase hex SHA-256 of the photo bytes.
        media_type: Format detected from the file contents.
        width: Width in pixels.
        height: Height in pixels.
        size_bytes: Size of the original file.
        created_at: When the photo was first stored, timezone-aware.
    """

    sha256: str
    media_type: MediaType
    width: int
    height: int
    size_bytes: int
    created_at: datetime

    def __post_init__(self) -> None:
        """Validates the timestamp.

        Raises:
            ValueError: If `created_at` has no UTC offset.
        """
        ensure_aware(self.created_at, "created_at")

    @classmethod
    def from_prepared(cls, image: PreparedImage, created_at: datetime) -> Self:
        """Builds the record of a decoded photo, so the metadata always matches the photo.

        Args:
            image: Decoded photo.
            created_at: Storage time, timezone-aware.

        Returns:
            Record with the hash, format and size taken from the photo.
        """
        return cls(
            sha256=image.sha256,
            media_type=image.media_type,
            width=image.width,
            height=image.height,
            size_bytes=len(image.raw),
            created_at=created_at,
        )


class AssessmentAlreadyExistsError(Exception):
    """An assessment for the same shipment and photo is already stored."""


class ShipmentRepository(Protocol):
    """Shipments within the current unit of work."""

    def get(self, shipment_id: ShipmentId) -> Shipment | None:
        """Returns the shipment, or None if it is unknown."""
        ...

    def add(self, shipment: Shipment) -> None:
        """Stores the shipment unless one with the same id exists; an existing one is kept."""
        ...


class AssessmentRepository(Protocol):
    """Assessments within the current unit of work."""

    def add(self, assessment: Assessment) -> None:
        """Stores a new assessment.

        Raises:
            AssessmentAlreadyExistsError: If an assessment for the same shipment and photo is
                stored. The unit of work stays usable, so the caller can read the stored one.
        """
        ...

    def get(self, assessment_id: UUID) -> Assessment | None:
        """Returns the assessment, or None if it is unknown."""
        ...

    def find_by_shipment_and_image(
        self, shipment_id: ShipmentId, image_sha256: str
    ) -> Assessment | None:
        """Returns the assessment of this photo within the shipment, or None."""
        ...

    def find_by_image_and_model(
        self, image_sha256: str, model_version: ModelVersionId
    ) -> Assessment | None:
        """Returns any assessment of this photo made by this model version, or None.

        Lets the same photo sent with another shipment reuse the prediction.
        """
        ...

    def list_by_shipment(self, shipment_id: ShipmentId) -> list[Assessment]:
        """Returns all assessments of the shipment, newest `created_at` first."""
        ...


class ImageRepository(Protocol):
    """Photo metadata within the current unit of work."""

    def add(self, record: ImageRecord) -> None:
        """Stores the record unless one with the same hash exists; an existing one is kept."""
        ...

    def get(self, sha256: str) -> ImageRecord | None:
        """Returns the record, or None if the photo is unknown."""
        ...


class ModelVersionRepository(Protocol):
    """Model versions within the current unit of work."""

    def get(self, version: ModelVersionId) -> ModelVersion | None:
        """Returns the version, or None if it is not registered."""
        ...

    def get_active(self) -> ModelVersion | None:
        """Returns the active version, or None before the first activation."""
        ...

    def list_all(self) -> list[ModelVersion]:
        """Returns all versions, newest registration first."""
        ...

    def add(self, model_version: ModelVersion) -> None:
        """Registers a version whose tag is not registered yet."""
        ...

    def update(self, model_version: ModelVersion) -> None:
        """Saves the status and activation time of a registered version."""
        ...


class ImageStorage(Protocol):
    """Original photo bytes addressed by their SHA-256, outside the database transaction.

    Save the bytes before committing the database record: a failed commit then leaves only an
    unused file, while the reverse order could leave a record without its file.
    """

    def save(self, image: PreparedImage) -> None:
        """Stores the photo bytes; saving the same photo again changes nothing."""
        ...

    def load(self, sha256: str) -> bytes | None:
        """Returns the stored bytes, or None if the photo is unknown."""
        ...


class UnitOfWork(Protocol):
    """One database transaction for one use case.

    Use it as a context manager. Changes are kept only after `commit`; leaving the block
    without it, or with an exception, rolls them back.
    """

    @property
    def shipments(self) -> ShipmentRepository: ...

    @property
    def assessments(self) -> AssessmentRepository: ...

    @property
    def images(self) -> ImageRepository: ...

    @property
    def model_versions(self) -> ModelVersionRepository: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None:
        """Keeps the changes made in this unit of work."""
        ...


type UnitOfWorkFactory = Callable[[], UnitOfWork]
