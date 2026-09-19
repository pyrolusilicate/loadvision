"""Protocols the application layer depends on: repositories, storage, predictor."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self
from uuid import UUID

from loadvision.domain.assessment import SHA256_HEX_PATTERN, Assessment, LoadPercent, ReviewReason
from loadvision.domain.confirmed_label import ConfirmedLabel
from loadvision.domain.model_version import ModelVersion, ModelVersionId
from loadvision.domain.operator import Operator, OperatorSession
from loadvision.domain.shipment import Shipment, ShipmentId
from loadvision.domain.timestamps import ensure_aware
from loadvision.domain.training_job import TrainingJob, TrainingMode
from loadvision.errors import ErrorCode
from loadvision.imaging.prepared_image import MediaType, PreparedImage

_SHA256_HEX_RE: Final = re.compile(SHA256_HEX_PATTERN)

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
        review_reasons: Why a person should check the prediction, by the model's own
            measures; empty when none apply.
    """

    load_pct: LoadPercent
    uncertainty: float | None = None
    ood_score: float | None = None
    review_reasons: frozenset[ReviewReason] = field(default_factory=frozenset[ReviewReason])


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


class ReviewStatus(StrEnum):
    """What a person said about the photo of an assessment; the values go into page addresses."""

    WAITING = "review"
    CONFIRMED = "confirmed"
    UNUSABLE = "unusable"
    ALL = "all"


class AssessmentOrder(StrEnum):
    """Orders of the review queue; the values are what the page puts into its address."""

    NEWEST = "new"
    OLDEST = "old"
    FULLEST = "full"
    EMPTIEST = "empty"


class AssessmentRepository(Protocol):
    """Assessments within the current unit of work."""

    def add(self, assessment: Assessment) -> None:
        """Stores a new assessment.

        Raises:
            AssessmentAlreadyExistsError: If an assessment for the same shipment and photo is
                stored. The unit of work stays usable, so the caller can read the stored one.
        """
        ...

    def replace(self, assessment: Assessment) -> None:
        """Stores a newer estimate of a photo that this shipment already has.

        The shipment keeps one assessment per photo, so a photo assessed again by another
        model version replaces the stored row and keeps its id.

        Args:
            assessment: Assessment with the id of the stored one, the same shipment and the
                same photo.

        Raises:
            ValueError: If no assessment with this id is stored.
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

    def list_page(
        self,
        *,
        order: AssessmentOrder,
        offset: int,
        limit: int,
        status: ReviewStatus = ReviewStatus.ALL,
    ) -> list[Assessment]:
        """Returns one page of assessments in the given order.

        Rows with equal keys are ordered by time and then by id, so the order is total and
        two pages never share a row.

        Args:
            order: Which assessments come first.
            offset: Rows to skip.
            limit: Most rows to return.
            status: Which labels the photos must have; `WAITING` leaves the photos nobody
                labelled, the work of the review queue.
        """
        ...

    def count(self, *, status: ReviewStatus = ReviewStatus.ALL) -> int:
        """Returns how many assessments `list_page` pages through."""
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


class LabelRepository(Protocol):
    """Confirmed labels within the current unit of work; a photo has at most one label."""

    def get(self, image_sha256: str) -> ConfirmedLabel | None:
        """Returns the label of the photo, or None if it has none."""
        ...

    def save(self, label: ConfirmedLabel) -> None:
        """Stores the label, replacing an earlier label of the same photo.

        The photo record must already be stored in `ImageRepository`.
        """
        ...

    def list_for_training(self) -> list[ConfirmedLabel]:
        """Returns the labels with a load, sorted by photo hash; unusable photos are left out."""
        ...


class TrainingJobRepository(Protocol):
    """Training jobs within the current unit of work: the queue the worker takes from."""

    def add(self, job: TrainingJob) -> None:
        """Stores a new job."""
        ...

    def get(self, job_id: UUID) -> TrainingJob | None:
        """Returns the job, or None if it is unknown."""
        ...

    def find_unfinished(self) -> TrainingJob | None:
        """Returns a job that is queued or running, or None when every job has ended."""
        ...

    def claim_next(self) -> TrainingJob | None:
        """Takes the oldest queued job for this worker.

        The job is locked until the unit of work ends, so no other worker sees it. Returns
        None when nothing waits or another job is already running: retraining runs one at a
        time, and the database holds that rule as well.
        """
        ...

    def list_recent(self, limit: int, offset: int = 0) -> list[TrainingJob]:
        """Returns the newest jobs first, at most `limit` of them after skipping `offset`."""
        ...

    def count(self) -> int:
        """Returns how many jobs there are."""
        ...

    def update(self, job: TrainingJob) -> None:
        """Saves the status and the outcome of a stored job.

        Raises:
            ValueError: If the job was never stored.
        """
        ...


class OperatorRepository(Protocol):
    """Operator accounts within the current unit of work."""

    def get(self, login: str) -> Operator | None:
        """Returns the account, or None if there is none with this login."""
        ...

    def add(self, operator: Operator) -> None:
        """Stores a new account whose login is not taken."""
        ...

    def update(self, operator: Operator) -> None:
        """Saves a stored account.

        Raises:
            ValueError: If the account was never stored.
        """
        ...

    def list_all(self) -> list[Operator]:
        """Returns every account, by login."""
        ...


class OperatorSessionRepository(Protocol):
    """Sign-in sessions within the current unit of work."""

    def add(self, session: OperatorSession) -> None:
        """Stores a new session."""
        ...

    def get(self, token_hash: str) -> OperatorSession | None:
        """Returns the session with this token hash, or None."""
        ...

    def delete(self, token_hash: str) -> None:
        """Ends one session; ending an unknown one changes nothing."""
        ...

    def delete_for(self, login: str) -> None:
        """Ends every session of an operator."""
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

    @property
    def labels(self) -> LabelRepository: ...

    @property
    def training_jobs(self) -> TrainingJobRepository: ...

    @property
    def operators(self) -> OperatorRepository: ...

    @property
    def operator_sessions(self) -> OperatorSessionRepository: ...

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


@dataclass(frozen=True, slots=True, kw_only=True)
class LabeledImage:
    """A confirmed label handed to training.

    Attributes:
        image_sha256: Lowercase hex SHA-256 of the photo; training reads the bytes from
            `ImageStorage`.
        load_pct: Load confirmed by a person.
    """

    image_sha256: str
    load_pct: LoadPercent

    def __post_init__(self) -> None:
        """Validates the photo hash.

        Raises:
            ValueError: If `image_sha256` is not 64 lowercase hex characters.
        """
        if _SHA256_HEX_RE.fullmatch(self.image_sha256) is None:
            raise ValueError("image_sha256 must be 64 lowercase hex characters")

    @classmethod
    def from_label(cls, label: ConfirmedLabel) -> Self:
        """Turns a confirmed label into a training example.

        Raises:
            ValueError: If the label marks the photo as unusable.
        """
        if label.load_pct is None:
            raise ValueError(f"photo {label.image_sha256} is unusable and has no load")
        return cls(image_sha256=label.image_sha256, load_pct=label.load_pct)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingRequest:
    """What the worker asks training to do.

    The base dataset is not listed here: training reads it from the dataset manifest, so the
    request stays small whatever the dataset size.

    Attributes:
        mode: How much of the pipeline to repeat.
        labels: Confirmed labels added on top of the base dataset; empty means retraining on
            the base dataset alone.
        base_version: Active model version the candidate is compared against, or None before
            the first activation.
    """

    mode: TrainingMode
    labels: tuple[LabeledImage, ...] = ()
    base_version: ModelVersionId | None = None

    def __post_init__(self) -> None:
        """Validates the labels.

        Raises:
            ValueError: If the same photo appears twice; two labels of one photo would
                contradict each other.
        """
        if len({label.image_sha256 for label in self.labels}) != len(self.labels):
            raise ValueError("labels must not repeat a photo")


class TrainingStatus(StrEnum):
    """Whether training produced a package."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingOutcome:
    """What training reports back to the worker.

    A rejected candidate is a success: training worked and the gate did its job. Only a run
    that produced no package is a failure.

    Attributes:
        run_id: Id of the run; its folder holds the events, the summary and the metrics.
        status: Whether a package was produced.
        package_path: Folder of the candidate package, or None when the run failed.
        gate_passed: Whether the candidate may replace the active version.
        gate_report: Numbers behind the gate decision, ready for JSON and for the API.
        error_code: Code of the failure, or None when the run succeeded.
        error_detail: Context of the failure for logs, never shown to clients.
    """

    run_id: str
    status: TrainingStatus
    package_path: Path | None = None
    gate_passed: bool = False
    gate_report: Mapping[str, object] = field(default_factory=dict[str, object])
    error_code: ErrorCode | None = None
    error_detail: str | None = None

    def __post_init__(self) -> None:
        """Validates that the fields agree with the status.

        Raises:
            ValueError: If a successful run has no package or carries an error code, if a
                failed run carries a package or no error code, or if a failed run reports a
                passed gate.
        """
        succeeded = self.status is TrainingStatus.SUCCEEDED
        if succeeded and (self.package_path is None or self.error_code is not None):
            raise ValueError("a succeeded run has a package and no error code")
        if not succeeded and (self.package_path is not None or self.error_code is None):
            raise ValueError("a failed run has an error code and no package")
        if not succeeded and self.gate_passed:
            raise ValueError("a failed run cannot pass the gate")


class TrainingRunner(Protocol):
    """Retraining on confirmed labels, called by the worker and by the CLI."""

    def run(self, request: TrainingRequest) -> TrainingOutcome:
        """Trains a candidate and compares it with the active version.

        Expected failures, such as too few labels or a stage postcondition, come back in the
        outcome, so the worker records them in the job instead of catching exceptions.

        Args:
            request: What to train and what to compare against.

        Returns:
            Outcome with the run id, the package and the gate decision.
        """
        ...
