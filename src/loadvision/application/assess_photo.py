"""Use case: assess a cargo photo and attach the result to a shipment."""

import time
from collections.abc import Callable
from dataclasses import dataclass

from loadvision.application.ports import (
    AssessmentAlreadyExistsError,
    Clock,
    IdFactory,
    ImageRecord,
    ImageStorage,
    LoadPredictor,
    UnitOfWork,
    UnitOfWorkFactory,
)
from loadvision.domain.assessment import Assessment, LoadPercent, ReviewReason, Signals
from loadvision.domain.shipment import Shipment, ShipmentId
from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.imaging.prepared_image import PreparedImage
from loadvision.imaging.validator import ImageLimits, prepare_image


@dataclass(frozen=True, slots=True, kw_only=True)
class AssessPhotoRequest:
    """A photo sent for assessment.

    Attributes:
        shipment_id: Shipment number as received; validated by the use case.
        image: Photo file bytes as received.
        request_id: Id of the API request, or None outside the API.
    """

    shipment_id: str
    image: bytes
    request_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AssessPhotoResult:
    """Outcome of an assessment request.

    Attributes:
        assessment: The new assessment, or the stored one for a repeated request.
        image: Stored metadata of the photo, for the response.
        reused: True if this photo had already been assessed in this shipment and nothing was
            computed again.
    """

    assessment: Assessment
    image: ImageRecord
    reused: bool


@dataclass(frozen=True, slots=True)
class _Estimate:
    load_pct: LoadPercent
    signals: Signals
    review_reasons: frozenset[ReviewReason]


@dataclass(frozen=True, slots=True, kw_only=True)
class AssessPhoto:
    """Assesses a photo for a shipment.

    A repeated request never computes twice: the same photo in the same shipment returns the
    stored assessment, and the same photo in another shipment reuses the prediction any earlier
    assessment of this photo got from the same model version.

    Attributes:
        uow_factory: Opens database transactions.
        storage: Keeps the original photo bytes.
        limits: Size and format limits of uploaded photos.
        clock: Current time, timezone-aware.
        new_id: New assessment ids.
        timer: Monotonic seconds, for the processing time.
    """

    uow_factory: UnitOfWorkFactory
    storage: ImageStorage
    limits: ImageLimits
    clock: Clock
    new_id: IdFactory
    timer: Callable[[], float] = time.perf_counter

    def __call__(self, request: AssessPhotoRequest, predictor: LoadPredictor) -> AssessPhotoResult:
        """Assesses the photo, or returns the stored result of the same request.

        Args:
            request: The photo and its shipment.
            predictor: Model to predict with. The caller passes one snapshot of the active
                model, so the version stored with the assessment is the one that predicted.

        Returns:
            The assessment with its photo metadata and whether it was reused.

        Raises:
            LoadvisionError: `INVALID_SHIPMENT_ID` for a malformed shipment number;
                `PAYLOAD_TOO_LARGE`, `UNSUPPORTED_FORMAT` or `CORRUPTED_IMAGE` for the photo;
                `MODEL_INFERENCE_FAILED` if the model fails or returns an invalid prediction.
        """
        started = self.timer()
        shipment_id = ShipmentId(request.shipment_id)
        image = prepare_image(request.image, self.limits)

        # Reads only: the transaction is closed before the model runs, so a slow prediction
        # holds no database connection.
        with self.uow_factory() as uow:
            stored = uow.assessments.find_by_shipment_and_image(shipment_id, image.sha256)
            if stored is not None:
                return self._reused(uow, stored)
            cached = uow.assessments.find_by_image_and_model(image.sha256, predictor.version)

        estimate = self._predict(image, predictor) if cached is None else _estimate_of(cached)
        now = self.clock()
        assessment = Assessment(
            id=self.new_id(),
            shipment_id=shipment_id,
            image_sha256=image.sha256,
            load_pct=estimate.load_pct,
            model_version=predictor.version,
            processing_ms=round((self.timer() - started) * 1000),
            created_at=now,
            signals=estimate.signals,
            review_reasons=estimate.review_reasons,
            request_id=request.request_id,
        )

        # The file goes first: a failed commit leaves an unused file, never a record whose
        # photo is missing.
        self.storage.save(image)
        try:
            with self.uow_factory() as uow:
                uow.shipments.add(Shipment(shipment_id, now))
                uow.images.add(ImageRecord.from_prepared(image, now))
                uow.assessments.add(assessment)
                record = _image_record(uow, assessment)
                uow.commit()
        except AssessmentAlreadyExistsError as error:
            # A parallel request with the same photo and shipment committed first.
            with self.uow_factory() as uow:
                stored = uow.assessments.find_by_shipment_and_image(shipment_id, image.sha256)
                if stored is None:
                    raise LoadvisionError(
                        ErrorCode.INTERNAL_ERROR, "duplicate assessment is not readable"
                    ) from error
                return self._reused(uow, stored)
        return AssessPhotoResult(assessment=assessment, image=record, reused=False)

    @staticmethod
    def _reused(uow: UnitOfWork, stored: Assessment) -> AssessPhotoResult:
        return AssessPhotoResult(assessment=stored, image=_image_record(uow, stored), reused=True)

    @staticmethod
    def _predict(image: PreparedImage, predictor: LoadPredictor) -> _Estimate:
        # The port promises nothing about how a model fails, so every failure, including a
        # prediction the domain rejects, is reported as one code.
        try:
            predictions = predictor.predict([image])
            if len(predictions) != 1:
                raise ValueError(f"expected one prediction, got {len(predictions)}")
            [prediction] = predictions
            signals = Signals(uncertainty=prediction.uncertainty, ood_score=prediction.ood_score)
        except Exception as error:
            raise LoadvisionError(
                ErrorCode.MODEL_INFERENCE_FAILED, f"{type(error).__name__}: {error}"
            ) from error
        # Brightness, sharpness (task 7.7) and the review policy are not implemented yet.
        return _Estimate(load_pct=prediction.load_pct, signals=signals, review_reasons=frozenset())


def _estimate_of(assessment: Assessment) -> _Estimate:
    return _Estimate(
        load_pct=assessment.load_pct,
        signals=assessment.signals,
        review_reasons=assessment.review_reasons,
    )


def _image_record(uow: UnitOfWork, assessment: Assessment) -> ImageRecord:
    record = uow.images.get(assessment.image_sha256)
    if record is None:
        raise LoadvisionError(
            ErrorCode.INTERNAL_ERROR, f"assessment {assessment.id} has no photo record"
        )
    return record
