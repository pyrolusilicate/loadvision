"""Use case: get one assessment with its photo metadata and label."""

from dataclasses import dataclass
from uuid import UUID

from loadvision.application.ports import ImageRecord, UnitOfWorkFactory
from loadvision.domain.assessment import Assessment
from loadvision.domain.confirmed_label import ConfirmedLabel
from loadvision.domain.weighing import Weighing
from loadvision.errors import ErrorCode, LoadvisionError


@dataclass(frozen=True, slots=True, kw_only=True)
class AssessmentDetails:
    """An assessment with what its response shows.

    Attributes:
        assessment: The assessment.
        image: Stored metadata of the assessed photo.
        label: Label of the photo, or None if nobody confirmed it.
        weighing: What an operator weighed this assessment with, or None if nobody did. It
            belongs to the assessment and not to the photo: the same photo in another shipment
            may sit in another body, and the body is what sets the volume.
    """

    assessment: Assessment
    image: ImageRecord
    label: ConfirmedLabel | None
    weighing: Weighing | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GetAssessment:
    """Reads an assessment by id.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self, assessment_id: UUID) -> AssessmentDetails:
        """Returns the assessment.

        Raises:
            LoadvisionError: `ASSESSMENT_NOT_FOUND` if the assessment is unknown;
                `INTERNAL_ERROR` if its photo record is missing.
        """
        with self.uow_factory() as uow:
            assessment = uow.assessments.get(assessment_id)
            if assessment is None:
                raise LoadvisionError(ErrorCode.ASSESSMENT_NOT_FOUND, str(assessment_id))
            image = uow.images.get(assessment.image_sha256)
            if image is None:
                raise LoadvisionError(
                    ErrorCode.INTERNAL_ERROR, f"assessment {assessment.id} has no photo record"
                )
            label = uow.labels.get(assessment.image_sha256)
            weighing = uow.weighings.get(assessment_id)
        return AssessmentDetails(assessment=assessment, image=image, label=label, weighing=weighing)
