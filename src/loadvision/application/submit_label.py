"""Use case: confirm the load on the photo of an assessment, or mark the photo unusable."""

from dataclasses import dataclass
from uuid import UUID

from loadvision.application.ports import Clock, UnitOfWorkFactory
from loadvision.domain.assessment import Assessment, LoadPercent
from loadvision.domain.confirmed_label import ConfirmedLabel, LabelSource
from loadvision.errors import ErrorCode, LoadvisionError


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmitLabelRequest:
    """A person's verdict about the photo of an assessment.

    Attributes:
        assessment_id: Assessment whose photo is labeled.
        load_pct: Confirmed load, or None if the photo is unusable.
        source: Where the label comes from.
        author: Who confirmed it; blank means unknown.
    """

    assessment_id: UUID
    load_pct: LoadPercent | None
    source: LabelSource
    author: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmitLabelResult:
    """Outcome of a label submission.

    Attributes:
        label: The stored label of the photo.
        assessment: The assessment the label was submitted for.
    """

    label: ConfirmedLabel
    assessment: Assessment


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkReadable:
    """Cancels the verdict the model made about a photo it found unreadable.

    Only the verdict of the model is cancelled. A person's label stays whoever asks: the pages
    show this as one button on a photo the service refused, and nothing else may lose a label.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self, assessment_id: UUID) -> Assessment:
        """Removes the label of the photo when the model is the one that made it.

        Args:
            assessment_id: Assessment whose photo returns to assessment.

        Returns:
            The assessment, so the caller can open its page.

        Raises:
            LoadvisionError: `ASSESSMENT_NOT_FOUND` if the assessment is unknown.
        """
        with self.uow_factory() as uow:
            assessment = uow.assessments.get(assessment_id)
            if assessment is None:
                raise LoadvisionError(ErrorCode.ASSESSMENT_NOT_FOUND, str(assessment_id))
            label = uow.labels.get(assessment.image_sha256)
            if label is not None and label.source is LabelSource.MODEL:
                uow.labels.delete(assessment.image_sha256)
                uow.commit()
        return assessment


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmitLabel:
    """Stores a label for the photo of an assessment; a repeated submission replaces it.

    The label belongs to the photo, so labeling one assessment labels every assessment of the
    same photo.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock

    def __call__(self, request: SubmitLabelRequest) -> SubmitLabelResult:
        """Stores the label.

        Args:
            request: The verdict and the assessment it is about.

        Returns:
            The stored label and its assessment.

        Raises:
            LoadvisionError: `ASSESSMENT_NOT_FOUND` if the assessment is unknown.
        """
        author = (request.author or "").strip() or None
        now = self.clock()
        with self.uow_factory() as uow:
            assessment = uow.assessments.get(request.assessment_id)
            if assessment is None:
                raise LoadvisionError(ErrorCode.ASSESSMENT_NOT_FOUND, str(request.assessment_id))
            previous = uow.labels.get(assessment.image_sha256)
            if previous is None:
                label = ConfirmedLabel.first(
                    image_sha256=assessment.image_sha256,
                    load_pct=request.load_pct,
                    source=request.source,
                    author=author,
                    at=now,
                )
            else:
                label = previous.replaced(
                    load_pct=request.load_pct, source=request.source, author=author, at=now
                )
            uow.labels.save(label)
            uow.commit()
        return SubmitLabelResult(label=label, assessment=assessment)
