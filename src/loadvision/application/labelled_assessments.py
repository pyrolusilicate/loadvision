"""Assessments shown together with the label of their photo."""

from collections.abc import Sequence
from dataclasses import dataclass

from loadvision.application.ports import UnitOfWork
from loadvision.domain.assessment import Assessment
from loadvision.domain.confirmed_label import ConfirmedLabel


@dataclass(frozen=True, slots=True, kw_only=True)
class LabeledAssessment:
    """An assessment together with the label of its photo.

    Attributes:
        assessment: The assessment.
        label: Label of the assessed photo, or None if nobody confirmed it.
    """

    assessment: Assessment
    label: ConfirmedLabel | None


def with_labels(
    uow: UnitOfWork, assessments: Sequence[Assessment]
) -> tuple[LabeledAssessment, ...]:
    """Pairs every assessment with the label of its photo, reading each photo's label once.

    Args:
        uow: Open unit of work.
        assessments: Assessments, in the order they are to be shown.

    Returns:
        The pairs, in the same order.
    """
    labels = {sha256: uow.labels.get(sha256) for sha256 in {a.image_sha256 for a in assessments}}
    return tuple(
        LabeledAssessment(assessment=assessment, label=labels[assessment.image_sha256])
        for assessment in assessments
    )
