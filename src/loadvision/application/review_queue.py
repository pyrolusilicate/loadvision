"""Use case: the assessments waiting for a person to confirm the load on their photo."""

from dataclasses import dataclass
from typing import Final

from loadvision.application.labelled_assessments import LabeledAssessment, with_labels
from loadvision.application.ports import UnitOfWorkFactory

DEFAULT_QUEUE_SIZE: Final = 50


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewQueue:
    """Reads the newest assessments, by default only those whose photo has no label yet.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(
        self, limit: int = DEFAULT_QUEUE_SIZE, *, unlabeled_only: bool = True
    ) -> tuple[LabeledAssessment, ...]:
        """Returns the queue, newest first.

        Args:
            limit: Most assessments to return.
            unlabeled_only: Show only the work left; False shows labelled ones too.
        """
        with self.uow_factory() as uow:
            recent = uow.assessments.list_recent(limit, unlabeled_only=unlabeled_only)
            return with_labels(uow, recent)
