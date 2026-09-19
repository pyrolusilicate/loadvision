"""Use case: the assessments waiting for a person to confirm the load on their photo."""

from dataclasses import dataclass

from loadvision.application.labelled_assessments import LabeledAssessment, with_labels
from loadvision.application.paging import Page, page_number
from loadvision.application.ports import AssessmentOrder, ReviewStatus, UnitOfWorkFactory


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewQueue:
    """Reads one page of assessments, by default those whose photo nobody labelled yet.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(
        self,
        *,
        page: int,
        size: int,
        order: AssessmentOrder = AssessmentOrder.NEWEST,
        status: ReviewStatus = ReviewStatus.WAITING,
    ) -> Page[LabeledAssessment]:
        """Returns one page of the queue.

        Args:
            page: Page to show, from 1; a number past the end shows the last page.
            size: Assessments on a page.
            order: Which assessments come first.
            status: Which labels the photos must have.
        """
        with self.uow_factory() as uow:
            total = uow.assessments.count(status=status)
            number = page_number(page, total, size)
            rows = uow.assessments.list_page(
                order=order,
                offset=(number - 1) * size,
                limit=size,
                status=status,
            )
            return Page(items=with_labels(uow, rows), number=number, size=size, total=total)
