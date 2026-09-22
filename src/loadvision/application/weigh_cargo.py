"""Use case: keep what an operator weighed an assessment with, and what came out."""

from dataclasses import dataclass
from uuid import UUID

from loadvision.application.ports import Clock, UnitOfWorkFactory
from loadvision.domain.cargo_weight import (
    DEFAULT_STACK_SHARE,
    BodySize,
    CargoKind,
    density_of,
    estimate_weight,
)
from loadvision.domain.weighing import Weighing
from loadvision.errors import ErrorCode, LoadvisionError


@dataclass(frozen=True, slots=True, kw_only=True)
class WeighCargoRequest:
    """What an operator chose to weigh an assessment with.

    Attributes:
        assessment_id: Assessment being weighed.
        body: Key of the body that was chosen, for the page to show it selected again.
        size: Inner size of that body.
        kind: Kind of cargo the density is taken for.
        author: Who weighed it; None if unknown.
    """

    assessment_id: UUID
    body: str
    size: BodySize
    kind: CargoKind
    author: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WeighCargo:
    """Weighs the cargo of an assessment and stores the result beside it.

    The arithmetic lives in the domain and the load comes from the assessment, so the operator
    supplies only what a photo cannot tell: the body and the kind of cargo. Weighing the same
    assessment again replaces the stored numbers.

    Attributes:
        uow_factory: Opens database transactions.
        clock: Current time, timezone-aware.
    """

    uow_factory: UnitOfWorkFactory
    clock: Clock

    def __call__(self, request: WeighCargoRequest) -> Weighing:
        """Weighs and stores.

        Args:
            request: The body, the cargo and the assessment they are about.

        Returns:
            The stored weighing.

        Raises:
            LoadvisionError: `ASSESSMENT_NOT_FOUND` if the assessment is unknown.
        """
        with self.uow_factory() as uow:
            assessment = uow.assessments.get(request.assessment_id)
            if assessment is None:
                raise LoadvisionError(ErrorCode.ASSESSMENT_NOT_FOUND, str(request.assessment_id))
            weighing = Weighing.of(
                assessment_id=request.assessment_id,
                body=request.body,
                size=request.size,
                kind=request.kind,
                estimate=estimate_weight(
                    assessment.load_pct.value,
                    body=request.size,
                    density=density_of(request.kind),
                    stack_share=DEFAULT_STACK_SHARE,
                ),
                author=request.author,
                at=self.clock(),
            )
            uow.weighings.save(weighing)
            uow.commit()
        return weighing
