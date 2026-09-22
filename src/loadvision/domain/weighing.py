"""What an operator weighed one assessment with, and what came out of it.

The weight is not a property of the photo. The same photo in two shipments may sit in two
different bodies, and it is the body that decides the volume, so a weighing belongs to the
assessment. It is also not the model's: the model gives the share of the floor that is taken,
and everything else - the body, the kind of cargo, what a cubic metre of it weighs - comes
from the person who knows the shipment.

Both sides are kept. The inputs, so anyone can see what the number was built from and redo it,
and the answer, so a later change to the table of densities does not silently rewrite what an
operator was shown on the day.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Self
from uuid import UUID

from loadvision.domain.cargo_weight import BodySize, CargoKind, WeightEstimate
from loadvision.domain.timestamps import ensure_aware


@dataclass(frozen=True, slots=True, kw_only=True)
class Weighing:
    """One weighing of one assessment.

    Attributes:
        assessment_id: The assessment this weighing belongs to; it has at most one.
        body: Key of the body that was chosen, or the one that stands for a size typed by hand.
        size: Inner size of that body, whichever way it was chosen.
        kind: Kind of cargo the density was taken for.
        estimate: What came out: the weight with its bounds and the volume behind them.
        author: Who weighed it, or None if unknown.
        created_at: When, timezone-aware.
    """

    assessment_id: UUID
    body: str
    size: BodySize
    kind: CargoKind
    estimate: WeightEstimate
    author: str | None
    created_at: datetime

    def __post_init__(self) -> None:
        """Validates the body key, the author and the time.

        Raises:
            ValueError: If `body` or `author` is blank, or `created_at` has no UTC offset.
        """
        if not self.body.strip():
            raise ValueError("body must name the body that was chosen")
        if self.author is not None and not self.author.strip():
            raise ValueError("author must not be blank; use None for an unknown author")
        ensure_aware(self.created_at, "created_at")

    @classmethod
    def of(
        cls,
        *,
        assessment_id: UUID,
        body: str,
        size: BodySize,
        kind: CargoKind,
        estimate: WeightEstimate,
        author: str | None,
        at: datetime,
    ) -> Self:
        """Builds a weighing done now, in the form the repository stores."""
        return cls(
            assessment_id=assessment_id,
            body=body,
            size=size,
            kind=kind,
            estimate=estimate,
            author=author,
            created_at=at,
        )
