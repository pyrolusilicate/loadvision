"""Use case: get the current assessment and history of a shipment."""

from dataclasses import dataclass

from loadvision.application.labelled_assessments import LabeledAssessment, with_labels
from loadvision.application.ports import UnitOfWorkFactory
from loadvision.domain.shipment import ShipmentId
from loadvision.errors import ErrorCode, LoadvisionError


@dataclass(frozen=True, slots=True, kw_only=True)
class ShipmentHistory:
    """Assessments of one shipment.

    Attributes:
        shipment_id: The shipment.
        assessments: All assessments, newest first; never empty.
    """

    shipment_id: ShipmentId
    assessments: tuple[LabeledAssessment, ...]

    def __post_init__(self) -> None:
        """Validates the history.

        Raises:
            ValueError: If there are no assessments; a shipment is stored with its first one.
        """
        if not self.assessments:
            raise ValueError("shipment history must not be empty")

    @property
    def current(self) -> LabeledAssessment:
        """The newest assessment."""
        return self.assessments[0]


@dataclass(frozen=True, slots=True, kw_only=True)
class GetShipment:
    """Reads a shipment with its assessments and the labels of their photos.

    Attributes:
        uow_factory: Opens database transactions.
    """

    uow_factory: UnitOfWorkFactory

    def __call__(self, shipment_id: str) -> ShipmentHistory:
        """Returns the history of the shipment.

        Args:
            shipment_id: Shipment number as received; validated here.

        Returns:
            The assessments, newest first.

        Raises:
            LoadvisionError: `INVALID_SHIPMENT_ID` for a malformed number;
                `SHIPMENT_NOT_FOUND` if the shipment has no assessments.
        """
        shipment = ShipmentId(shipment_id)
        with self.uow_factory() as uow:
            assessments = uow.assessments.list_by_shipment(shipment)
            # A shipment is stored in the same transaction as its first assessment, so no
            # assessments means no shipment.
            if not assessments:
                raise LoadvisionError(ErrorCode.SHIPMENT_NOT_FOUND, shipment.value)
            return ShipmentHistory(shipment_id=shipment, assessments=with_labels(uow, assessments))
