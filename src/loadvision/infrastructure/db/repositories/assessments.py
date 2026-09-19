"""Postgres repository for assessments."""

from typing import Any, Final
from uuid import UUID

from sqlalchemy import Connection, Row, Select, func, select, update
from sqlalchemy.dialects.postgresql import insert

from loadvision.application.ports import (
    AssessmentAlreadyExistsError,
    AssessmentOrder,
    ReviewStatus,
)
from loadvision.domain.assessment import Assessment, LoadPercent, ReviewReason, Signals
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.shipment import ShipmentId
from loadvision.infrastructure.db.tables import assessments, confirmed_labels

_NEWEST: Final = (assessments.c.created_at.desc(), assessments.c.id.desc())
_ORDER: Final = {
    AssessmentOrder.NEWEST: _NEWEST,
    AssessmentOrder.OLDEST: (assessments.c.created_at.asc(), assessments.c.id.asc()),
    AssessmentOrder.FULLEST: (assessments.c.load_pct.desc(), *_NEWEST),
    AssessmentOrder.EMPTIEST: (assessments.c.load_pct.asc(), *_NEWEST),
}


class PostgresAssessmentRepository:
    """`AssessmentRepository` on the connection of one unit of work."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def add(self, assessment: Assessment) -> None:
        # ON CONFLICT instead of catching the unique violation: a failed statement would abort
        # the whole transaction, and the port promises the unit of work stays usable. A parallel
        # transaction with the same shipment and photo makes the insert wait for its commit.
        # RETURNING tells whether a row went in: a Core insert on psycopg reports rowcount -1.
        inserted = self._connection.execute(
            insert(assessments)
            .values(_values(assessment))
            .on_conflict_do_nothing(
                index_elements=[assessments.c.shipment_id, assessments.c.image_sha256]
            )
            .returning(assessments.c.id)
        ).first()
        if inserted is None:
            raise AssessmentAlreadyExistsError(
                f"shipment {assessment.shipment_id.value} already has this photo"
            )

    def replace(self, assessment: Assessment) -> None:
        # RETURNING for the same reason as in `add`: a Core update on psycopg reports
        # rowcount -1, so the id that comes back is what tells that a row was there.
        updated = self._connection.execute(
            update(assessments)
            .where(assessments.c.id == assessment.id)
            .values(_values(assessment))
            .returning(assessments.c.id)
        ).first()
        if updated is None:
            raise ValueError(f"assessment {assessment.id} is not stored")

    def get(self, assessment_id: UUID) -> Assessment | None:
        return self._first(select(assessments).where(assessments.c.id == assessment_id))

    def find_by_shipment_and_image(
        self, shipment_id: ShipmentId, image_sha256: str
    ) -> Assessment | None:
        return self._first(
            select(assessments).where(
                assessments.c.shipment_id == shipment_id.value,
                assessments.c.image_sha256 == image_sha256,
            )
        )

    def find_by_image_and_model(
        self, image_sha256: str, model_version: ModelVersionId
    ) -> Assessment | None:
        return self._first(
            select(assessments)
            .where(
                assessments.c.image_sha256 == image_sha256,
                assessments.c.model_version == model_version.value,
            )
            .limit(1)
        )

    def list_by_shipment(self, shipment_id: ShipmentId) -> list[Assessment]:
        rows = self._connection.execute(
            select(assessments)
            .where(assessments.c.shipment_id == shipment_id.value)
            .order_by(assessments.c.created_at.desc())
        )
        return [_assessment(row) for row in rows]

    def list_page(
        self,
        *,
        order: AssessmentOrder,
        offset: int,
        limit: int,
        status: ReviewStatus = ReviewStatus.ALL,
    ) -> list[Assessment]:
        statement = (
            _filtered(select(assessments), status)
            .order_by(*_ORDER[order])
            .offset(offset)
            .limit(limit)
        )
        return [_assessment(row) for row in self._connection.execute(statement)]

    def count(self, *, status: ReviewStatus = ReviewStatus.ALL) -> int:
        statement = _filtered(select(func.count()).select_from(assessments), status)
        return self._connection.execute(statement).scalar_one()

    def _first(self, statement: Select[Any]) -> Assessment | None:
        row = self._connection.execute(statement).first()
        return None if row is None else _assessment(row)


def _filtered(statement: Select[Any], status: ReviewStatus) -> Select[Any]:
    label = select(confirmed_labels.c.image_sha256).where(
        confirmed_labels.c.image_sha256 == assessments.c.image_sha256
    )
    match status:
        case ReviewStatus.ALL:
            return statement
        case ReviewStatus.WAITING:
            return statement.where(~label.exists())
        case ReviewStatus.CONFIRMED:
            return statement.where(label.where(~confirmed_labels.c.unusable).exists())
        case ReviewStatus.UNUSABLE:
            return statement.where(label.where(confirmed_labels.c.unusable).exists())


def _values(assessment: Assessment) -> dict[str, object]:
    signals = assessment.signals
    return {
        "id": assessment.id,
        "shipment_id": assessment.shipment_id.value,
        "image_sha256": assessment.image_sha256,
        "load_pct": assessment.load_pct.value,
        "uncertainty": signals.uncertainty,
        "ood_score": signals.ood_score,
        "brightness": signals.brightness,
        "sharpness": signals.sharpness,
        # Sorted, so the same reasons are always stored as the same array.
        "review_reasons": sorted(reason.value for reason in assessment.review_reasons),
        "model_version": assessment.model_version.value,
        "processing_ms": assessment.processing_ms,
        "request_id": assessment.request_id,
        "created_at": assessment.created_at,
    }


def _assessment(row: Row[Any]) -> Assessment:
    return Assessment(
        id=row.id,
        shipment_id=ShipmentId(row.shipment_id),
        image_sha256=row.image_sha256,
        load_pct=LoadPercent(row.load_pct),
        model_version=ModelVersionId(row.model_version),
        processing_ms=row.processing_ms,
        created_at=row.created_at,
        signals=Signals(
            uncertainty=row.uncertainty,
            ood_score=row.ood_score,
            brightness=row.brightness,
            sharpness=row.sharpness,
        ),
        review_reasons=frozenset(ReviewReason(reason) for reason in row.review_reasons),
        request_id=row.request_id,
    )
