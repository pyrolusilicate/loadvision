"""Postgres repository for assessments."""

from typing import Any
from uuid import UUID

from sqlalchemy import Connection, Row, Select, select
from sqlalchemy.dialects.postgresql import insert

from loadvision.application.ports import AssessmentAlreadyExistsError
from loadvision.domain.assessment import Assessment, LoadPercent, ReviewReason, Signals
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.shipment import ShipmentId
from loadvision.infrastructure.db.tables import assessments, confirmed_labels


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

    def list_recent(self, limit: int, *, unlabeled_only: bool = False) -> list[Assessment]:
        statement = select(assessments).order_by(assessments.c.created_at.desc()).limit(limit)
        if unlabeled_only:
            labelled = select(confirmed_labels.c.image_sha256).where(
                confirmed_labels.c.image_sha256 == assessments.c.image_sha256
            )
            statement = statement.where(~labelled.exists())
        return [_assessment(row) for row in self._connection.execute(statement)]

    def _first(self, statement: Select[Any]) -> Assessment | None:
        row = self._connection.execute(statement).first()
        return None if row is None else _assessment(row)


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
