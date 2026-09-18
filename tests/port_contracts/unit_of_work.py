"""Behavior every UnitOfWork implementation must have: the in-memory fake and Postgres."""

from datetime import timedelta
from uuid import UUID

import pytest

from loadvision.application.ports import (
    AssessmentAlreadyExistsError,
    UnitOfWork,
    UnitOfWorkFactory,
)
from loadvision.domain.assessment import Assessment, LoadPercent, ReviewReason, Signals
from loadvision.domain.confirmed_label import LabelSource
from loadvision.domain.model_version import ModelVersionId, ModelVersionStatus
from loadvision.domain.shipment import ShipmentId
from loadvision.errors import ErrorCode
from tests.builders import (
    T0,
    make_assessment,
    make_image,
    make_image_record,
    make_job,
    make_label,
    make_model_version,
    make_shipment,
)

PHOTO_A = b"photo-a"
PHOTO_B = b"photo-b"
UNKNOWN_SHA256 = "0" * 64


def store_assessment(uow: UnitOfWork, assessment: Assessment, raw: bytes = PHOTO_A) -> None:
    """Stores the assessment together with the shipment, photo and model rows it refers to."""
    assert make_image(raw).sha256 == assessment.image_sha256, "raw must be the assessed photo"
    uow.shipments.add(make_shipment(assessment.shipment_id.value))
    uow.images.add(make_image_record(raw))
    if uow.model_versions.get(assessment.model_version) is None:
        uow.model_versions.add(make_model_version(assessment.model_version.value))
    uow.assessments.add(assessment)


class UnitOfWorkContract:
    @pytest.fixture
    def uow_factory(self) -> UnitOfWorkFactory:
        raise NotImplementedError

    def test_committed_changes_are_seen_by_the_next_unit_of_work(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        shipment = make_shipment()
        with uow_factory() as uow:
            uow.shipments.add(shipment)
            uow.commit()
        with uow_factory() as uow:
            assert uow.shipments.get(shipment.id) == shipment

    def test_changes_are_seen_inside_the_unit_of_work_before_commit(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        shipment = make_shipment()
        with uow_factory() as uow:
            uow.shipments.add(shipment)
            assert uow.shipments.get(shipment.id) == shipment

    def test_changes_without_commit_are_rolled_back(self, uow_factory: UnitOfWorkFactory) -> None:
        with uow_factory() as uow:
            uow.shipments.add(make_shipment())
        with uow_factory() as uow:
            assert uow.shipments.get(ShipmentId("RP-1")) is None

    def test_exception_rolls_back_and_propagates(self, uow_factory: UnitOfWorkFactory) -> None:
        with pytest.raises(RuntimeError, match="boom"), uow_factory() as uow:
            uow.shipments.add(make_shipment())
            raise RuntimeError("boom")
        with uow_factory() as uow:
            assert uow.shipments.get(ShipmentId("RP-1")) is None

    def test_unit_of_work_is_not_used_outside_its_with_block(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        uow = uow_factory()
        with pytest.raises(RuntimeError, match="outside its with block"):
            uow.shipments.get(ShipmentId("RP-1"))
        with uow:
            pass
        with pytest.raises(RuntimeError, match="outside its with block"):
            uow.commit()

    def test_unknown_rows_are_absent(self, uow_factory: UnitOfWorkFactory) -> None:
        with uow_factory() as uow:
            assert uow.shipments.get(ShipmentId("RP-404")) is None
            assert uow.assessments.get(UUID(int=404)) is None
            assert uow.assessments.list_by_shipment(ShipmentId("RP-404")) == []
            assert uow.images.get(UNKNOWN_SHA256) is None
            assert uow.model_versions.get(ModelVersionId("v404")) is None
            assert uow.model_versions.get_active() is None
            assert uow.model_versions.list_all() == []

    def test_adding_a_known_shipment_keeps_the_stored_one(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        first = make_shipment(created_at=T0)
        with uow_factory() as uow:
            uow.shipments.add(first)
            uow.commit()
        with uow_factory() as uow:
            uow.shipments.add(make_shipment(created_at=T0 + timedelta(hours=1)))
            uow.commit()
        with uow_factory() as uow:
            assert uow.shipments.get(first.id) == first

    def test_adding_a_known_image_keeps_the_stored_record(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        first = make_image_record(PHOTO_A, created_at=T0)
        with uow_factory() as uow:
            uow.images.add(first)
            uow.commit()
        with uow_factory() as uow:
            uow.images.add(make_image_record(PHOTO_A, created_at=T0 + timedelta(hours=1)))
            uow.commit()
        with uow_factory() as uow:
            assert uow.images.get(first.sha256) == first

    def test_assessment_is_found_by_id_and_by_shipment_and_photo(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        assessment = make_assessment()
        with uow_factory() as uow:
            store_assessment(uow, assessment)
            uow.commit()
        with uow_factory() as uow:
            found = uow.assessments.find_by_shipment_and_image
            assert uow.assessments.get(assessment.id) == assessment
            assert found(assessment.shipment_id, assessment.image_sha256) == assessment
            assert found(ShipmentId("RP-2"), assessment.image_sha256) is None
            assert found(assessment.shipment_id, UNKNOWN_SHA256) is None

    def test_assessment_keeps_its_signals_reasons_and_request_id(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        assessment = make_assessment(
            signals=Signals(uncertainty=1.5, ood_score=0.25, brightness=0.5, sharpness=0.75),
            review_reasons=frozenset({ReviewReason.TOO_DARK, ReviewReason.LOW_CONFIDENCE}),
            request_id="req-1",
        )
        with uow_factory() as uow:
            store_assessment(uow, assessment)
            uow.commit()
        with uow_factory() as uow:
            assert uow.assessments.get(assessment.id) == assessment

    def test_duplicate_photo_in_a_shipment_raises_and_unit_of_work_stays_usable(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        stored = make_assessment()
        with uow_factory() as uow:
            store_assessment(uow, stored)
            uow.commit()
        with uow_factory() as uow:
            with pytest.raises(AssessmentAlreadyExistsError):
                uow.assessments.add(make_assessment(id=UUID(int=2)))
            assert (
                uow.assessments.find_by_shipment_and_image(stored.shipment_id, stored.image_sha256)
                == stored
            )
            uow.shipments.add(make_shipment("RP-2"))
            uow.commit()
        with uow_factory() as uow:
            assert uow.shipments.get(ShipmentId("RP-2")) is not None
            assert uow.assessments.get(UUID(int=2)) is None

    def test_same_photo_is_allowed_in_another_shipment(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        second = make_assessment(id=UUID(int=2), shipment_id=ShipmentId("RP-2"))
        with uow_factory() as uow:
            store_assessment(uow, make_assessment())
            store_assessment(uow, second)
            uow.commit()
        with uow_factory() as uow:
            assert uow.assessments.get(second.id) == second

    def test_assessment_is_found_by_photo_and_model_version(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        assessment = make_assessment(model_version=ModelVersionId("v1"))
        with uow_factory() as uow:
            store_assessment(uow, assessment)
            uow.commit()
        with uow_factory() as uow:
            found = uow.assessments.find_by_image_and_model
            assert found(assessment.image_sha256, ModelVersionId("v1")) == assessment
            assert found(assessment.image_sha256, ModelVersionId("v2")) is None
            assert found(UNKNOWN_SHA256, ModelVersionId("v1")) is None

    def test_shipment_history_is_newest_first(self, uow_factory: UnitOfWorkFactory) -> None:
        older = make_assessment(id=UUID(int=1), created_at=T0)
        newer = make_assessment(
            id=UUID(int=2),
            image_sha256=make_image(PHOTO_B).sha256,
            created_at=T0 + timedelta(hours=1),
        )
        elsewhere = make_assessment(id=UUID(int=3), shipment_id=ShipmentId("RP-2"))
        with uow_factory() as uow:
            store_assessment(uow, older)
            store_assessment(uow, newer, PHOTO_B)
            store_assessment(uow, elsewhere)
            uow.commit()
        with uow_factory() as uow:
            assert uow.assessments.list_by_shipment(ShipmentId("RP-1")) == [newer, older]

    def test_model_versions_track_the_active_one_and_list_newest_first(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        old = make_model_version(
            "v1", status=ModelVersionStatus.ACTIVE, created_at=T0, activated_at=T0
        )
        new = make_model_version("v2", created_at=T0 + timedelta(hours=1))
        with uow_factory() as uow:
            uow.model_versions.add(old)
            uow.model_versions.add(new)
            uow.commit()
        with uow_factory() as uow:
            assert uow.model_versions.get_active() == old
            assert uow.model_versions.list_all() == [new, old]
            uow.model_versions.update(old.retire())
            uow.model_versions.update(new.activate(T0 + timedelta(hours=2)))
            uow.commit()
        with uow_factory() as uow:
            assert uow.model_versions.get_active() == new.activate(T0 + timedelta(hours=2))
            assert uow.model_versions.get(old.version) == old.retire()

    def test_job_is_stored_with_its_outcome_and_read_back(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        version = make_model_version("v1")
        finished = (
            make_job()
            .started(at=T0 + timedelta(minutes=1), base_version=version.version)
            .succeeded(
                at=T0 + timedelta(minutes=9),
                run_id="run_20260918_101500_ab12",
                result_version=version.version,
                gate_passed=True,
            )
        )
        with uow_factory() as uow:
            uow.model_versions.add(version)
            uow.training_jobs.add(make_job())
            uow.training_jobs.update(finished)
            uow.commit()
        with uow_factory() as uow:
            assert uow.training_jobs.get(finished.id) == finished
            assert uow.training_jobs.get(UUID(int=404)) is None

    def test_failed_job_keeps_its_error_code(self, uow_factory: UnitOfWorkFactory) -> None:
        failed = (
            make_job()
            .started(at=T0, base_version=None)
            .failed(at=T0, run_id=None, error_code=ErrorCode.STAGE_CHECK_FAILED)
        )
        with uow_factory() as uow:
            uow.training_jobs.add(make_job())
            uow.training_jobs.update(failed)
            uow.commit()
        with uow_factory() as uow:
            assert uow.training_jobs.get(failed.id) == failed

    def test_the_oldest_queued_job_is_claimed_first(self, uow_factory: UnitOfWorkFactory) -> None:
        older = make_job(UUID(int=1), at=T0)
        newer = make_job(UUID(int=2), at=T0 + timedelta(minutes=5))
        with uow_factory() as uow:
            uow.training_jobs.add(newer)
            uow.training_jobs.add(older)
            uow.commit()
        with uow_factory() as uow:
            assert uow.training_jobs.claim_next() == older

    def test_nothing_is_claimed_while_a_job_is_running(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        with uow_factory() as uow:
            uow.training_jobs.add(make_job(UUID(int=1)))
            uow.training_jobs.update(make_job(UUID(int=1)).started(at=T0, base_version=None))
            uow.training_jobs.add(make_job(UUID(int=2), at=T0 + timedelta(minutes=5)))
            uow.commit()
        with uow_factory() as uow:
            assert uow.training_jobs.claim_next() is None

    def test_an_empty_queue_claims_nothing(self, uow_factory: UnitOfWorkFactory) -> None:
        with uow_factory() as uow:
            assert uow.training_jobs.claim_next() is None
            assert uow.training_jobs.find_unfinished() is None

    def test_a_job_is_unfinished_until_it_ends(self, uow_factory: UnitOfWorkFactory) -> None:
        job = make_job()
        with uow_factory() as uow:
            uow.training_jobs.add(job)
            uow.commit()
        with uow_factory() as uow:
            assert uow.training_jobs.find_unfinished() == job
            uow.training_jobs.update(
                job.started(at=T0, base_version=None).failed(
                    at=T0, run_id="run-1", error_code=ErrorCode.INTERNAL_ERROR
                )
            )
            uow.commit()
        with uow_factory() as uow:
            assert uow.training_jobs.find_unfinished() is None

    def test_recent_jobs_come_newest_first(self, uow_factory: UnitOfWorkFactory) -> None:
        jobs = [make_job(UUID(int=n), at=T0 + timedelta(minutes=n)) for n in range(1, 4)]
        with uow_factory() as uow:
            for job in jobs:
                uow.training_jobs.add(job)
            uow.commit()
        with uow_factory() as uow:
            assert uow.training_jobs.list_recent(2) == [jobs[2], jobs[1]]
            assert uow.training_jobs.list_recent(10) == list(reversed(jobs))

    def test_updating_an_unstored_job_fails(self, uow_factory: UnitOfWorkFactory) -> None:
        with uow_factory() as uow, pytest.raises(ValueError, match="not stored"):
            uow.training_jobs.update(make_job().started(at=T0, base_version=None))

    def test_updating_an_unregistered_model_version_fails(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        with uow_factory() as uow, pytest.raises(ValueError, match="not registered"):
            uow.model_versions.update(make_model_version("v404"))

    def test_label_is_saved_and_read_back(self, uow_factory: UnitOfWorkFactory) -> None:
        label = make_label(PHOTO_A, 70.0)
        with uow_factory() as uow:
            uow.images.add(make_image_record(PHOTO_A))
            uow.labels.save(label)
            uow.commit()
        with uow_factory() as uow:
            assert uow.labels.get(label.image_sha256) == label
            assert uow.labels.get(UNKNOWN_SHA256) is None

    def test_saving_a_label_of_the_same_photo_replaces_it(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        first = make_label(PHOTO_A, 70.0)
        second = first.replaced(
            load_pct=LoadPercent(90.0),
            source=LabelSource.UI,
            author=None,
            at=T0 + timedelta(minutes=5),
        )
        with uow_factory() as uow:
            uow.images.add(make_image_record(PHOTO_A))
            uow.labels.save(first)
            uow.commit()
        with uow_factory() as uow:
            uow.labels.save(second)
            uow.commit()
        with uow_factory() as uow:
            assert uow.labels.get(first.image_sha256) == second
            assert uow.labels.list_for_training() == [second]

    def test_training_gets_usable_labels_sorted_by_photo_hash(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        photos = [b"photo-a", b"photo-b", b"photo-c"]
        labels = [
            make_label(photos[0], 40.0),
            make_label(photos[1], None),
            make_label(photos[2], 90.0),
        ]
        with uow_factory() as uow:
            for photo, label in zip(photos, labels, strict=True):
                uow.images.add(make_image_record(photo))
                uow.labels.save(label)
            uow.commit()
        with uow_factory() as uow:
            assert uow.labels.get(labels[1].image_sha256) == labels[1]
            usable = [labels[0], labels[2]]
            assert uow.labels.list_for_training() == sorted(
                usable, key=lambda row: row.image_sha256
            )
