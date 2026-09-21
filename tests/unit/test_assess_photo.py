from collections.abc import Sequence
from dataclasses import replace
from uuid import UUID

import pytest

from loadvision.application.assess_photo import AssessPhoto, AssessPhotoRequest
from loadvision.application.ports import (
    ImageRecord,
    LoadPredictor,
    Prediction,
    UnitOfWorkFactory,
)
from loadvision.domain.assessment import LoadPercent, ReviewReason
from loadvision.domain.confirmed_label import LabelSource
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.shipment import Shipment, ShipmentId
from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.validator import ImageLimits, prepare_image
from tests import image_files
from tests.builders import T0, make_assessment, make_image_record, make_label
from tests.fakes.clock import FixedClock, SequentialIds
from tests.fakes.image_storage import InMemoryImageStorage
from tests.fakes.in_memory_db import InMemoryStore, InMemoryUnitOfWork
from tests.fakes.predictor import OOD_THRESHOLD, FakePredictor, band_line

LIMITS = ImageLimits(
    max_bytes=2_000_000,
    max_pixels=1_000_000,
    min_side=256,
    media_types=frozenset(MediaType),
)
PHOTO = image_files.jpeg()
DARK_PHOTO = image_files.dark_jpeg()
MIN_BRIGHTNESS = 20.0
UNUSUAL_SHARE = 0.05
UNREADABLE_SHARE = 0.8
# The distance those shares stand at on the scale the fake model reports.
UNREADABLE_OOD = band_line(UNREADABLE_SHARE)
# Mean brightness of the two photos above, measured by `imaging.quality.brightness`.
PHOTO_BRIGHTNESS = 127.11
DARK_BRIGHTNESS = 5.0
OTHER_PHOTO = image_files.png()
PHOTO_SHA256 = prepare_image(PHOTO, LIMITS).sha256


class StepTimer:
    """Monotonic timer that moves 0.125 s forward on every call."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        self._now += 0.125
        return self._now


class StubPredictor:
    """Predictor that returns a fixed list for every batch."""

    def __init__(self, predictions: list[Prediction]) -> None:
        self.predictions = predictions

    @property
    def version(self) -> ModelVersionId:
        return ModelVersionId("stub-1")

    def predict(self, images: Sequence[PreparedImage]) -> list[Prediction]:
        return self.predictions


class FailingImageStorage(InMemoryImageStorage):
    def save(self, image: PreparedImage) -> None:
        raise OSError("disk full")


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def storage() -> InMemoryImageStorage:
    return InMemoryImageStorage()


def use_case(
    store: InMemoryStore,
    storage: InMemoryImageStorage,
    uow_factory: UnitOfWorkFactory | None = None,
    min_brightness: float = MIN_BRIGHTNESS,
    unusual_band_share: float = UNUSUAL_SHARE,
    unreadable_band_share: float = UNREADABLE_SHARE,
) -> AssessPhoto:
    return AssessPhoto(
        uow_factory=store.unit_of_work if uow_factory is None else uow_factory,
        storage=storage,
        limits=LIMITS,
        min_brightness=min_brightness,
        unusual_band_share=unusual_band_share,
        unreadable_band_share=unreadable_band_share,
        clock=FixedClock(T0),
        new_id=SequentialIds(),
        timer=StepTimer(),
    )


def request(photo: bytes = PHOTO, shipment_id: str = "RP-1") -> AssessPhotoRequest:
    return AssessPhotoRequest(shipment_id=shipment_id, image=photo, request_id="req-1")


def rejection(assess: AssessPhoto, req: AssessPhotoRequest, predictor: LoadPredictor) -> ErrorCode:
    with pytest.raises(LoadvisionError) as caught:
        assess(req, predictor)
    return caught.value.code


def assert_nothing_stored(store: InMemoryStore, storage: InMemoryImageStorage) -> None:
    assert (store.shipments, store.images, store.assessments, storage.files) == ({}, {}, {}, {})


@pytest.mark.unit
def test_new_photo_is_assessed_stored_and_attached_to_a_new_shipment(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    predictor = FakePredictor(loads={PHOTO_SHA256: 85.0})

    result = use_case(store, storage)(request(), predictor)

    assessment = result.assessment
    assert result.reused is False
    assert assessment.id == UUID(int=1)
    assert assessment.shipment_id == ShipmentId("RP-1")
    assert assessment.image_sha256 == PHOTO_SHA256
    assert assessment.load_pct == LoadPercent(85.0)
    assert assessment.model_version == ModelVersionId("fake-1")
    assert assessment.processing_ms == 125
    assert assessment.created_at == T0
    assert assessment.request_id == "req-1"
    assert result.image == ImageRecord(
        sha256=PHOTO_SHA256,
        media_type=MediaType.JPEG,
        width=300,
        height=400,
        size_bytes=len(PHOTO),
        created_at=T0,
    )
    assert store.shipments == {ShipmentId("RP-1"): Shipment(ShipmentId("RP-1"), T0)}
    assert store.images == {PHOTO_SHA256: result.image}
    assert store.assessments == {assessment.id: assessment}
    assert storage.files == {PHOTO_SHA256: PHOTO}
    assert predictor.batches == [[PHOTO_SHA256]]


@pytest.mark.unit
def test_same_photo_in_the_same_shipment_returns_the_stored_assessment(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    predictor = FakePredictor()
    first = use_case(store, storage)(request(), predictor)

    again = use_case(store, storage)(request(), predictor)

    assert again.reused is True
    assert again.replaced is None
    assert again.assessment == first.assessment
    assert again.image == first.image
    assert list(store.assessments) == [first.assessment.id]
    assert predictor.batches == [[PHOTO_SHA256]]


@pytest.mark.unit
def test_a_newer_model_assesses_the_photo_again_in_place_of_the_stored_estimate(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    assess = use_case(store, storage)
    first = assess(request(), FakePredictor(version="fake-1"))
    newer = FakePredictor(version="fake-2", loads={PHOTO_SHA256: 40.0})

    again = assess(request(), newer)

    assert again.reused is False
    assert again.replaced == ModelVersionId("fake-1")
    assert newer.batches == [[PHOTO_SHA256]]
    # One photo of one shipment keeps one assessment, and its address stays the same.
    assert again.assessment.id == first.assessment.id
    assert store.assessments == {first.assessment.id: again.assessment}
    assert again.assessment.load_pct == LoadPercent(40.0)
    assert again.assessment.model_version == ModelVersionId("fake-2")


@pytest.mark.unit
def test_same_photo_in_another_shipment_reuses_the_prediction_of_the_same_model(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    predictor = FakePredictor()
    assess = use_case(store, storage)
    first = assess(request(shipment_id="RP-1"), predictor)

    second = assess(request(shipment_id="RP-2"), predictor)

    assert second.reused is False
    assert second.assessment.shipment_id == ShipmentId("RP-2")
    assert second.assessment.load_pct == first.assessment.load_pct
    assert second.assessment.signals == first.assessment.signals
    assert predictor.batches == [[PHOTO_SHA256]]
    assert len(store.assessments) == 2


@pytest.mark.unit
def test_another_model_version_predicts_again(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    assess = use_case(store, storage)
    assess(request(shipment_id="RP-1"), FakePredictor(version="fake-1"))
    newer = FakePredictor(version="fake-2", loads={PHOTO_SHA256: 40.0})

    result = assess(request(shipment_id="RP-2"), newer)

    assert newer.batches == [[PHOTO_SHA256]]
    assert result.assessment.load_pct == LoadPercent(40.0)
    assert result.assessment.model_version == ModelVersionId("fake-2")


@pytest.mark.unit
def test_new_photo_joins_the_history_of_a_known_shipment(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    predictor = FakePredictor()
    assess = use_case(store, storage)
    assess(request(PHOTO), predictor)
    assess(request(OTHER_PHOTO), predictor)

    with store.unit_of_work() as uow:
        history = uow.assessments.list_by_shipment(ShipmentId("RP-1"))
    assert len(history) == 2
    assert len(store.shipments) == 1


@pytest.mark.unit
def test_prediction_signals_are_kept(store: InMemoryStore, storage: InMemoryImageStorage) -> None:
    predictor = StubPredictor([Prediction(LoadPercent(60.0), uncertainty=7.5, ood_score=0.2)])

    result = use_case(store, storage)(request(), predictor)

    signals = result.assessment.signals
    assert (signals.uncertainty, signals.ood_score) == (7.5, 0.2)
    assert signals.brightness == pytest.approx(PHOTO_BRIGHTNESS, abs=0.01)


@pytest.mark.unit
def test_the_reasons_of_the_model_to_check_it_are_stored(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    unusual = frozenset({ReviewReason.UNUSUAL_SCENE})
    predictor = StubPredictor([Prediction(LoadPercent(60.0), review_reasons=unusual)])

    result = use_case(store, storage)(request(), predictor)

    with store.unit_of_work() as uow:
        stored = uow.assessments.get(result.assessment.id)
    assert stored is not None
    assert stored.review_reasons == unusual
    assert stored.needs_review is True


@pytest.mark.unit
def test_a_photo_shot_in_the_dark_is_sent_to_a_person(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    result = use_case(store, storage)(request(photo=DARK_PHOTO), FakePredictor())

    assert result.assessment.review_reasons == frozenset({ReviewReason.TOO_DARK})
    assert result.assessment.signals.brightness == pytest.approx(DARK_BRIGHTNESS, abs=0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("photo", "min_brightness"),
    [(PHOTO, MIN_BRIGHTNESS), (DARK_PHOTO, DARK_BRIGHTNESS)],
    ids=["an ordinary photo", "a photo exactly at the threshold"],
)
def test_a_photo_light_enough_to_see_is_not_marked(
    store: InMemoryStore,
    storage: InMemoryImageStorage,
    photo: bytes,
    min_brightness: float,
) -> None:
    assess = use_case(store, storage, min_brightness=min_brightness)

    result = assess(request(photo=photo), FakePredictor())

    assert result.assessment.review_reasons == frozenset()


@pytest.mark.unit
def test_a_frame_with_no_truck_body_in_it_is_marked_unusable_by_the_model(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    result = use_case(store, storage)(request(), FakePredictor(ood_score=UNREADABLE_OOD))

    label = store.labels[result.assessment.image_sha256]
    assert (label.unusable, label.load_pct, label.author) == (True, None, None)
    assert label.source is LabelSource.MODEL


@pytest.mark.unit
@pytest.mark.parametrize(
    "ood_score",
    [UNREADABLE_OOD - 0.001, None],
    ids=["a photo just below the line", "a model that measures no distance"],
)
def test_a_photo_the_model_can_read_gets_no_verdict(
    store: InMemoryStore, storage: InMemoryImageStorage, ood_score: float | None
) -> None:
    use_case(store, storage)(request(), FakePredictor(ood_score=ood_score))

    assert store.labels == {}


@pytest.mark.unit
def test_the_line_is_read_against_the_scale_of_the_model_that_answered(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    # One and the same distance. For the model above it is twice its own threshold, so nothing
    # of a body is in the frame; for a model whose training photos lie further apart it is an
    # ordinary photo. A distance fixed in the settings would have called both the same.
    assess = use_case(store, storage)

    assess(request(PHOTO), FakePredictor(ood_score=UNREADABLE_OOD))
    assess(
        request(OTHER_PHOTO, shipment_id="RP-2"),
        FakePredictor(version="fake-2", ood_score=UNREADABLE_OOD, ood_threshold=OOD_THRESHOLD * 2),
    )

    assert list(store.labels) == [PHOTO_SHA256]


@pytest.mark.unit
def test_a_model_that_measures_no_distance_refuses_no_frame(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    use_case(store, storage)(request(), FakePredictor(ood_score=UNREADABLE_OOD, ood_threshold=None))

    assert store.labels == {}


@pytest.mark.unit
def test_the_model_never_overrules_a_person(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    # A photo somebody labelled before it was ever assessed here, such as an imported one.
    with store.unit_of_work() as uow:
        uow.images.add(make_image_record(PHOTO))
        uow.labels.save(make_label(PHOTO, 60.0))
        uow.commit()

    use_case(store, storage)(request(), FakePredictor(ood_score=UNREADABLE_OOD))

    label = store.labels[PHOTO_SHA256]
    assert (label.source, label.load_pct) == (LabelSource.API, LoadPercent(60.0))


@pytest.mark.unit
def test_a_photo_the_model_already_judged_is_not_judged_again(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    assess = use_case(store, storage)
    assess(request(shipment_id="RP-1"), FakePredictor(ood_score=UNREADABLE_OOD))
    with store.unit_of_work() as uow:
        uow.labels.delete(PHOTO_SHA256)
        uow.commit()

    assess(request(shipment_id="RP-2"), FakePredictor(ood_score=UNREADABLE_OOD))

    assert store.labels == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("req", "code"),
    [
        (request(shipment_id="RP 1"), ErrorCode.INVALID_SHIPMENT_ID),
        (request(photo=image_files.heic_header()), ErrorCode.UNSUPPORTED_FORMAT),
        (request(photo=image_files.truncated(PHOTO)), ErrorCode.CORRUPTED_IMAGE),
        (request(photo=b""), ErrorCode.CORRUPTED_IMAGE),
    ],
    ids=["shipment-id", "heic", "truncated", "empty"],
)
def test_invalid_request_is_rejected_before_the_model_runs(
    store: InMemoryStore, storage: InMemoryImageStorage, req: AssessPhotoRequest, code: ErrorCode
) -> None:
    predictor = FakePredictor()
    assert rejection(use_case(store, storage), req, predictor) is code
    assert predictor.batches == []
    assert_nothing_stored(store, storage)


@pytest.mark.unit
def test_too_large_photo_is_rejected(store: InMemoryStore, storage: InMemoryImageStorage) -> None:
    assess = replace(use_case(store, storage), limits=replace(LIMITS, max_bytes=len(PHOTO) - 1))
    assert rejection(assess, request(), FakePredictor()) is ErrorCode.PAYLOAD_TOO_LARGE


@pytest.mark.unit
@pytest.mark.parametrize(
    "predictor",
    [
        FakePredictor(error=RuntimeError("CUDA out of memory")),
        StubPredictor([]),
        StubPredictor([Prediction(LoadPercent(10.0)), Prediction(LoadPercent(20.0))]),
        StubPredictor([Prediction(LoadPercent(10.0), uncertainty=float("nan"))]),
        StubPredictor([Prediction(LoadPercent(10.0), ood_score=-1.0)]),
    ],
    ids=["raises", "no-prediction", "two-predictions", "nan-uncertainty", "negative-ood"],
)
def test_model_failure_is_reported_and_nothing_is_stored(
    store: InMemoryStore, storage: InMemoryImageStorage, predictor: LoadPredictor
) -> None:
    assert rejection(use_case(store, storage), request(), predictor) is (
        ErrorCode.MODEL_INFERENCE_FAILED
    )
    assert_nothing_stored(store, storage)


@pytest.mark.unit
def test_failed_file_save_leaves_no_records(store: InMemoryStore) -> None:
    storage = FailingImageStorage()
    with pytest.raises(OSError, match="disk full"):
        use_case(store, storage)(request(), FakePredictor())
    assert (store.shipments, store.images, store.assessments) == ({}, {}, {})


@pytest.mark.unit
def test_parallel_request_that_committed_first_wins(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    competitor = make_assessment(id=UUID(int=99), image_sha256=PHOTO_SHA256)
    record = ImageRecord.from_prepared(prepare_image(PHOTO, LIMITS), T0)
    opened = 0

    def unit_of_work() -> InMemoryUnitOfWork:
        # Between our read and our write, another request stores the same photo and shipment.
        nonlocal opened
        opened += 1
        if opened == 2:
            with store.unit_of_work() as uow:
                uow.shipments.add(Shipment(ShipmentId("RP-1"), T0))
                uow.images.add(record)
                uow.assessments.add(competitor)
                uow.commit()
        return store.unit_of_work()

    result = use_case(store, storage, unit_of_work)(request(), FakePredictor())

    assert result.reused is True
    assert result.assessment == competitor
    assert list(store.assessments) == [competitor.id]


@pytest.mark.unit
def test_a_photo_just_past_the_model_line_is_not_sent_to_a_person(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    """A package flags a scene from its own far edge, which ordinary photos reach too."""
    just_over = OOD_THRESHOLD * 1.01

    result = use_case(store, storage)(
        request(),
        FakePredictor(ood_score=just_over, review_reasons={ReviewReason.UNUSUAL_SCENE}),
    )

    assert result.assessment.review_reasons == frozenset()
    assert result.assessment.needs_review is False


@pytest.mark.unit
def test_a_photo_past_the_service_line_is_sent_to_a_person(
    store: InMemoryStore, storage: InMemoryImageStorage
) -> None:
    far = band_line(UNUSUAL_SHARE)

    result = use_case(store, storage)(
        request(), FakePredictor(ood_score=far, review_reasons={ReviewReason.UNUSUAL_SCENE})
    )

    assert result.assessment.review_reasons == frozenset({ReviewReason.UNUSUAL_SCENE})
