import pytest

from loadvision.application.explain_photo import ExplainPhoto
from loadvision.application.ports import ExplanationKind
from loadvision.domain.model_version import ModelVersionId
from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.validator import ImageLimits, prepare_image
from tests import image_files
from tests.fakes.heatmaps import PNG
from tests.fakes.image_storage import InMemoryImageStorage

LIMITS = ImageLimits(
    max_bytes=2_000_000, max_pixels=1_000_000, min_side=256, media_types=frozenset(MediaType)
)
PHOTO = image_files.jpeg()


class StubExplainer:
    """Records the photo it was asked about, or fails on every photo."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.images: list[PreparedImage] = []

    @property
    def version(self) -> ModelVersionId:
        return ModelVersionId("stub-1")

    def explain(self, image: PreparedImage, kind: ExplanationKind) -> bytes:
        self.images.append(image)
        if self.error is not None:
            raise self.error
        return PNG


def use_case(storage: InMemoryImageStorage) -> ExplainPhoto:
    return ExplainPhoto(storage=storage, limits=LIMITS)


def stored_photo() -> tuple[InMemoryImageStorage, str]:
    storage = InMemoryImageStorage()
    image = prepare_image(PHOTO, LIMITS)
    storage.save(image)
    return storage, image.sha256


@pytest.mark.unit
def test_the_map_is_drawn_on_the_photo_as_the_model_sees_it() -> None:
    storage, sha256 = stored_photo()
    explainer = StubExplainer()

    png = use_case(storage)(sha256, ExplanationKind.ATTENTION, explainer)

    assert png == PNG
    # The same bytes the prediction was made on, decoded the same way.
    assert [image.sha256 for image in explainer.images] == [sha256]


@pytest.mark.unit
def test_a_photo_that_is_not_stored_is_a_missing_page() -> None:
    with pytest.raises(LoadvisionError) as caught:
        use_case(InMemoryImageStorage())("0" * 64, ExplanationKind.ATTENTION, StubExplainer())

    assert caught.value.code is ErrorCode.ROUTE_NOT_FOUND


@pytest.mark.unit
def test_a_model_failure_is_reported_as_one_code() -> None:
    storage, sha256 = stored_photo()
    explainer = StubExplainer(error=RuntimeError("CUDA out of memory"))

    with pytest.raises(LoadvisionError) as caught:
        use_case(storage)(sha256, ExplanationKind.CONTRIBUTION, explainer)

    assert caught.value.code is ErrorCode.MODEL_INFERENCE_FAILED
