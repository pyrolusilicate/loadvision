import pytest

from loadvision.application.ports import ExplanationKind, LoadExplainer
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.validator import ImageLimits, prepare_image
from loadvision.ml.inference.active_explainer import ActiveExplainer
from tests import image_files

LIMITS = ImageLimits(
    max_bytes=2_000_000, max_pixels=1_000_000, min_side=256, media_types=frozenset(MediaType)
)
PHOTO = prepare_image(image_files.jpeg(), LIMITS)
V1 = ModelVersionId("v1")
V2 = ModelVersionId("v2")


class StubExplainer:
    def __init__(self, version: ModelVersionId) -> None:
        self._version = version

    @property
    def version(self) -> ModelVersionId:
        return self._version

    def explain(self, image: PreparedImage, kind: ExplanationKind) -> bytes:
        return f"{self._version.value}-{kind.value}".encode()


class Loader:
    """Builds an explainer per version and records how often it was asked to."""

    def __init__(self) -> None:
        self.versions: list[ModelVersionId] = []

    def __call__(self, version: ModelVersionId) -> LoadExplainer:
        self.versions.append(version)
        return StubExplainer(version)


@pytest.mark.unit
def test_the_explainer_is_built_on_the_first_request_and_kept() -> None:
    loader = Loader()
    active = ActiveExplainer(loader)

    assert loader.versions == []
    first = active.of(V1).explain(PHOTO, ExplanationKind.ATTENTION)
    second = active.of(V1).explain(PHOTO, ExplanationKind.CONTRIBUTION)

    assert (first, second) == (b"v1-attention", b"v1-contribution")
    assert loader.versions == [V1]


@pytest.mark.unit
def test_another_model_version_gets_its_own_explainer() -> None:
    loader = Loader()
    active = ActiveExplainer(loader)

    active.of(V1)
    newer = active.of(V2)

    assert newer.version == V2
    assert loader.versions == [V1, V2]
