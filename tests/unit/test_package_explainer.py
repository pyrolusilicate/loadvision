import io

import pytest
from PIL import Image

from loadvision.application.ports import ExplanationKind
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.validator import ImageLimits, prepare_image
from loadvision.ml.inference.explainer import MAP_MAX_SIDE, PackageExplainer
from tests import image_files

LIMITS = ImageLimits(
    max_bytes=20_000_000, max_pixels=10_000_000, min_side=1, media_types=frozenset(MediaType)
)
PHOTO = prepare_image(image_files.jpeg(), LIMITS)
VERSION = ModelVersionId("v2")


class Drawing:
    """Stands in for the drawing: gives back a PNG of the size it was asked for."""

    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.asked: list[tuple[str, str]] = []

    def picture(self, image: PreparedImage, kind: str) -> bytes:
        self.asked.append((image.sha256, kind))
        buffer = io.BytesIO()
        image_files.picture(*self.size).save(buffer, "PNG")
        return buffer.getvalue()


def shown(picture: bytes) -> tuple[str, tuple[int, int]]:
    with Image.open(io.BytesIO(picture)) as drawn:
        return drawn.format or "", drawn.size


@pytest.mark.unit
def test_a_map_of_a_big_photo_is_cut_down_to_what_a_page_shows() -> None:
    drawing = Drawing(2400, 3200)
    explainer = PackageExplainer(drawing, VERSION)

    picture = explainer.explain(PHOTO, ExplanationKind.CONTRIBUTION)

    assert drawing.asked == [(PHOTO.sha256, "contribution")]
    kind, size = shown(picture)
    # JPEG, and the long side is the one a browser is asked to download.
    assert kind == "JPEG"
    assert max(size) == MAP_MAX_SIDE
    assert size == (1200, MAP_MAX_SIDE)


@pytest.mark.unit
def test_a_small_map_keeps_its_size() -> None:
    explainer = PackageExplainer(Drawing(600, 800), VERSION)

    picture = explainer.explain(PHOTO, ExplanationKind.ATTENTION)

    assert shown(picture) == ("JPEG", (600, 800))


@pytest.mark.unit
def test_the_explainer_names_the_version_of_its_package() -> None:
    assert PackageExplainer(Drawing(8, 8), VERSION).version == VERSION
