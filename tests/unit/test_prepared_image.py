import hashlib

import numpy as np
import pytest
from numpy.typing import NDArray

from loadvision.imaging.prepared_image import MediaType, PreparedImage

RAW = b"\xff\xd8\xff\xe0 bytes of a photo"


def make_pixels(height: int = 4, width: int = 3) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


@pytest.mark.unit
def test_media_type_values_are_mime_types() -> None:
    assert [media_type.value for media_type in MediaType] == [
        "image/jpeg",
        "image/png",
        "image/webp",
    ]


@pytest.mark.unit
def test_prepared_image_derives_size_and_hash() -> None:
    image = PreparedImage(RAW, MediaType.JPEG, make_pixels(height=4, width=3))
    assert (image.height, image.width) == (4, 3)
    assert image.sha256 == hashlib.sha256(RAW).hexdigest()


@pytest.mark.unit
def test_prepared_image_rejects_empty_bytes() -> None:
    with pytest.raises(ValueError, match="raw"):
        PreparedImage(b"", MediaType.JPEG, make_pixels())


@pytest.mark.unit
@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((4, 3, 3), dtype=np.uint16),
        np.zeros((4, 3, 3), dtype=np.float32),
        np.zeros((4, 3), dtype=np.uint8),
        np.zeros((4, 3, 4), dtype=np.uint8),
        np.zeros((0, 3, 3), dtype=np.uint8),
        np.zeros((4, 0, 3), dtype=np.uint8),
    ],
    ids=["uint16", "float32", "grayscale", "rgba", "no-rows", "no-columns"],
)
def test_prepared_image_rejects_malformed_pixels(pixels: NDArray[np.generic]) -> None:
    with pytest.raises(ValueError, match="pixels"):
        PreparedImage(RAW, MediaType.JPEG, pixels)  # pyright: ignore[reportArgumentType]


@pytest.mark.unit
def test_pixels_cannot_be_modified_through_the_image() -> None:
    image = PreparedImage(RAW, MediaType.JPEG, make_pixels())
    with pytest.raises(ValueError, match="read-only"):
        image.pixels[0, 0, 0] = 255


@pytest.mark.unit
def test_images_compare_by_identity_without_inspecting_pixels() -> None:
    image = PreparedImage(RAW, MediaType.JPEG, make_pixels())
    assert image == image
    assert image != PreparedImage(RAW, MediaType.JPEG, make_pixels())
