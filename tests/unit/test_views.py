import numpy as np
import pytest

from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.views import IDENTITY, VIEWS, get, names


def photo(width: int = 4, height: int = 3) -> PreparedImage:
    """A photo whose left half differs from its right, so a mirror is visible."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, : width // 2] = 200
    return PreparedImage(b"photo", MediaType.JPEG, pixels)


@pytest.mark.unit
def test_the_registry_lists_its_views() -> None:
    assert names() == ("identity", "mirror")
    assert set(VIEWS) == set(names())


@pytest.mark.unit
def test_identity_returns_the_same_pixels() -> None:
    image = photo()

    assert np.array_equal(get(IDENTITY).make(image).pixels, image.pixels)
    assert not get(IDENTITY).mirrored


@pytest.mark.unit
def test_mirror_swaps_left_and_right() -> None:
    image = photo()

    mirrored = get("mirror").make(image)

    assert np.array_equal(mirrored.pixels, image.pixels[:, ::-1])
    assert get("mirror").mirrored
    # The bright half moved from the left to the right.
    assert mirrored.pixels[0, 0, 0] == 0
    assert mirrored.pixels[0, -1, 0] == 200


@pytest.mark.unit
def test_a_view_keeps_the_original_bytes_and_hash() -> None:
    image = photo()

    mirrored = get("mirror").make(image)

    # The photo the service stored is the one it received; a view changes only what the
    # encoder looks at.
    assert mirrored.raw == image.raw
    assert mirrored.sha256 == image.sha256


@pytest.mark.unit
def test_mirroring_twice_returns_the_original() -> None:
    image = photo()
    once = get("mirror").make(image)

    twice = get("mirror").make(once)

    assert np.array_equal(twice.pixels, image.pixels)


@pytest.mark.unit
def test_an_unknown_view_is_refused_by_name() -> None:
    with pytest.raises(KeyError, match="unknown view 'zoom'"):
        get("zoom")
