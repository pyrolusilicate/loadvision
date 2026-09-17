import numpy as np
import pytest
from numpy.typing import NDArray

from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.quality import brightness, sharpness


def photo(pixels: NDArray[np.uint8]) -> PreparedImage:
    return PreparedImage(b"photo", MediaType.JPEG, pixels)


def flat(value: int, height: int = 64, width: int = 48) -> PreparedImage:
    return photo(np.full((height, width, 3), value, dtype=np.uint8))


def checkerboard(height: int = 64, width: int = 48) -> PreparedImage:
    rows, cols = np.indices((height, width))
    pattern = np.where((rows + cols) % 2 == 0, 255, 0).astype(np.uint8)
    return photo(np.repeat(pattern[:, :, None], 3, axis=2))


@pytest.mark.unit
@pytest.mark.parametrize(("value", "expected"), [(0, 0.0), (255, 255.0), (128, 128.0)])
def test_brightness_of_a_flat_photo_is_its_value(value: int, expected: float) -> None:
    assert brightness(flat(value)) == pytest.approx(expected, abs=0.01)


@pytest.mark.unit
def test_brightness_follows_the_eye_not_the_channels() -> None:
    def single(channel: int) -> PreparedImage:
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[:, :, channel] = 255
        return photo(pixels)

    green, blue = single(1), single(2)

    # Green looks much brighter than blue at the same channel value, and the signal must agree.
    assert brightness(green) > brightness(blue)


@pytest.mark.unit
def test_a_flat_photo_has_no_sharpness() -> None:
    assert sharpness(flat(128)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_a_sharp_photo_scores_higher_than_a_blurred_one() -> None:
    sharp = checkerboard()
    # Averaging neighbouring pixels is what blur does.
    pixels = sharp.pixels.astype(np.float64)
    blurred = (pixels + np.roll(pixels, 1, axis=0) + np.roll(pixels, 1, axis=1)) / 3

    assert sharpness(sharp) > sharpness(photo(blurred.astype(np.uint8)))


@pytest.mark.unit
def test_signals_never_go_negative() -> None:
    assert brightness(flat(0)) >= 0.0
    assert sharpness(flat(0)) >= 0.0


@pytest.mark.unit
def test_a_photo_smaller_than_the_window_has_no_sharpness() -> None:
    assert sharpness(flat(200, height=2, width=2)) == 0.0


@pytest.mark.unit
def test_signals_do_not_depend_much_on_the_size_of_the_photo() -> None:
    small = checkerboard(64, 48)
    large = checkerboard(256, 192)

    assert brightness(small) == pytest.approx(brightness(large), abs=2.0)
