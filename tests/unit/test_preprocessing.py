import hashlib

import numpy as np
import pytest
from numpy.typing import NDArray

from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.preprocessing import names, preprocess

BOX = {
    "height": 8,
    "width": 6,
    "mean_r": 0.5,
    "mean_g": 0.5,
    "mean_b": 0.5,
    "std_r": 0.5,
    "std_g": 0.5,
    "std_b": 0.5,
}


def make_image(height: int, width: int, value: int = 255) -> PreparedImage:
    pixels: NDArray[np.uint8] = np.full((height, width, 3), value, dtype=np.uint8)
    return PreparedImage(b"photo", MediaType.JPEG, pixels)


@pytest.mark.unit
def test_registry_lists_both_preprocessings() -> None:
    assert names() == ("baseline_v0", "resize_pad", "resize_pad_mean")


@pytest.mark.unit
def test_unknown_preprocessing_names_the_known_ones() -> None:
    with pytest.raises(LoadvisionError) as error:
        preprocess("resize_pad_3x4", make_image(4, 3), BOX)
    assert error.value.code is ErrorCode.MODEL_PACKAGE_INVALID
    assert "baseline_v0" in str(error.value)


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["height", "std_b"])
def test_resize_pad_reports_a_missing_parameter(missing: str) -> None:
    params = {key: value for key, value in BOX.items() if key != missing}
    with pytest.raises(LoadvisionError, match=missing):
        preprocess("resize_pad", make_image(4, 3), params)


@pytest.mark.unit
@pytest.mark.parametrize(("key", "value"), [("height", 0), ("width", 6.5), ("std_r", 0.0)])
def test_resize_pad_rejects_a_bad_parameter(key: str, value: float) -> None:
    with pytest.raises(LoadvisionError):
        preprocess("resize_pad", make_image(4, 3), {**BOX, key: value})


@pytest.mark.unit
def test_resize_pad_returns_the_box_in_channel_first_order() -> None:
    result = preprocess("resize_pad", make_image(40, 30), BOX)
    assert result.shape == (3, 8, 6)
    assert result.dtype == np.float32


@pytest.mark.unit
def test_resize_pad_normalizes_the_photo() -> None:
    result = preprocess("resize_pad", make_image(40, 30), BOX)
    # The photo is white and fills the box exactly: (1.0 - 0.5) / 0.5.
    assert np.allclose(result, 1.0)


@pytest.mark.unit
def test_resize_pad_keeps_proportions_and_pads_the_rest() -> None:
    # A wide photo in a tall box: the middle rows hold the photo, the rest stays padding.
    result = preprocess("resize_pad", make_image(10, 30), BOX)
    rows = result[0].mean(axis=1)
    padding = (0.0 - 0.5) / 0.5
    assert rows[0] == pytest.approx(padding)
    assert rows[-1] == pytest.approx(padding)
    assert rows[4] == pytest.approx(1.0)


@pytest.mark.unit
def test_baseline_v0_returns_the_233_features_of_the_baseline() -> None:
    result = preprocess("baseline_v0", make_image(40, 30), {})
    assert result.shape == (233,)
    assert result.dtype == np.float64
    # Last feature is the aspect ratio of the original photo.
    assert result[-1] == pytest.approx(30 / 40)


@pytest.mark.unit
def test_preprocessing_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    pixels: NDArray[np.uint8] = rng.integers(0, 256, (37, 23, 3), dtype=np.uint8)
    image = PreparedImage(b"photo", MediaType.JPEG, pixels)
    digests = {
        name: hashlib.sha256(
            np.ascontiguousarray(preprocess(name, image, {} if name == "baseline_v0" else BOX))
        ).hexdigest()
        for name in names()
    }
    # Frozen on 16.09.2026: a change here means the model input changed, and every cached
    # embedding and every trained head goes stale.
    assert digests == {
        "baseline_v0": "c0dfeca716ff61a1e3de9d7e3194faa09d1ff2370f19adee3b03360a4786cf15",
        "resize_pad": "2afb83023ccd03e1b366401ecb430645750818dd35d6fea9308a19a5b71e4ba5",
        "resize_pad_mean": "73944473d99dc27d80d8b326ac4b6965962d62e1ddfd52c7720e8783db8249ba",
    }


@pytest.mark.unit
def test_mean_fill_leaves_the_padding_at_zero() -> None:
    # A photo far from the box proportions: a wide band of the input is padding.
    pixels: NDArray[np.uint8] = np.full((10, 40, 3), 200, dtype=np.uint8)
    image = PreparedImage(b"photo", MediaType.JPEG, pixels)

    black = preprocess("resize_pad", image, BOX)
    neutral = preprocess("resize_pad_mean", image, BOX)

    # The top row of the box is padding in both. Black normalizes to a value no real photo
    # produces; the mean colour normalizes to the neutral zero.
    # With this box mean and std, black lands exactly on -1; what matters is that it is far
    # from the neutral zero a pretrained encoder expects where nothing was photographed.
    assert float(black[:, 0, 0].max()) < -0.5
    assert abs(float(neutral[:, 0, 0].max())) < 0.01
    # The photo itself lands identically either way.
    middle = neutral.shape[1] // 2
    assert np.allclose(black[:, middle, :], neutral[:, middle, :])
