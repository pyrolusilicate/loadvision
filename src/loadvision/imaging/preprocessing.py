"""Single path from image bytes to model input, shared by training and serving.

A package manifest names a preprocessing by its registry key and passes its numeric
parameters, so training and the service prepare a photo the same way.
"""

from collections.abc import Callable, Mapping
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.imaging.prepared_image import PreparedImage

type Params = Mapping[str, float]
type ModelInput = NDArray[np.floating[Any]]
type Preprocessor = Callable[[PreparedImage, Params], ModelInput]

# Baseline features: 8x8 cells of mean color, 4x4 cells of texture, 8 histogram bins per channel
# and the aspect ratio.
_BASELINE_GRID: Final = 32
_BASELINE_SPATIAL_CELLS: Final = 8
_BASELINE_TEXTURE_CELLS: Final = 4
_BASELINE_HIST_BINS: Final = 8

_CHANNELS: Final = ("r", "g", "b")

_MAX_CHANNEL: Final = 255


def _require(params: Params, name: str) -> float:
    try:
        return float(params[name])
    except KeyError:
        raise LoadvisionError(
            ErrorCode.MODEL_PACKAGE_INVALID, f"preprocessing parameter {name!r} is missing"
        ) from None


def _side(params: Params, name: str) -> int:
    value = _require(params, name)
    side = int(value)
    if side != value or side < 1:
        raise LoadvisionError(
            ErrorCode.MODEL_PACKAGE_INVALID,
            f"preprocessing parameter {name!r} must be a positive integer",
        )
    return side


def _channel_triple(params: Params, prefix: str) -> NDArray[np.float32]:
    return np.array([_require(params, f"{prefix}_{channel}") for channel in _CHANNELS], np.float32)


def resize_pad(image: PreparedImage, params: Params) -> ModelInput:
    """Fits a photo into the model input box without changing its proportions.

    The photo is scaled until it touches the box, the rest of the box stays black, and the
    result is normalized per channel. Proportions are kept because a stretched truck body
    changes the very thing the model measures, the share of the floor taken by cargo.

    Args:
        image: Decoded photo.
        params: `height` and `width` of the box, plus `mean_r`, `mean_g`, `mean_b`, `std_r`,
            `std_g` and `std_b` of the encoder's normalization.

    Returns:
        Float32 array of shape (3, height, width) in channel-first order.

    Raises:
        LoadvisionError: With code `MODEL_PACKAGE_INVALID` if a parameter is missing, a side is
            not a positive integer or a standard deviation is not positive.
    """
    return _letterbox(image, params, fill=None)


def resize_pad_mean(image: PreparedImage, params: Params) -> ModelInput:
    """Fits a photo into the box and fills the rest with the encoder's mean colour.

    Same as `resize_pad` except for what the free space holds. Black padding becomes about
    -2.1 after normalization, a value no real photo of a cargo bay produces, and half of the
    dataset has proportions far enough from the box to leave a wide band of it. The mean colour
    normalizes to exactly zero, which is the neutral input for a pretrained encoder.

    Args:
        image: Decoded photo.
        params: Same as `resize_pad`.

    Returns:
        Float32 array of shape (3, height, width) in channel-first order.

    Raises:
        LoadvisionError: With code `MODEL_PACKAGE_INVALID` if a parameter is missing, a side is
            not a positive integer or a standard deviation is not positive.
    """
    mean = _channel_triple(params, "mean")
    return _letterbox(image, params, fill=np.round(mean * _MAX_CHANNEL).astype(np.uint8))


def letterbox_region(
    image_height: int, image_width: int, box_height: int, box_width: int
) -> tuple[int, int, int, int]:
    """Where a photo lands when it is fitted into the input box without changing proportions.

    Args:
        image_height: Photo height in pixels.
        image_width: Photo width in pixels.
        box_height: Height of the input box.
        box_width: Width of the input box.

    Returns:
        Top, left, height and width of the photo inside the box, in box pixels.
    """
    scale = min(box_width / image_width, box_height / image_height)
    height = max(1, round(image_height * scale))
    width = max(1, round(image_width * scale))
    return (box_height - height) // 2, (box_width - width) // 2, height, width


def _letterbox(
    image: PreparedImage, params: Params, *, fill: NDArray[np.uint8] | None
) -> ModelInput:
    height = _side(params, "height")
    width = _side(params, "width")
    mean = _channel_triple(params, "mean")
    std = _channel_triple(params, "std")
    if not (std > 0).all():
        raise LoadvisionError(ErrorCode.MODEL_PACKAGE_INVALID, "preprocessing std must be positive")

    top, left, inner_height, inner_width = letterbox_region(
        image.height, image.width, height, width
    )
    resized = Image.fromarray(image.pixels).resize(
        (inner_width, inner_height),
        # Bicubic, because every encoder we consider was pretrained with it; baseline_v0 keeps
        # bilinear, since it must reproduce the organizers' reference code byte for byte.
        Image.Resampling.BICUBIC,
    )
    if fill is None:
        canvas = np.zeros((height, width, 3), np.uint8)
    else:
        canvas = np.broadcast_to(fill, (height, width, 3)).copy()
    canvas[top : top + inner_height, left : left + inner_width] = np.asarray(resized)

    normalized = (canvas.astype(np.float32) / _MAX_CHANNEL - mean) / std
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


def baseline_v0(image: PreparedImage, params: Params) -> ModelInput:
    """Repeats the feature vector of the organizers' baseline.

    Kept only for the v0 package, which wraps the baseline model in our package format. It is
    computed in float64 because the baseline head was fitted on float64 features, and v0 has to
    reproduce the baseline predictions exactly.

    Args:
        image: Decoded photo.
        params: Not used.

    Returns:
        Float64 array of 233 features: 192 mean colors, 16 texture values, 24 histogram bins
        and the aspect ratio.
    """
    del params
    small = (
        np.asarray(
            Image.fromarray(image.pixels).resize(
                (_BASELINE_GRID, _BASELINE_GRID), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )
        / 255
    )
    block = _BASELINE_GRID // _BASELINE_SPATIAL_CELLS
    spatial = small.reshape(_BASELINE_SPATIAL_CELLS, block, _BASELINE_SPATIAL_CELLS, block, 3).mean(
        axis=(1, 3)
    )
    texture_block = _BASELINE_GRID // _BASELINE_TEXTURE_CELLS
    gray = small.mean(axis=2)
    texture = gray.reshape(
        _BASELINE_TEXTURE_CELLS, texture_block, _BASELINE_TEXTURE_CELLS, texture_block
    ).std(axis=(1, 3))
    pixels = _BASELINE_GRID * _BASELINE_GRID
    histograms = [
        np.histogram(small[:, :, channel], bins=_BASELINE_HIST_BINS, range=(0, 1))[0] / pixels
        for channel in range(3)
    ]
    aspect = image.width / image.height
    return np.concatenate((spatial.ravel(), texture.ravel(), *histograms, [aspect]))


_REGISTRY: Final[Mapping[str, Preprocessor]] = {
    "resize_pad": resize_pad,
    "resize_pad_mean": resize_pad_mean,
    "baseline_v0": baseline_v0,
}


def names() -> tuple[str, ...]:
    """Returns the registry keys a manifest may name, sorted."""
    return tuple(sorted(_REGISTRY))


def preprocess(name: str, image: PreparedImage, params: Params) -> ModelInput:
    """Prepares a photo with the preprocessing a manifest names.

    Args:
        name: Registry key from the manifest.
        image: Decoded photo.
        params: Parameters from the manifest.

    Returns:
        The model input of that preprocessing.

    Raises:
        LoadvisionError: With code `MODEL_PACKAGE_INVALID` if the key is unknown or a parameter
            is wrong.
    """
    preprocessor = _REGISTRY.get(name)
    if preprocessor is None:
        raise LoadvisionError(
            ErrorCode.MODEL_PACKAGE_INVALID,
            f"unknown preprocessing {name!r}, known: {', '.join(names())}",
        )
    return preprocessor(image, params)
