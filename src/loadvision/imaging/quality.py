"""Brightness and sharpness of a photo, the trust signals that need no model.

Both are read from a small thumbnail: the answers describe the whole frame, and a thumbnail
makes them cost microseconds instead of milliseconds on a photo of several megapixels.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from loadvision.imaging.prepared_image import PreparedImage

THUMBNAIL: Final = 32
MAX_CHANNEL: Final = 255.0

# The Laplacian needs a pixel on every side of the one it looks at.
_MIN_SIDE: Final = 3


def brightness(image: PreparedImage) -> float:
    """Mean brightness of a photo, from 0 for black to 255 for white.

    Args:
        image: Decoded photo.

    Returns:
        Mean brightness.
    """
    return float(_grey(image).mean())


def sharpness(image: PreparedImage) -> float:
    """How sharp a photo is: the spread of its second derivative, never negative.

    A blurred photo has soft edges, so the differences between neighbouring pixels change
    slowly and their second derivative stays near zero. The number has no unit; only its size
    relative to other photos of the dataset means anything, which is why the review threshold
    is chosen from the data rather than written down.

    Args:
        image: Decoded photo.

    Returns:
        Standard deviation of the discrete Laplacian of the thumbnail.
    """
    grey = _grey(image)
    if min(grey.shape) < _MIN_SIDE:
        return 0.0
    # Discrete Laplacian: four neighbours minus four times the centre.
    laplacian = (
        grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:] - 4 * grey[1:-1, 1:-1]
    )
    return float(laplacian.std())


def _grey(image: PreparedImage) -> NDArray[np.float64]:
    # Averaging, not picking every Nth pixel: on a fine pattern a stride would return whatever
    # parity it happened to land on. The same thumbnail the dataset manifest is built from.
    side = min(THUMBNAIL, image.height, image.width)
    thumbnail = Image.fromarray(image.pixels).convert("L").resize((side, side))
    return np.asarray(thumbnail, dtype=np.float64)
