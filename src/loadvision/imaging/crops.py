"""What an encoder is shown: the whole frame, or a part of it cut out and enlarged.

A crop changes how much floor is in frame, so it is never a view in the sense of
`imaging/views.py`, whose transformations leave the answer unchanged. It is a separate input:
a model reads the whole frame and, if it was trained that way, the far end as well, and its
head gets both vectors. The registry keeps the names a model package may refer to, so adding
an input never touches the service.
"""

from collections.abc import Callable, Mapping
from typing import Final

import numpy as np

from loadvision.imaging.prepared_image import PreparedImage

WHOLE: Final = "whole"
FAR_END: Final = "far_end"

# Photos are taken from the doors, so the far wall sits at the vanishing point: in the middle
# of the frame and a little above it. Half of each side around it doubles its size in the box.
_LEFT: Final = 0.25
_RIGHT: Final = 0.75
_TOP: Final = 0.15
_BOTTOM: Final = 0.65


def far_end(image: PreparedImage) -> PreparedImage:
    """Crops the middle of the frame, where the far end of the body is.

    The crop keeps the proportions of the photo, so it fills the same input box as the whole
    frame and goes through the same preprocessing.

    Args:
        image: Decoded photo.

    Returns:
        The crop, with the raw bytes of the original photo.
    """
    height, width = image.pixels.shape[:2]
    pixels = image.pixels[
        round(height * _TOP) : round(height * _BOTTOM), round(width * _LEFT) : round(width * _RIGHT)
    ]
    return PreparedImage(image.raw, image.media_type, np.ascontiguousarray(pixels))


def _whole(image: PreparedImage) -> PreparedImage:
    return image


CROPS: Final[Mapping[str, Callable[[PreparedImage], PreparedImage]]] = {
    WHOLE: _whole,
    FAR_END: far_end,
}


def names() -> tuple[str, ...]:
    """Registry keys of every input, sorted."""
    return tuple(sorted(CROPS))


def get(name: str) -> Callable[[PreparedImage], PreparedImage]:
    """Returns the input by registry key.

    Args:
        name: Registry key.

    Raises:
        KeyError: If the name is not in the registry.
    """
    if name not in CROPS:
        raise KeyError(f"unknown input {name!r}, known: {', '.join(names())}")
    return CROPS[name]
