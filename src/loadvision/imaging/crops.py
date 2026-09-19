"""The far end of the truck body, cut out of a photo so an encoder sees it up close."""

from typing import Final

import numpy as np

from loadvision.imaging.prepared_image import PreparedImage

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
