"""Draws an explanation map over the photo it was computed from."""

import io
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from loadvision.imaging.prepared_image import PreparedImage
from loadvision.imaging.preprocessing import letterbox_region
from loadvision.ml.inference.explain import Explanation, MapKind

_UP: Final = np.array([230, 50, 40], np.float32)  # red: raises the answer
_DOWN: Final = np.array([40, 110, 235], np.float32)  # blue: lowers it
_LOOK: Final = np.array([255, 205, 40], np.float32)  # yellow: where the model looks
_OPACITY: Final = 0.75


def render(image: PreparedImage, explanation: Explanation, kind: MapKind) -> bytes:
    """Returns the photo with one map drawn over it, as PNG at the photo's own size.

    The map covers the encoder's input box; only the part where the photo sits inside the box
    is kept and stretched back over the photo, so every patch lands on the pixels it saw.

    Args:
        image: The photo that was explained.
        explanation: Its maps.
        kind: Which map to draw.

    Returns:
        PNG bytes.

    Raises:
        ValueError: If the contribution map is asked for but was not computed.
    """
    grid = explanation.attention if kind == "attention" else explanation.contribution
    if grid is None:
        raise ValueError("the contribution map was not computed for this photo")
    # Bicubic stretching overshoots a little past the grid's range; clip it back.
    low = 0.0 if kind == "attention" else -1.0
    values = _over_photo(image, grid, explanation.box_height, explanation.box_width).clip(low, 1.0)
    photo = image.pixels.astype(np.float32)
    if kind == "attention":
        base = photo * 0.35
        strength = values**1.2
        colour = np.broadcast_to(_LOOK, photo.shape)
    else:
        base = photo.mean(-1, keepdims=True) * 0.55 + photo * 0.45
        strength = np.abs(values) ** 0.8
        colour = np.where(values[..., None] > 0, _UP, _DOWN)
    alpha = (strength * _OPACITY)[..., None]
    blended = (base * (1 - alpha) + colour * alpha).clip(0, 255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(blended).save(out, "PNG")
    return out.getvalue()


def _over_photo(
    image: PreparedImage, grid: NDArray[np.float32], box_height: int, box_width: int
) -> NDArray[np.float32]:
    box = Image.fromarray(grid, mode="F").resize((box_width, box_height), Image.Resampling.BICUBIC)
    top, left, height, width = letterbox_region(image.height, image.width, box_height, box_width)
    photo = box.crop((left, top, left + width, top + height))
    resized = photo.resize((image.width, image.height), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32)
