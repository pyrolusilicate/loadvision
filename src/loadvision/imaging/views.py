"""Views of one photo that an encoder averages over, known as test-time augmentation.

A view is a transformation that leaves the answer unchanged. The share of the floor taken by
cargo does not care whether the photo is mirrored, so the mirror is a second look at the same
scene, and averaging the two embeddings cancels part of the encoder's own noise.

Views live in a registry like preprocessing and encoders do, so turning them on is a name in a
registry entry rather than an edit inside training or inside the service. A view that changed
the answer, such as a crop, would belong nowhere near here: cropping changes how much floor is
in frame, which is the very thing the label measures.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from loadvision.imaging.prepared_image import PreparedImage

IDENTITY: Final = "identity"


@dataclass(frozen=True, slots=True, kw_only=True)
class View:
    """One look at a photo.

    Attributes:
        name: Registry key.
        make: Returns the transformed photo.
        mirrored: Whether the view swaps left and right. An encoder that keeps a spatial grid
            has to put the columns back in order before averaging, otherwise the left of the
            body would be averaged with its right and the grid would lose the very geometry it
            exists for.
    """

    name: str
    make: Callable[[PreparedImage], PreparedImage]
    mirrored: bool = False


def _identity(image: PreparedImage) -> PreparedImage:
    return image


def _mirror(image: PreparedImage) -> PreparedImage:
    pixels = np.ascontiguousarray(image.pixels[:, ::-1])
    return PreparedImage(image.raw, image.media_type, pixels)


VIEWS: Final[Mapping[str, View]] = {
    IDENTITY: View(name=IDENTITY, make=_identity),
    "mirror": View(name="mirror", make=_mirror, mirrored=True),
}


def names() -> tuple[str, ...]:
    """Registry keys of every view, sorted."""
    return tuple(sorted(VIEWS))


def get(name: str) -> View:
    """Returns a view by registry key.

    Args:
        name: Registry key.

    Returns:
        The view.

    Raises:
        KeyError: If the name is not in the registry.
    """
    if name not in VIEWS:
        raise KeyError(f"unknown view {name!r}, known: {', '.join(names())}")
    return VIEWS[name]
