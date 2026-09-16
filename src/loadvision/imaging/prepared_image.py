"""Decoded photo passed from image validation to storage and to the model."""

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class MediaType(StrEnum):
    """Image formats the service accepts, as detected from the file contents."""

    JPEG = "image/jpeg"
    PNG = "image/png"
    WEBP = "image/webp"


# eq=False: field-by-field equality would compare numpy arrays, which yields an array, not a bool.
@dataclass(frozen=True, slots=True, eq=False)
class PreparedImage:
    """Photo decoded to RGB pixels, together with its original bytes.

    This is the decoded photo, not the model input: resizing and normalization happen in
    `imaging.preprocessing` with parameters from the model package manifest.

    Attributes:
        raw: Original file bytes, stored as they came.
        media_type: Format detected from the file contents.
        pixels: Read-only uint8 RGB array of shape (height, width, 3) at the original size,
            with EXIF rotation applied.
        sha256: Lowercase hex SHA-256 of `raw`, computed here so it always matches the bytes.
    """

    raw: bytes
    media_type: MediaType
    pixels: NDArray[np.uint8]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validates the pixels, locks them against writes and computes the hash.

        Raises:
            ValueError: If `raw` is empty or `pixels` is not a non-empty (H, W, 3) uint8 array.
        """
        if not self.raw:
            raise ValueError("raw must not be empty")
        shape = self.pixels.shape
        if self.pixels.dtype != np.uint8 or len(shape) != 3 or shape[2] != 3 or 0 in shape:
            raise ValueError(
                f"pixels must be a non-empty (H, W, 3) uint8 array, got {self.pixels.dtype} {shape}"
            )
        # Ensemble members read the same image; one normalizing it in place would corrupt the
        # input of the next, so a write fails loudly instead.
        pixels = self.pixels.view()
        pixels.flags.writeable = False
        object.__setattr__(self, "pixels", pixels)
        object.__setattr__(self, "sha256", hashlib.sha256(self.raw).hexdigest())

    @property
    def height(self) -> int:
        """Photo height in pixels."""
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        """Photo width in pixels."""
        return self.pixels.shape[1]
