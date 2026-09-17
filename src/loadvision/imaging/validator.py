"""Validation of uploaded image bytes: format, size and decodability."""

import io
import struct
from dataclasses import dataclass, fields
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.imaging.prepared_image import MediaType, PreparedImage

# ISO base media brands of HEIC and HEIF files, the default photo format of iPhone cameras.
_HEIF_BRANDS: Final = frozenset(
    {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}
)

# What Pillow raises on broken or hostile files.
_DECODE_ERRORS: Final = (
    OSError,
    ValueError,
    SyntaxError,
    EOFError,
    struct.error,
    Image.DecompressionBombError,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageLimits:
    """Limits an uploaded photo must fit; the values come from `configs/app.yaml`.

    Attributes:
        max_bytes: Largest accepted file size.
        max_pixels: Largest accepted width times height, checked before decoding.
        min_side: Smallest accepted shorter side, in pixels.
        media_types: Accepted formats.
    """

    max_bytes: int
    max_pixels: int
    min_side: int
    media_types: frozenset[MediaType]

    def __post_init__(self) -> None:
        """Validates the limits.

        Raises:
            ValueError: If a numeric limit is not positive or no format is accepted.
        """
        for limit in fields(self):
            value = getattr(self, limit.name)
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{limit.name} must be positive, got {value}")
        if not self.media_types:
            raise ValueError("media_types must not be empty")


def detect_media_type(raw: bytes) -> MediaType | None:
    """Recognizes a supported format by the first bytes of a file.

    Args:
        raw: File bytes.

    Returns:
        The format, or None if the bytes start like no supported format.
    """
    if raw.startswith(b"\xff\xd8\xff"):
        return MediaType.JPEG
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return MediaType.PNG
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return MediaType.WEBP
    return None


def prepare_image(raw: bytes, limits: ImageLimits) -> PreparedImage:
    """Checks uploaded bytes and decodes them into RGB pixels.

    Pixels are produced exactly as training and the baseline read files,
    ``ImageOps.exif_transpose(image).convert("RGB")``, so the service and a model package see
    the same photo.

    Args:
        raw: File bytes as uploaded.
        limits: Size and format limits.

    Returns:
        The decoded photo with its original bytes.

    Raises:
        LoadvisionError: `PAYLOAD_TOO_LARGE` if the file is larger than `max_bytes`;
            `UNSUPPORTED_FORMAT` if it is not an accepted JPEG, PNG or WebP, HEIC included;
            `CORRUPTED_IMAGE` if it is empty, does not decode completely or its size is
            outside the limits.
    """
    if not raw:
        raise LoadvisionError(ErrorCode.CORRUPTED_IMAGE, "empty file")
    if len(raw) > limits.max_bytes:
        raise LoadvisionError(
            ErrorCode.PAYLOAD_TOO_LARGE, f"{len(raw)} bytes, limit {limits.max_bytes}"
        )
    media_type = detect_media_type(raw)
    if media_type is None:
        detail = "HEIC/HEIF photo" if _is_heif(raw) else "unknown file signature"
        raise LoadvisionError(ErrorCode.UNSUPPORTED_FORMAT, detail)
    if media_type not in limits.media_types:
        raise LoadvisionError(ErrorCode.UNSUPPORTED_FORMAT, f"{media_type} is not accepted")
    try:
        pixels = _decode(raw, limits)
    except _DECODE_ERRORS as error:
        raise LoadvisionError(
            ErrorCode.CORRUPTED_IMAGE, f"{type(error).__name__}: {error}"
        ) from error
    return PreparedImage(raw, media_type, pixels)


def _is_heif(raw: bytes) -> bool:
    return raw[4:8] == b"ftyp" and raw[8:12] in _HEIF_BRANDS


def _decode(raw: bytes, limits: ImageLimits) -> NDArray[np.uint8]:
    with Image.open(io.BytesIO(raw)) as image:
        # The header is read, the pixels are not: a hostile size is refused before any
        # memory is spent on decoding.
        width, height = image.size
        if width * height > limits.max_pixels:
            raise LoadvisionError(
                ErrorCode.CORRUPTED_IMAGE,
                f"{width}x{height} is more than {limits.max_pixels} pixels",
            )
        if min(width, height) < limits.min_side:
            raise LoadvisionError(
                ErrorCode.CORRUPTED_IMAGE,
                f"{width}x{height} has a side shorter than {limits.min_side} pixels",
            )
        # Rotation and conversion decode every pixel, so a truncated file fails here, not in
        # the model.
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(rgb, dtype=np.uint8)
