"""Image files for tests, built in memory: valid photos in every format and broken ones."""

import io
import struct
import zlib
from typing import Any

import numpy as np
from PIL import Image

EXIF_ORIENTATION_TAG = 0x0112
# EXIF orientation 6: the camera was turned, the photo must be rotated 90 degrees clockwise.
ROTATE_CLOCKWISE = 6


def picture(width: int = 300, height: int = 400, mode: str = "RGB") -> Image.Image:
    """Returns a gradient picture: every row and column differs, so a rotation is visible."""
    x = np.linspace(0, 255, width, dtype=np.uint8)[np.newaxis, :].repeat(height, axis=0)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, np.newaxis].repeat(width, axis=1)
    rgb = Image.fromarray(np.dstack([x, y, 255 - x]))
    return rgb.convert(mode)


def encode(image: Image.Image, image_format: str, **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


def jpeg(
    width: int = 300, height: int = 400, *, mode: str = "RGB", orientation: int | None = None
) -> bytes:
    options: dict[str, Any] = {"quality": 95}
    if orientation is not None:
        exif = Image.Exif()
        exif[EXIF_ORIENTATION_TAG] = orientation
        options["exif"] = exif
    return encode(picture(width, height, mode), "JPEG", **options)


def dark_jpeg(width: int = 300, height: int = 400, *, value: int = 5) -> bytes:
    """Returns a photo shot in the dark: one dim colour, nothing to make out."""
    return encode(Image.new("RGB", (width, height), (value, value, value)), "JPEG", quality=95)


def png(width: int = 300, height: int = 400, *, mode: str = "RGB") -> bytes:
    return encode(picture(width, height, mode), "PNG")


def webp(width: int = 300, height: int = 400) -> bytes:
    return encode(picture(width, height), "WEBP", lossless=True)


def gif(width: int = 300, height: int = 400) -> bytes:
    return encode(picture(width, height), "GIF")


def heic_header() -> bytes:
    """Returns the start of a HEIC file: an ISO base media `ftyp` box with the `heic` brand."""
    return struct.pack(">I", 24) + b"ftypheic" + struct.pack(">I", 0) + b"mif1heic" + bytes(64)


def png_header(width: int, height: int) -> bytes:
    """Returns a PNG that declares a size in its header but holds almost no pixel data."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )


def truncated(data: bytes) -> bytes:
    """Returns the first half of a file, as after an interrupted upload."""
    return data[: len(data) // 2]
