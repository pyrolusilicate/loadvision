import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageOps

from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.imaging.prepared_image import MediaType
from loadvision.imaging.validator import ImageLimits, detect_media_type, prepare_image
from tests import image_files

# Limits of the tests themselves, small enough to reach every boundary quickly; the service
# limits live in configs/app.yaml.
LIMITS = ImageLimits(
    max_bytes=2_000_000,
    max_pixels=1_000_000,
    min_side=256,
    media_types=frozenset(MediaType),
)


def rejection(raw: bytes, limits: ImageLimits = LIMITS) -> LoadvisionError:
    with pytest.raises(LoadvisionError) as caught:
        prepare_image(raw, limits)
    return caught.value


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "media_type"),
    [
        (image_files.jpeg(), MediaType.JPEG),
        (image_files.png(), MediaType.PNG),
        (image_files.webp(), MediaType.WEBP),
    ],
    ids=["jpeg", "png", "webp"],
)
def test_accepts_supported_format(raw: bytes, media_type: MediaType) -> None:
    image = prepare_image(raw, LIMITS)
    assert image.media_type is media_type
    assert image.pixels.shape == (400, 300, 3)
    assert image.raw == raw


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        image_files.jpeg(mode="CMYK"),
        image_files.jpeg(mode="L"),
        image_files.png(mode="RGBA"),
        image_files.png(mode="P"),
    ],
    ids=["cmyk-jpeg", "grayscale-jpeg", "rgba-png", "palette-png"],
)
def test_other_color_modes_become_rgb(raw: bytes) -> None:
    assert prepare_image(raw, LIMITS).pixels.shape == (400, 300, 3)


@pytest.mark.unit
def test_exif_rotation_is_applied_like_in_training() -> None:
    raw = image_files.jpeg(300, 400, orientation=image_files.ROTATE_CLOCKWISE)
    image = prepare_image(raw, LIMITS)
    with Image.open(io.BytesIO(raw)) as source:
        expected = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    assert (image.width, image.height) == (400, 300)
    assert np.array_equal(image.pixels, expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "media_type"),
    [
        (b"\xff\xd8\xff\xe0rest", MediaType.JPEG),
        (b"\x89PNG\r\n\x1a\nrest", MediaType.PNG),
        (b"RIFF\x00\x00\x00\x00WEBPrest", MediaType.WEBP),
        (b"RIFF\x00\x00\x00\x00WAVErest", None),
        (b"GIF89a", None),
        (b"\xff\xd8", None),
        (b"", None),
    ],
)
def test_detects_format_by_signature(raw: bytes, media_type: MediaType | None) -> None:
    assert detect_media_type(raw) is media_type


@pytest.mark.unit
def test_empty_file_is_corrupted() -> None:
    error = rejection(b"")
    assert (error.code, error.detail) == (ErrorCode.CORRUPTED_IMAGE, "empty file")


@pytest.mark.unit
def test_file_larger_than_the_limit_is_too_large() -> None:
    raw = image_files.png()
    assert prepare_image(raw, replace(LIMITS, max_bytes=len(raw))).raw == raw
    error = rejection(raw, replace(LIMITS, max_bytes=len(raw) - 1))
    assert error.code is ErrorCode.PAYLOAD_TOO_LARGE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "detail"),
    [
        (image_files.heic_header(), "HEIC/HEIF photo"),
        (image_files.gif(), "unknown file signature"),
        (b"%PDF-1.7 not a photo", "unknown file signature"),
    ],
    ids=["heic", "gif", "pdf"],
)
def test_unsupported_format_is_named_in_the_detail(raw: bytes, detail: str) -> None:
    error = rejection(raw)
    assert (error.code, error.detail) == (ErrorCode.UNSUPPORTED_FORMAT, detail)


@pytest.mark.unit
def test_format_outside_the_accepted_list_is_unsupported() -> None:
    limits = replace(LIMITS, media_types=frozenset({MediaType.JPEG}))
    assert rejection(image_files.png(), limits).code is ErrorCode.UNSUPPORTED_FORMAT


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        image_files.truncated(image_files.jpeg()),
        image_files.truncated(image_files.png()),
        image_files.truncated(image_files.webp()),
        b"\xff\xd8\xff\xe0" + bytes(500),
        image_files.png()[:60],
    ],
    ids=["truncated-jpeg", "truncated-png", "truncated-webp", "jpeg-signature-only", "png-header"],
)
def test_broken_file_is_corrupted(raw: bytes) -> None:
    assert rejection(raw).code is ErrorCode.CORRUPTED_IMAGE


@pytest.mark.unit
@pytest.mark.parametrize(("width", "height"), [(256, 256), (1000, 1000)])
def test_accepts_size_at_the_limits(width: int, height: int) -> None:
    assert prepare_image(image_files.png(width, height), LIMITS).pixels.shape == (height, width, 3)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("width", "height", "detail"),
    [
        (255, 400, "255x400 has a side shorter than 256 pixels"),
        (400, 255, "400x255 has a side shorter than 256 pixels"),
        (1, 1, "1x1 has a side shorter than 256 pixels"),
        (1000, 1001, "1000x1001 is more than 1000000 pixels"),
    ],
)
def test_size_outside_the_limits_is_corrupted(width: int, height: int, detail: str) -> None:
    error = rejection(image_files.png(width, height))
    assert (error.code, error.detail) == (ErrorCode.CORRUPTED_IMAGE, detail)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("side", "detail"),
    [
        (2_000, "2000x2000 is more than 1000000 pixels"),
        (30_000, "DecompressionBombError"),
    ],
    ids=["over-our-limit", "over-pillow-guard"],
)
def test_pixel_bomb_is_refused_by_its_header(side: int, detail: str) -> None:
    # The file declares a huge size but holds no pixel data: a refusal that names the size
    # proves the header was checked before decoding.
    error = rejection(image_files.png_header(side, side))
    assert error.code is ErrorCode.CORRUPTED_IMAGE
    assert detail in str(error.detail)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["max_bytes", "max_pixels", "min_side"])
def test_limits_must_be_positive(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        replace(LIMITS, **{name: 0})


@pytest.mark.unit
def test_limits_must_accept_some_format() -> None:
    with pytest.raises(ValueError, match="media_types"):
        replace(LIMITS, media_types=frozenset())
