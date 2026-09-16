from datetime import UTC, datetime

import numpy as np
import pytest

from loadvision.application.ports import ImageRecord
from loadvision.imaging.prepared_image import MediaType, PreparedImage

RAW = b"\x89PNG bytes of a photo"
NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def make_image() -> PreparedImage:
    return PreparedImage(RAW, MediaType.PNG, np.zeros((4, 3, 3), dtype=np.uint8))


@pytest.mark.unit
def test_record_takes_metadata_from_the_photo() -> None:
    image = make_image()
    assert ImageRecord.from_prepared(image, NOW) == ImageRecord(
        sha256=image.sha256,
        media_type=MediaType.PNG,
        width=3,
        height=4,
        size_bytes=len(RAW),
        created_at=NOW,
    )


@pytest.mark.unit
def test_record_rejects_naive_time() -> None:
    with pytest.raises(ValueError, match="created_at"):
        ImageRecord.from_prepared(make_image(), datetime(2026, 9, 16, 12, 0))
