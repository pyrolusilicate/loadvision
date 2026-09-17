from datetime import datetime, timedelta

import pytest

from loadvision.application.ports import LabeledImage
from loadvision.domain.assessment import LoadPercent
from loadvision.domain.confirmed_label import ConfirmedLabel, LabelSource
from tests.builders import T0, make_label


@pytest.mark.unit
def test_first_label_is_created_and_updated_at_once() -> None:
    label = make_label(load_pct=70.0)
    assert (label.created_at, label.updated_at) == (T0, T0)
    assert label.load_pct == LoadPercent(70.0)
    assert label.unusable is False


@pytest.mark.unit
def test_label_without_load_marks_the_photo_unusable() -> None:
    assert make_label(load_pct=None).unusable is True


@pytest.mark.unit
def test_replaced_label_keeps_its_creation_time() -> None:
    later = T0 + timedelta(hours=1)
    label = make_label(load_pct=70.0).replaced(
        load_pct=None, source=LabelSource.UI, author=None, at=later
    )
    assert (label.created_at, label.updated_at) == (T0, later)
    assert (label.load_pct, label.source, label.author) == (None, LabelSource.UI, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"image_sha256": "A" * 64}, "image_sha256"),
        ({"image_sha256": "a" * 63}, "image_sha256"),
        ({"author": "  "}, "author"),
        ({"created_at": datetime(2026, 9, 16, 12, 0)}, "created_at"),
        ({"updated_at": datetime(2026, 9, 16, 12, 0)}, "updated_at"),
        ({"updated_at": T0 - timedelta(seconds=1)}, "before created_at"),
    ],
)
def test_invalid_label_is_rejected(overrides: dict[str, object], match: str) -> None:
    fields: dict[str, object] = {
        "image_sha256": "a" * 64,
        "load_pct": LoadPercent(50.0),
        "source": LabelSource.API,
        "author": "operator-17",
        "created_at": T0,
        "updated_at": T0,
    }
    with pytest.raises(ValueError, match=match):
        ConfirmedLabel(**(fields | overrides))  # type: ignore[arg-type]


@pytest.mark.unit
def test_usable_label_becomes_a_training_example() -> None:
    label = make_label(load_pct=70.0)
    example = LabeledImage.from_label(label)
    assert (example.image_sha256, example.load_pct) == (label.image_sha256, LoadPercent(70.0))


@pytest.mark.unit
def test_unusable_label_is_not_a_training_example() -> None:
    with pytest.raises(ValueError, match="unusable"):
        LabeledImage.from_label(make_label(load_pct=None))
