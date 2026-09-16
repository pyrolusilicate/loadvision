"""Checks that package v0 reproduces the organizers' baseline on the real photos.

Needs `baseline/` and `data/test`, which are not in the repository, so the test is marked
`data` and is skipped where they are missing.
"""

import csv
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.baseline_package import build_baseline_package

BASELINE_DIR = Path("baseline")
TEST_IMAGES = Path("data/test/images")
PHOTOS = 8
TOLERANCE = 1e-6

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (BASELINE_DIR / "model.npz").is_file() or not TEST_IMAGES.is_dir(),
        reason="baseline/ or data/test is missing",
    ),
]


def load_photo(image_id: str) -> PreparedImage:
    path = TEST_IMAGES / f"{image_id}.jpg"
    raw = path.read_bytes()
    with Image.open(path) as source:
        pixels = np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
    return PreparedImage(raw, MediaType.JPEG, pixels)


def test_package_v0_repeats_the_baseline_predictions(tmp_path: Path) -> None:
    with (BASELINE_DIR / "submission.csv").open(encoding="utf-8-sig", newline="") as file:
        expected = [(row["image_id"], float(row["load_pct"])) for row in csv.DictReader(file)][
            :PHOTOS
        ]

    package = build_baseline_package(
        BASELINE_DIR,
        tmp_path / "package",
        version="baseline-v0",
        git_commit="0" * 40,
        data_hash="0" * 64,
        created_at=datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
    )
    predictions = PackagePredictor(package).predict(
        [load_photo(image_id) for image_id, _ in expected]
    )

    for (image_id, baseline_value), prediction in zip(expected, predictions, strict=True):
        assert prediction.load_pct.value == pytest.approx(baseline_value, abs=TOLERANCE), image_id
