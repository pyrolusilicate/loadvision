"""Checks that photos decoded by the service validator give the baseline predictions.

The parity test of package v0 decodes files on its own; this one goes through
`imaging.validator.prepare_image`, the path uploaded photos take, so a decoding difference
between the service and training shows up here. Needs `baseline/` and `data/test`.
"""

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from loadvision.imaging.prepared_image import MediaType
from loadvision.imaging.validator import ImageLimits, prepare_image
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.baseline_package import build_baseline_package

BASELINE_DIR = Path("baseline")
TEST_IMAGES = Path("data/test/images")
PHOTOS = 8
TOLERANCE = 1e-6
# Loose limits: this test is about decoding; the service limits are checked by unit tests.
LIMITS = ImageLimits(
    max_bytes=100_000_000,
    max_pixels=500_000_000,
    min_side=1,
    media_types=frozenset(MediaType),
)

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (BASELINE_DIR / "model.npz").is_file() or not TEST_IMAGES.is_dir(),
        reason="baseline/ or data/test is missing",
    ),
]


def test_validator_decoding_gives_the_baseline_predictions(tmp_path: Path) -> None:
    with (BASELINE_DIR / "submission.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))[:PHOTOS]
    package = build_baseline_package(
        BASELINE_DIR,
        tmp_path / "package",
        version="baseline-v0",
        git_commit="0" * 40,
        data_hash="0" * 64,
        created_at=datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
    )
    images = [
        prepare_image((TEST_IMAGES / f"{row['image_id']}.jpg").read_bytes(), LIMITS) for row in rows
    ]

    predictions = PackagePredictor(package).predict(images)

    for row, prediction in zip(rows, predictions, strict=True):
        expected = float(row["load_pct"])
        assert prediction.load_pct.value == pytest.approx(expected, abs=TOLERANCE), row["image_id"]
