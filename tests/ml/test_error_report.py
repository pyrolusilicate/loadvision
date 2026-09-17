from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from loadvision.ml.data.manifest import Sample
from loadvision.ml.evaluation.ensemble import Member
from loadvision.ml.evaluation.error_report import (
    WORST_SHOWN,
    brightness_of,
    bucket_mae,
    build_report,
    kurtosis,
    label_bin,
    light_bin,
    skewness,
)

MEMBERS = (Member(encoder="siglip2_vitb16", alpha=300.0),)


def sample(image_id: str, label: float, group: str, path: Path | None = None) -> Sample:
    return Sample(
        image_id=image_id,
        path=path or Path(f"{image_id}.jpg"),
        sha256="a" * 64,
        label=label,
        group_id=group,
    )


@pytest.mark.ml
def test_symmetric_errors_have_no_skew_and_light_tails() -> None:
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert skewness(values) == pytest.approx(0.0)
    assert kurtosis(values) < 0.0


@pytest.mark.ml
def test_a_long_right_tail_shows_up_as_positive_skew() -> None:
    values = np.array([-1.0] * 20 + [20.0])
    assert skewness(values) > 1.0
    assert kurtosis(values) > 1.0


@pytest.mark.ml
def test_shape_numbers_match_the_definition() -> None:
    # mean 1, so the centred values are -1, -1, -1, -1, 4: m2 = 4, m3 = 12, m4 = 52.
    values = np.array([0.0, 0.0, 0.0, 0.0, 5.0])
    assert skewness(values) == pytest.approx(12 / 4**1.5)
    assert kurtosis(values) == pytest.approx(52 / 4**2 - 3)


@pytest.mark.ml
def test_constant_errors_have_no_shape() -> None:
    values = np.zeros(5)
    assert (skewness(values), kurtosis(values)) == (0.0, 0.0)


@pytest.mark.ml
@pytest.mark.parametrize(
    ("value", "name"),
    [(0.0, "0-9"), (9.9, "0-9"), (10.0, "10-19"), (99.9, "90-99"), (100.0, "100")],
)
def test_labels_fall_into_ten_point_buckets(value: float, name: str) -> None:
    assert label_bin(value) == name


@pytest.mark.ml
@pytest.mark.parametrize(
    ("value", "name"), [(0.0, "0-40"), (39.9, "0-40"), (40.0, "40-60"), (255.0, "100+")]
)
def test_brightness_falls_into_buckets(value: float, name: str) -> None:
    assert light_bin(value) == name


@pytest.mark.ml
def test_buckets_keep_the_asked_for_order_and_drop_empty_ones() -> None:
    errors = np.array([2.0, 4.0, 10.0])
    assert bucket_mae(["b", "b", "a"], errors, ["a", "b", "c"]) == [("a", 10.0), ("b", 3.0)]


@pytest.mark.ml
def test_brightness_is_measured_on_the_photo(tmp_path: Path) -> None:
    dark, light = tmp_path / "dark.jpg", tmp_path / "light.jpg"
    Image.new("RGB", (64, 48), (0, 0, 0)).save(dark)
    Image.new("RGB", (64, 48), (255, 255, 255)).save(light)

    values = brightness_of([sample("dark", 0.0, "g", dark), sample("light", 0.0, "g", light)])

    assert values[0] == pytest.approx(0.0, abs=1.0)
    assert values[1] == pytest.approx(255.0, abs=1.0)


@pytest.mark.ml
def test_report_shows_the_metrics_the_ensemble_and_the_worst_photos() -> None:
    photos = WORST_SHOWN + 6
    loads = [float(i % 11 * 10) for i in range(photos)]
    samples = [sample(f"img_{i:04d}", load, f"grp_{i // 2}") for i, load in enumerate(loads)]
    labels = np.array(loads)
    # The error grows with the index, so the gallery must show the last photos and not the first.
    predicted = labels + np.arange(photos, dtype=np.float64)
    folds = [i % 5 for i in range(photos)]

    page = build_report(samples, labels, predicted, folds, np.full(photos, 120.0), MEMBERS)

    assert page.startswith("<!doctype html>")
    assert "siglip2_vitb16 alpha=300" in page
    assert f"img_{photos - 1:04d}.jpg" in page
    assert "img_0000.jpg" not in page
    assert "<svg" in page
