import numpy as np
import pytest

from loadvision.ml.evaluation.metrics import (
    mae,
    mae_by_fold,
    mae_by_range,
    summary,
    within_tolerance,
)


@pytest.mark.ml
def test_mae_is_the_mean_absolute_error() -> None:
    assert mae([80.0, 50.0], [65.0, 55.0]) == pytest.approx(10.0)


@pytest.mark.ml
def test_perfect_predictions_score_zero() -> None:
    assert mae([0.0, 50.0, 100.0], [0.0, 50.0, 100.0]) == 0.0


@pytest.mark.ml
def test_within_tolerance_counts_the_boundary_as_a_hit() -> None:
    assert within_tolerance([50.0, 50.0], [60.0, 61.0]) == pytest.approx(0.5)


@pytest.mark.ml
def test_ranges_are_split_by_the_true_load() -> None:
    true = [10.0, 50.0, 90.0]
    pred = [12.0, 40.0, 90.0]

    report = mae_by_range(true, pred)

    assert report == pytest.approx({"0-20": 2.0, "20-80": 10.0, "80-100": 0.0})


@pytest.mark.ml
def test_label_100_belongs_to_the_last_range() -> None:
    assert mae_by_range([100.0], [95.0]) == pytest.approx({"80-100": 5.0})


@pytest.mark.ml
def test_empty_range_is_left_out() -> None:
    assert set(mae_by_range([90.0], [80.0])) == {"80-100"}


@pytest.mark.ml
def test_mae_by_fold_splits_the_photos() -> None:
    report = mae_by_fold([10.0, 20.0, 30.0, 40.0], [12.0, 20.0, 35.0, 40.0], [0, 0, 1, 1])

    assert report == pytest.approx({0: 1.0, 1: 2.5})


@pytest.mark.ml
def test_summary_reports_spread_between_folds() -> None:
    report = summary([10.0, 20.0, 30.0, 40.0], [12.0, 20.0, 35.0, 40.0], [0, 0, 1, 1])

    assert report["mae"] == pytest.approx(1.75)
    assert report["within_10"] == 1.0
    assert report["mae_fold_spread"] == pytest.approx(1.5)
    assert report["bias"] == pytest.approx(1.75)


@pytest.mark.ml
def test_summary_without_folds_has_no_fold_numbers() -> None:
    report = summary([10.0, 20.0], [10.0, 20.0])

    assert not [key for key in report if key.startswith("mae_fold")]


@pytest.mark.ml
@pytest.mark.parametrize(
    ("true", "pred", "message"),
    [
        ([10.0, 20.0], [10.0], "shapes differ"),
        ([], [], "no predictions"),
        ([10.0], [np.nan], "must be finite"),
    ],
)
def test_broken_inputs_are_rejected(true: list[float], pred: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        mae(true, pred)
