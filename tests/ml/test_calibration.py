from pathlib import Path

import numpy as np
import pytest

from loadvision.ml.training.calibration import IsotonicCalibrator


def hedged(rng: np.random.Generator, photos: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Labels over the whole range and predictions pulled towards the middle, as a head does."""
    labels = rng.uniform(0, 100, size=photos)
    predictions = 50 + 0.6 * (labels - 50) + rng.normal(scale=4.0, size=photos)
    return predictions, labels


@pytest.mark.ml
def test_calibration_undoes_the_pull_towards_the_middle() -> None:
    predictions, labels = hedged(np.random.default_rng(0))

    curve = IsotonicCalibrator.fit(predictions, labels)
    corrected = curve.apply(predictions)

    assert np.abs(labels - corrected).mean() < np.abs(labels - predictions).mean()
    # The hedged prediction never reached the ends; the corrected one does.
    assert corrected.max() > predictions.max()
    assert corrected.min() < predictions.min()


@pytest.mark.ml
def test_the_curve_never_goes_down() -> None:
    predictions, labels = hedged(np.random.default_rng(1))

    curve = IsotonicCalibrator.fit(predictions, labels)

    assert np.all(np.diff(curve.knots_out) >= 0)


@pytest.mark.ml
def test_results_stay_inside_the_allowed_range() -> None:
    predictions, labels = hedged(np.random.default_rng(2))
    curve = IsotonicCalibrator.fit(predictions, labels)

    corrected = curve.apply(np.array([-500.0, 0.0, 50.0, 500.0]))

    assert corrected.min() >= 0.0
    assert corrected.max() <= 100.0


@pytest.mark.ml
def test_round_trip_through_a_file(tmp_path: Path) -> None:
    predictions, labels = hedged(np.random.default_rng(3))
    curve = IsotonicCalibrator.fit(predictions, labels)
    path = tmp_path / "calibration.npz"

    curve.save(path)
    restored = IsotonicCalibrator.load(path)

    assert np.allclose(restored.apply(predictions), curve.apply(predictions))


@pytest.mark.ml
def test_a_file_without_the_curve_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "calibration.npz"
    np.savez(path, something_else=np.zeros(3))

    with pytest.raises(ValueError, match="missing knots_in"):
        IsotonicCalibrator.load(path)


@pytest.mark.ml
def test_a_decreasing_curve_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        IsotonicCalibrator(knots_in=np.array([0.0, 10.0]), knots_out=np.array([10.0, 0.0]))


@pytest.mark.ml
def test_fitting_on_a_single_prediction_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two distinct"):
        IsotonicCalibrator.fit(np.full(10, 50.0), np.arange(10, dtype=float))


@pytest.mark.ml
def test_fitting_on_disagreeing_shapes_is_rejected() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        IsotonicCalibrator.fit(np.zeros(4), np.zeros(5))
