import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from loadvision.ml.postprocess import (
    APPLIERS,
    load_curve,
    make_chain,
    make_step,
    names,
    save_curve,
)


def curve() -> tuple[np.ndarray, np.ndarray]:
    """A monotone curve that pushes hedged predictions back towards the ends."""
    return np.array([20.0, 50.0, 80.0]), np.array([0.0, 50.0, 100.0])


@pytest.mark.ml
def test_registry_lists_every_step() -> None:
    assert set(names()) == set(APPLIERS)
    assert names() == tuple(sorted(names()))


@pytest.mark.ml
def test_applying_needs_no_training_library() -> None:
    # In a fresh process: other tests import scikit-learn, and the service must not. Applying
    # a chain has to work in an image that carries no training library at all.
    code = (
        "import sys, loadvision.ml.postprocess;"
        "print(any(m.startswith('sklearn') for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"


@pytest.mark.ml
def test_clip_keeps_loads_inside_the_range() -> None:
    assert list(make_step("clip")(np.array([-10.0, 50.0, 150.0]))) == [0.0, 50.0, 100.0]


@pytest.mark.ml
def test_round_to_moves_to_the_nearest_step() -> None:
    step = make_step("round_to", {"step": 5.0})

    assert list(step(np.array([12.0, 13.0, 97.6]))) == [10.0, 15.0, 100.0]


@pytest.mark.ml
def test_round_to_sends_a_half_up_like_the_service() -> None:
    step = make_step("round_to", {"step": 5.0})

    # numpy would send 92.5 to the even neighbour, 90; rounding half up gives 95.
    assert list(step(np.array([92.5, 17.5]))) == [95.0, 20.0]


@pytest.mark.ml
def test_snap_high_sends_confident_predictions_to_a_hundred() -> None:
    step = make_step("snap_high", {"threshold": 92.0})

    assert list(step(np.array([91.9, 92.0, 40.0]))) == [91.9, 100.0, 40.0]


@pytest.mark.ml
def test_snap_low_sends_confident_predictions_to_zero() -> None:
    step = make_step("snap_low", {"threshold": 4.0})

    assert list(step(np.array([3.9, 4.0, 40.0]))) == [0.0, 0.0, 40.0]


@pytest.mark.ml
def test_linear_expand_pushes_predictions_away_from_the_centre() -> None:
    step = make_step("linear_expand", {"centre": 50.0, "factor": 2.0})

    assert list(step(np.array([40.0, 50.0, 60.0]))) == [30.0, 50.0, 70.0]


@pytest.mark.ml
def test_a_curve_step_interpolates_between_its_knots() -> None:
    step = make_step("isotonic", {}, curve())

    assert list(step(np.array([20.0, 35.0, 80.0]))) == [0.0, 25.0, 100.0]


@pytest.mark.ml
def test_a_chain_applies_its_steps_in_order() -> None:
    chain = make_chain([("isotonic", {}, curve()), ("round_to", {"step": 5.0}, None)])

    corrected = chain(np.array([20.0, 36.0, 80.0]))

    assert list(corrected) == [0.0, 25.0, 100.0]


@pytest.mark.ml
def test_results_never_leave_the_allowed_range() -> None:
    chain = make_chain([("linear_expand", {"centre": 50.0, "factor": 10.0}, None)])

    corrected = chain(np.array([0.0, 100.0]))

    assert list(corrected) == [0.0, 100.0]


@pytest.mark.ml
def test_a_curve_step_without_a_curve_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs a curve"):
        make_step("median_bins", {"bins": 16.0})


@pytest.mark.ml
@pytest.mark.parametrize("name", ["snap_high", "snap_low"])
def test_a_snap_without_its_threshold_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="threshold"):
        make_step(name)


@pytest.mark.ml
def test_unknown_step_is_rejected() -> None:
    with pytest.raises(KeyError):
        make_step("magic")


@pytest.mark.ml
def test_a_step_with_a_zero_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="step must be positive"):
        make_step("round_to", {"step": 0.0})


@pytest.mark.ml
def test_curve_round_trip_through_a_file(tmp_path: Path) -> None:
    path = tmp_path / "calibration.npz"

    save_curve(path, curve())
    restored = load_curve(path)

    assert np.allclose(restored[0], curve()[0])
    assert np.allclose(restored[1], curve()[1])


@pytest.mark.ml
def test_a_file_without_the_curve_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "calibration.npz"
    np.savez(path, something_else=np.zeros(3))

    with pytest.raises(ValueError, match="missing knots_in"):
        load_curve(path)


@pytest.mark.ml
def test_a_decreasing_curve_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        make_step("isotonic", {}, (np.array([0.0, 10.0]), np.array([10.0, 0.0])))
