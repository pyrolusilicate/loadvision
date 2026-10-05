"""How high the cargo stands, read off a depth grid.

The grid holds relative depth, larger meaning nearer, so a stack is a run of rows whose depth
barely changes and which stands nearer than the far wall. The grids below are built by hand:
a profile down the frame, repeated across every column.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from loadvision.domain.cargo_weight import MAX_STACK_SHARE
from loadvision.ml.inference.cargo import STACK_COLS, STACK_ROWS, stack_share


def grid_of(profile: list[float], cols: int = STACK_COLS) -> NDArray[np.float32]:
    """Builds a flat depth grid whose every column repeats this profile down the frame."""
    down = np.asarray(profile, dtype=np.float32).reshape(-1, 1)
    return np.repeat(down, cols, axis=1).reshape(-1)


@pytest.mark.unit
def test_a_wall_of_cargo_is_measured_by_the_rows_it_covers() -> None:
    """Eight flat rows near the camera, then the floor running away from it."""
    floor = np.linspace(0.875, 0.0, 8).tolist()
    share = stack_share(grid_of([1.0] * 8 + floor), STACK_ROWS, STACK_COLS)

    # Seven steps inside the wall are flat; the eighth, down onto the floor, is not.
    assert share == pytest.approx(7 / STACK_ROWS)


@pytest.mark.unit
def test_a_taller_stack_measures_higher_than_a_shorter_one() -> None:
    short = stack_share(
        grid_of([1.0] * 4 + np.linspace(0.9, 0.0, 12).tolist()), STACK_ROWS, STACK_COLS
    )
    tall = stack_share(
        grid_of([1.0] * 10 + np.linspace(0.9, 0.0, 6).tolist()), STACK_ROWS, STACK_COLS
    )

    assert tall > short > 0


@pytest.mark.unit
def test_an_empty_body_measures_nothing() -> None:
    """Floor to far wall is one steady slope: no row is flat, so no face is found."""
    share = stack_share(grid_of(np.linspace(1.0, 0.0, STACK_ROWS).tolist()), STACK_ROWS, STACK_COLS)

    assert share == 0.0


@pytest.mark.unit
def test_a_frame_of_one_depth_measures_nothing() -> None:
    """A wall filling the view has no spread to measure a stack against."""
    assert stack_share(grid_of([0.5] * STACK_ROWS), STACK_ROWS, STACK_COLS) == 0.0


@pytest.mark.unit
def test_a_single_flat_step_is_noise_and_not_a_stack() -> None:
    """Two rows of sixteen is the shortest run that counts; one step is less than that."""
    profile = [1.0, 1.0, *np.linspace(0.8, 0.0, STACK_ROWS - 2).tolist()]
    share = stack_share(grid_of(profile), STACK_ROWS, STACK_COLS)

    assert share == 0.0


@pytest.mark.unit
def test_a_stack_filling_the_frame_stops_at_the_cap() -> None:
    """Above the cap the roof and the far wall merge with the cargo and the number stops meaning
    anything, so it is never reported higher."""
    rows = 32
    share = stack_share(grid_of([1.0] * 31 + [0.0]), rows, STACK_COLS)

    assert share == MAX_STACK_SHARE
