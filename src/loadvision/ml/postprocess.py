"""Applying postprocessing steps to raw predictions, as a model package describes them.

A step is a name from this registry, its numeric parameters and, for steps that learned a
shape, a curve of two arrays. Everything here is plain numpy on purpose: the service applies
the chain and must not pull a training library into its image to do it. Fitting the steps
lives in `ml.training.postprocess_fit`.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

type Loads = NDArray[np.float64]
type Curve = tuple[NDArray[np.float64], NDArray[np.float64]]
type Applier = Callable[[Loads], Loads]
type Builder = Callable[[Mapping[str, float], Curve | None], Applier]

MIN_LOAD: Final = 0.0
MAX_LOAD: Final = 100.0

CURVE_KEYS: Final = ("knots_in", "knots_out")


def clip(values: Loads) -> Loads:
    """Keeps loads inside the allowed range."""
    return np.clip(np.asarray(values, dtype=np.float64), MIN_LOAD, MAX_LOAD)


def _build_clip(_params: Mapping[str, float], _curve: Curve | None) -> Applier:
    return clip


def _build_round_to(params: Mapping[str, float], _curve: Curve | None) -> Applier:
    step = _required(params, "step", default=5.0)
    if step <= 0:
        raise ValueError(f"round_to step must be positive, got {step}")
    # A half always goes up. numpy sends a half to the even neighbour, which would turn 92.5
    # into 90 while a service rounding half up turns the same value into 95.
    return lambda values: clip(np.floor(clip(values) / step + 0.5) * step)


def _build_snap_high(params: Mapping[str, float], _curve: Curve | None) -> Applier:
    threshold = _required(params, "threshold")
    return lambda values: clip(np.where(clip(values) >= threshold, MAX_LOAD, clip(values)))


def _build_snap_low(params: Mapping[str, float], _curve: Curve | None) -> Applier:
    threshold = _required(params, "threshold")
    return lambda values: clip(np.where(clip(values) <= threshold, MIN_LOAD, clip(values)))


def _build_linear_expand(params: Mapping[str, float], _curve: Curve | None) -> Applier:
    centre = _required(params, "centre")
    factor = _required(params, "factor")
    return lambda values: clip(centre + factor * (clip(values) - centre))


def _build_curve(_params: Mapping[str, float], curve: Curve | None) -> Applier:
    if curve is None:
        raise ValueError("this step needs a curve; the package must carry its file")
    knots_in, knots_out = curve
    check_curve(knots_in, knots_out)
    return lambda values: clip(np.interp(clip(values), knots_in, knots_out))


# Steps that learned a monotone shape share one applier, interpolation between knots. What
# differs is how the knots were found, and that belongs to the fitting side.
APPLIERS: Final[Mapping[str, Builder]] = {
    "clip": _build_clip,
    "round_to": _build_round_to,
    "snap_high": _build_snap_high,
    "snap_low": _build_snap_low,
    "linear_expand": _build_linear_expand,
    "isotonic": _build_curve,
    "median_bins": _build_curve,
}

# Steps whose shape is learned and stored in a file next to the head.
CURVE_STEPS: Final = frozenset({"isotonic", "median_bins"})


def names() -> tuple[str, ...]:
    """Registry keys of every postprocessing step, sorted."""
    return tuple(sorted(APPLIERS))


def make_step(
    name: str, params: Mapping[str, float] | None = None, curve: Curve | None = None
) -> Applier:
    """Builds one step from what a manifest says about it.

    Args:
        name: Registry key.
        params: Numeric parameters, including the ones learned during fitting.
        curve: Learned curve, for steps that need one.

    Returns:
        Callable that maps loads to loads.

    Raises:
        KeyError: If the name is not in the registry.
        ValueError: If a parameter or the curve is missing or invalid.
    """
    return APPLIERS[name](params or {}, curve)


def make_chain(steps: Sequence[tuple[str, Mapping[str, float], Curve | None]]) -> Applier:
    """Builds a chain that applies its steps in order and clips the result.

    Args:
        steps: Name, parameters and curve of every step, in manifest order.

    Returns:
        Callable that maps loads to loads.

    Raises:
        KeyError: If a name is not in the registry.
        ValueError: If a parameter or a curve is missing or invalid.
    """
    appliers = [make_step(name, params, curve) for name, params, curve in steps]

    def apply(values: Loads) -> Loads:
        result = clip(values)
        for applier in appliers:
            result = applier(result)
        return result

    return apply


def check_curve(knots_in: NDArray[Any], knots_out: NDArray[Any]) -> None:
    """Validates a learned curve.

    Args:
        knots_in: Raw predictions the curve is defined at, ascending.
        knots_out: Corrected loads, non-decreasing.

    Raises:
        ValueError: If the arrays disagree, hold fewer than two points, or are not sorted.
    """
    if knots_in.shape != knots_out.shape:
        raise ValueError(f"curve knots disagree: {knots_in.shape} and {knots_out.shape}")
    if knots_in.size < 2:
        raise ValueError("a curve needs at least two points")
    if np.any(np.diff(knots_in) < 0) or np.any(np.diff(knots_out) < 0):
        raise ValueError("curve knots must be non-decreasing")


def load_curve(path: Path) -> Curve:
    """Reads a curve from the `.npz` file a package carries.

    Two arrays and no pickle: the service loads the file without executing anything.

    Args:
        path: File inside a model package.

    Returns:
        The curve.

    Raises:
        ValueError: If a key is missing or the curve is invalid.
    """
    with np.load(path, allow_pickle=False) as stored:
        missing = [key for key in CURVE_KEYS if key not in stored]
        if missing:
            raise ValueError(f"{path.name}: missing {', '.join(missing)}")
        knots_in = np.asarray(stored["knots_in"], dtype=np.float64)
        knots_out = np.asarray(stored["knots_out"], dtype=np.float64)
    check_curve(knots_in, knots_out)
    return knots_in, knots_out


def save_curve(path: Path, curve: Curve) -> None:
    """Writes a curve as the `.npz` file a package carries."""
    knots_in, knots_out = curve
    check_curve(knots_in, knots_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, knots_in=knots_in, knots_out=knots_out)


def _required(params: Mapping[str, float], name: str, default: float | None = None) -> float:
    value = params.get(name, default)
    if value is None:
        raise ValueError(f"postprocessing parameter {name!r} is missing")
    return float(value)
