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
# What a step may know about the photos besides the prediction. Keys are fact names, values
# hold one number per prediction. Conditional steps read them; the rest ignore them.
type Facts = Mapping[str, NDArray[np.float64]]
type Applier = Callable[[Loads, Facts], Loads]
type Builder = Callable[[Mapping[str, float], Curves | None], Applier]
# A step may carry one curve or one per group; both arrive as a list in group order.
type Curves = Sequence[Curve]

ASPECT: Final = "aspect"

MIN_LOAD: Final = 0.0
MAX_LOAD: Final = 100.0

CURVE_KEYS: Final = ("knots_in", "knots_out")


def clip(values: Loads) -> Loads:
    """Keeps loads inside the allowed range."""
    return np.clip(np.asarray(values, dtype=np.float64), MIN_LOAD, MAX_LOAD)


def _build_clip(_params: Mapping[str, float], _curves: Curves | None) -> Applier:
    return lambda values, _facts: clip(values)


def _build_round_to(params: Mapping[str, float], _curves: Curves | None) -> Applier:
    step = _required(params, "step", default=5.0)
    if step <= 0:
        raise ValueError(f"round_to step must be positive, got {step}")
    # A half always goes up. numpy sends a half to the even neighbour, which would turn 92.5
    # into 90 while a service rounding half up turns the same value into 95.
    return lambda values, _facts: clip(np.floor(clip(values) / step + 0.5) * step)


def _build_snap_high(params: Mapping[str, float], _curves: Curves | None) -> Applier:
    threshold = _required(params, "threshold")
    return lambda values, _facts: clip(np.where(clip(values) >= threshold, MAX_LOAD, clip(values)))


def _build_snap_low(params: Mapping[str, float], _curves: Curves | None) -> Applier:
    threshold = _required(params, "threshold")
    return lambda values, _facts: clip(np.where(clip(values) <= threshold, MIN_LOAD, clip(values)))


def _build_linear_expand(params: Mapping[str, float], _curves: Curves | None) -> Applier:
    centre = _required(params, "centre")
    factor = _required(params, "factor")
    return lambda values, _facts: clip(centre + factor * (clip(values) - centre))


def _build_curve(_params: Mapping[str, float], curves: Curves | None) -> Applier:
    if not curves:
        raise ValueError("this step needs a curve; the package must carry its file")
    knots_in, knots_out = curves[0]
    check_curve(knots_in, knots_out)
    return lambda values, _facts: clip(np.interp(clip(values), knots_in, knots_out))


def _build_curve_by_aspect(params: Mapping[str, float], curves: Curves | None) -> Applier:
    """One curve per population of photos, chosen by the proportions of the frame.

    The dataset holds two populations with different load distributions, and one curve is a
    compromise that helps the larger one and hurts the other. Proportions separate them and
    cost nothing: the service already knows the size of the decoded photo.
    """
    if not curves or len(curves) != _ASPECT_GROUPS:
        raise ValueError(f"this step needs {_ASPECT_GROUPS} curves, got {len(curves or ())}")
    for knots_in, knots_out in curves:
        check_curve(knots_in, knots_out)
    low = _required(params, "low")
    high = _required(params, "high")
    if low > high:
        raise ValueError(f"aspect range is empty: {low} to {high}")

    def apply(values: Loads, facts: Facts) -> Loads:
        aspect = facts.get(ASPECT)
        if aspect is None:
            raise ValueError(f"this step needs the {ASPECT!r} fact of every photo")
        inside = (np.asarray(aspect, np.float64) >= low) & (np.asarray(aspect, np.float64) <= high)
        result = clip(values)
        for group, (knots_in, knots_out) in enumerate(curves):
            chosen = inside if group == _USUAL_GROUP else ~inside
            if chosen.any():
                result[chosen] = np.interp(result[chosen], knots_in, knots_out)
        return clip(result)

    return apply


# Steps that learned a monotone shape share one applier, interpolation between knots. What
# differs is how the knots were found, and that belongs to the fitting side.
APPLIERS: Final[Mapping[str, Builder]] = {
    "clip": _build_clip,
    "round_to": _build_round_to,
    "snap_high": _build_snap_high,
    "snap_low": _build_snap_low,
    "linear_expand": _build_linear_expand,
    "isotonic": _build_curve,
    "isotonic_smooth": _build_curve,
    "median_bins": _build_curve,
    "median_bins_smooth": _build_curve,
    "isotonic_by_aspect": _build_curve_by_aspect,
    "isotonic_smooth_by_aspect": _build_curve_by_aspect,
    "median_bins_by_aspect": _build_curve_by_aspect,
    "median_bins_smooth_by_aspect": _build_curve_by_aspect,
}

# Steps whose shape is learned and stored in a file next to the head.
CURVE_STEPS: Final = frozenset(
    {
        "isotonic",
        "isotonic_smooth",
        "median_bins",
        "median_bins_smooth",
        "isotonic_by_aspect",
        "isotonic_smooth_by_aspect",
        "median_bins_by_aspect",
        "median_bins_smooth_by_aspect",
    }
)

# Steps that pick their curve by a fact of the photo rather than by the prediction alone.
CONDITIONAL_STEPS: Final = frozenset(
    {
        "isotonic_by_aspect",
        "isotonic_smooth_by_aspect",
        "median_bins_by_aspect",
        "median_bins_smooth_by_aspect",
    }
)

_ASPECT_GROUPS: Final = 2
_USUAL_GROUP: Final = 0


def names() -> tuple[str, ...]:
    """Registry keys of every postprocessing step, sorted."""
    return tuple(sorted(APPLIERS))


def make_step(
    name: str, params: Mapping[str, float] | None = None, curves: Curves | None = None
) -> Applier:
    """Builds one step from what a manifest says about it.

    Args:
        name: Registry key.
        params: Numeric parameters, including the ones learned during fitting.
        curves: Learned curves, one for a plain step and one per group for a conditional one.

    Returns:
        Callable that maps loads and photo facts to loads.

    Raises:
        KeyError: If the name is not in the registry.
        ValueError: If a parameter or a curve is missing or invalid.
    """
    return APPLIERS[name](params or {}, curves)


def make_chain(steps: Sequence[tuple[str, Mapping[str, float], Curves | None]]) -> Applier:
    """Builds a chain that applies its steps in order and clips the result.

    Args:
        steps: Name, parameters and curves of every step, in manifest order.

    Returns:
        Callable that maps loads and photo facts to loads.

    Raises:
        KeyError: If a name is not in the registry.
        ValueError: If a parameter or a curve is missing or invalid.
    """
    appliers = [make_step(name, params, curves) for name, params, curves in steps]

    def apply(values: Loads, facts: Facts | None = None) -> Loads:
        result = clip(values)
        for applier in appliers:
            result = applier(result, facts or {})
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


def load_curves(path: Path) -> Curves:
    """Reads the curves of a step from the `.npz` file a package carries.

    Arrays and no pickle: the service loads the file without executing anything. A plain step
    stores one curve, a conditional one stores a curve per group, named by its index.

    Args:
        path: File inside a model package.

    Returns:
        The curves, in group order.

    Raises:
        ValueError: If no curve is stored or one of them is invalid.
    """
    curves: list[Curve] = []
    with np.load(path, allow_pickle=False) as stored:
        for index in range(len(stored.files)):
            keys = [f"{key}_{index}" for key in CURVE_KEYS]
            if not all(key in stored for key in keys):
                break
            curves.append(
                (
                    np.asarray(stored[keys[0]], dtype=np.float64),
                    np.asarray(stored[keys[1]], dtype=np.float64),
                )
            )
    if not curves:
        raise ValueError(f"{path.name}: no curve stored, expected knots_in_0 and knots_out_0")
    for knots_in, knots_out in curves:
        check_curve(knots_in, knots_out)
    return curves


def save_curves(path: Path, curves: Curves) -> None:
    """Writes the curves of a step as the `.npz` file a package carries."""
    if not curves:
        raise ValueError("a step with a curve must store at least one")
    # Typed as Any because np.savez treats a keyword named like its own parameters, such as
    # allow_pickle, as that parameter; the keys here are always knots_in_N and knots_out_N.
    arrays: dict[str, Any] = {}
    for index, (knots_in, knots_out) in enumerate(curves):
        check_curve(knots_in, knots_out)
        arrays[f"knots_in_{index}"] = knots_in
        arrays[f"knots_out_{index}"] = knots_out
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _required(params: Mapping[str, float], name: str, default: float | None = None) -> float:
    value = params.get(name, default)
    if value is None:
        raise ValueError(f"postprocessing parameter {name!r} is missing")
    return float(value)
