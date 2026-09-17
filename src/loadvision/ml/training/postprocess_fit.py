"""Fitting postprocessing steps on out-of-fold predictions.

A step learns from predictions a model made for photos it had not seen, which is the only
place the hedging of a fitted head is visible without cheating. What it learns is either a few
numbers, which go into the manifest as `params`, or a monotone curve, which goes into the
package as a file. Applying what was learned needs neither this module nor scikit-learn: that
is `ml.postprocess`.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression

from loadvision.ml.postprocess import (
    ASPECT,
    CONDITIONAL_STEPS,
    CURVE_STEPS,
    MAX_LOAD,
    MIN_LOAD,
    Applier,
    Curve,
    Curves,
    Facts,
    Loads,
    clip,
    make_chain,
    make_step,
    save_curves,
)

type Learned = tuple[dict[str, float], Curves | None]
type Fitter = Callable[[Loads, Loads, Facts, Mapping[str, float]], Learned]

# Proportions that separate the two populations of the dataset; see docs/ml/user-cases.md.
DEFAULT_ASPECT_RANGE: Final = (0.70, 0.80)
_MIN_GROUP: Final = 30

# Candidate values a step searches over, in percentage points.
_SNAP_HIGH_GRID: Final = np.arange(60.0, 100.0, 1.0)
_SNAP_LOW_GRID: Final = np.arange(0.0, 20.0, 0.5)
_EXPAND_GRID: Final = np.arange(1.0, 2.51, 0.02)
_MIN_BIN_SIZE: Final = 5
_DEFAULT_BINS: Final = 16

# Points a smoothed curve is stored at. Dense enough that the straight lines the service draws
# between them are indistinguishable from the spline, so the service still needs only numpy.
_SMOOTH_POINTS: Final = 101


@dataclass(frozen=True, slots=True)
class FittedStep:
    """One fitted step, in the form a manifest and a package store it.

    Attributes:
        name: Registry key in `ml.postprocess`.
        params: Numeric parameters, learned ones included.
        curves: Learned curves, one per group; None for a step without any.
    """

    name: str
    params: dict[str, float] = field(default_factory=dict[str, float])
    curves: Curves | None = None

    def apply(self, values: Loads, facts: Facts | None = None) -> Loads:
        """Applies the step through the registry, exactly as the service will."""
        return make_step(self.name, self.params, self.curves)(values, facts or {})

    def save_curves_to(self, path: Path) -> Path | None:
        """Writes the curves next to the head, or does nothing for a step without any.

        Args:
            path: File to write inside the package.

        Returns:
            The path, or None if this step has no curve.
        """
        if self.curves is None:
            return None
        save_curves(path, self.curves)
        return path


@dataclass(frozen=True, slots=True)
class FittedChain:
    """A fitted chain, callable like the applier the service builds from a manifest.

    Attributes:
        steps: Fitted steps in application order.
    """

    steps: tuple[FittedStep, ...]

    def __call__(self, values: Loads, facts: Facts | None = None) -> Loads:
        """Applies every step in order through the registry."""
        return self.applier()(values, facts or {})

    def applier(self) -> Applier:
        """Builds the same callable the service builds from a manifest."""
        return make_chain([(step.name, step.params, step.curves) for step in self.steps])


def _fit_stateless(
    _predictions: Loads, _labels: Loads, _facts: Facts, params: Mapping[str, float]
) -> Learned:
    return dict(params), None


def _fit_snap_high(
    predictions: Loads, labels: Loads, _facts: Facts, params: Mapping[str, float]
) -> Learned:
    if "threshold" in params:
        return {"threshold": float(params["threshold"])}, None
    threshold = _best_value(
        predictions,
        labels,
        _SNAP_HIGH_GRID,
        lambda values, edge: np.where(values >= edge, MAX_LOAD, values),
    )
    return {"threshold": threshold}, None


def _fit_snap_low(
    predictions: Loads, labels: Loads, _facts: Facts, params: Mapping[str, float]
) -> Learned:
    if "threshold" in params:
        return {"threshold": float(params["threshold"])}, None
    threshold = _best_value(
        predictions,
        labels,
        _SNAP_LOW_GRID,
        lambda values, edge: np.where(values <= edge, MIN_LOAD, values),
    )
    return {"threshold": threshold}, None


def _fit_linear_expand(
    predictions: Loads, labels: Loads, _facts: Facts, _params: Mapping[str, float]
) -> Learned:
    centre = float(predictions.mean())
    factor = _best_value(
        predictions, labels, _EXPAND_GRID, lambda values, k: centre + k * (values - centre)
    )
    return {"centre": centre, "factor": factor}, None


def _fit_isotonic(
    predictions: Loads, labels: Loads, _facts: Facts, _params: Mapping[str, float]
) -> Learned:
    """Learns a monotone curve that undoes the pull of a fitted head towards the middle."""
    if np.unique(predictions).size < 2:
        raise ValueError("isotonic needs at least two distinct predictions")
    model = IsotonicRegression(y_min=MIN_LOAD, y_max=MAX_LOAD, out_of_bounds="clip")
    model.fit(predictions, labels)
    knots_in = np.unique(predictions)
    return {}, [(knots_in, np.asarray(model.predict(knots_in), dtype=np.float64))]


def _fit_median_bins(
    predictions: Loads, labels: Loads, _facts: Facts, params: Mapping[str, float]
) -> Learned:
    """Maps each band of predictions to the median label of that band.

    The metric is the absolute error, and the value that minimizes it is the median, not the
    mean an isotonic fit produces.
    """
    bins = int(params.get("bins", _DEFAULT_BINS))
    if bins < 2:
        raise ValueError(f"median_bins needs at least two bins, got {bins}")
    edges = np.quantile(predictions, np.linspace(0.0, 1.0, bins + 1))
    which = np.digitize(predictions, edges[1:-1])
    centres: list[float] = []
    medians: list[float] = []
    for index in range(bins):
        inside = which == index
        if int(inside.sum()) >= _MIN_BIN_SIZE:
            centres.append(float(predictions[inside].mean()))
            medians.append(float(np.median(labels[inside])))
    if len(centres) < 2:
        raise ValueError("median_bins found fewer than two usable bins")
    knots_out = np.maximum.accumulate(np.array(medians))
    return {"bins": float(bins)}, [(np.array(centres), knots_out)]


def _smooth_knots(knots_in: Loads, knots_out: Loads) -> Curve:
    """Redraws a curve as a dense monotone spline through its own knots.

    Binning leaves a staircase: every prediction inside one band gets the same answer, which
    throws away the ordering the model found inside that band. A shape-preserving cubic spline
    keeps the knots and fills the gaps smoothly, and cannot overshoot between them the way an
    ordinary cubic would.
    """
    keep = np.concatenate(([True], np.diff(knots_in) > 0))
    inputs, outputs = knots_in[keep], knots_out[keep]
    if inputs.size < 2:
        return knots_in, knots_out
    grid = np.linspace(float(inputs[0]), float(inputs[-1]), _SMOOTH_POINTS)
    return grid, clip(np.maximum.accumulate(PchipInterpolator(inputs, outputs)(grid)))


def _smoothed(base: Fitter) -> Fitter:
    """Runs a fitter that produces knots and replaces its staircase with a smooth curve."""

    def fit(
        predictions: Loads, labels: Loads, facts: Facts, params: Mapping[str, float]
    ) -> Learned:
        learned, curves = base(predictions, labels, facts, params)
        if curves is None:
            raise ValueError("smoothing needs a base step with a curve")
        return learned, [_smooth_knots(knots_in, knots_out) for knots_in, knots_out in curves]

    return fit


def _aspect_groups(
    facts: Facts, params: Mapping[str, float], size: int
) -> tuple[Loads, float, float]:
    aspect = facts.get(ASPECT)
    if aspect is None or len(aspect) != size:
        raise ValueError(f"a conditional step needs the {ASPECT!r} fact of every photo")
    low = float(params.get("low", DEFAULT_ASPECT_RANGE[0]))
    high = float(params.get("high", DEFAULT_ASPECT_RANGE[1]))
    if low > high:
        raise ValueError(f"aspect range is empty: {low} to {high}")
    return np.asarray(aspect, dtype=np.float64), low, high


def _fit_by_aspect(base: Fitter) -> Fitter:
    """Fits the base step once per population of photos.

    The dataset holds two populations with different load distributions, so one curve is a
    compromise: it helps the larger one and hurts the other. A group too small to fit on its
    own falls back to the curve of all photos, which is never worse than guessing.
    """

    def fit(
        predictions: Loads, labels: Loads, facts: Facts, params: Mapping[str, float]
    ) -> Learned:
        aspect, low, high = _aspect_groups(facts, params, predictions.size)
        inside = (aspect >= low) & (aspect <= high)
        learned = dict(params)
        learned["low"] = low
        learned["high"] = high
        curves: list[Curve] = []
        for mask in (inside, ~inside):
            chosen = mask if int(mask.sum()) >= _MIN_GROUP else np.ones_like(mask)
            numbers, curve = base(predictions[chosen], labels[chosen], {}, params)
            if curve is None:
                raise ValueError("a conditional step needs a base step with a curve")
            curves.append(curve[0])
            learned.update(numbers)
        return learned, curves

    return fit


FITTERS: Final[Mapping[str, Fitter]] = {
    "clip": _fit_stateless,
    "round_to": _fit_stateless,
    "snap_high": _fit_snap_high,
    "snap_low": _fit_snap_low,
    "linear_expand": _fit_linear_expand,
    "isotonic": _fit_isotonic,
    "median_bins": _fit_median_bins,
    "median_bins_smooth": _smoothed(_fit_median_bins),
    "isotonic_by_aspect": _fit_by_aspect(_fit_isotonic),
    "median_bins_by_aspect": _fit_by_aspect(_fit_median_bins),
    "median_bins_smooth_by_aspect": _fit_by_aspect(_smoothed(_fit_median_bins)),
}


def fit_step(
    name: str,
    predictions: Loads,
    labels: Loads,
    params: Mapping[str, float] | None = None,
    facts: Facts | None = None,
) -> FittedStep:
    """Fits one step by registry name.

    Args:
        name: Registry key.
        predictions: Predictions for photos the model had not seen.
        labels: Reference loads for the same photos.
        params: Numeric parameters given by the experiment; learned ones are added.
        facts: What is known about the photos; conditional steps need them.

    Returns:
        The fitted step.

    Raises:
        KeyError: If the name is not in the registry.
        ValueError: If the inputs disagree, a parameter is invalid, or a conditional step was
            given no facts.
    """
    raw = clip(predictions)
    target = np.asarray(labels, dtype=np.float64)
    if raw.shape != target.shape:
        raise ValueError(f"shapes differ: {raw.shape} and {target.shape}")
    if name in CONDITIONAL_STEPS and not facts:
        raise ValueError(f"{name} needs the facts of every photo")
    learned, curves = FITTERS[name](raw, target, facts or {}, params or {})
    if name in CURVE_STEPS and not curves:
        raise ValueError(f"{name} must produce a curve")
    return FittedStep(name=name, params=learned, curves=curves)


def fit_chain(
    steps: Sequence[tuple[str, Mapping[str, float]]] | Sequence[str],
    predictions: Loads,
    labels: Loads,
    facts: Facts | None = None,
) -> FittedChain:
    """Fits several steps in order, each on the output of the ones before it.

    Args:
        steps: Registry names, or pairs of name and parameters.
        predictions: Predictions for photos the model had not seen.
        labels: Reference loads for the same photos.
        facts: What is known about the photos; conditional steps need them.

    Returns:
        The fitted chain, callable like the applier built from a manifest.

    Raises:
        KeyError: If a name is not in the registry.
        ValueError: If the inputs disagree or a parameter is invalid.
    """
    fitted: list[FittedStep] = []
    current = clip(predictions)
    target = np.asarray(labels, dtype=np.float64)
    for step in steps:
        name, params = (step, {}) if isinstance(step, str) else step
        one = fit_step(name, current, target, params, facts)
        fitted.append(one)
        current = one.apply(current, facts)
    return FittedChain(steps=tuple(fitted))


def _best_value(
    predictions: Loads,
    labels: Loads,
    grid: NDArray[Any],
    make: Callable[[Loads, float], Loads],
) -> float:
    errors = [
        float(np.abs(labels - clip(make(predictions, float(value)))).mean()) for value in grid
    ]
    return float(grid[int(np.argmin(errors))])
