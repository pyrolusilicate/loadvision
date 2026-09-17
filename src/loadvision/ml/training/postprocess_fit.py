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
from sklearn.isotonic import IsotonicRegression

from loadvision.ml.postprocess import (
    CURVE_STEPS,
    MAX_LOAD,
    MIN_LOAD,
    Applier,
    Curve,
    Loads,
    clip,
    make_chain,
    make_step,
    save_curve,
)

type Learned = tuple[dict[str, float], Curve | None]
type Fitter = Callable[[Loads, Loads, Mapping[str, float]], Learned]

# Candidate values a step searches over, in percentage points.
_SNAP_HIGH_GRID: Final = np.arange(60.0, 100.0, 1.0)
_SNAP_LOW_GRID: Final = np.arange(0.0, 20.0, 0.5)
_EXPAND_GRID: Final = np.arange(1.0, 2.51, 0.02)
_MIN_BIN_SIZE: Final = 5
_DEFAULT_BINS: Final = 16


@dataclass(frozen=True, slots=True)
class FittedStep:
    """One fitted step, in the form a manifest and a package store it.

    Attributes:
        name: Registry key in `ml.postprocess`.
        params: Numeric parameters, learned ones included.
        curve: Learned curve, or None for a step without one.
    """

    name: str
    params: dict[str, float] = field(default_factory=dict[str, float])
    curve: Curve | None = None

    def apply(self, values: Loads) -> Loads:
        """Applies the step through the registry, exactly as the service will."""
        return make_step(self.name, self.params, self.curve)(values)

    def save_curve_to(self, path: Path) -> Path | None:
        """Writes the curve next to the head, or does nothing for a step without one.

        Args:
            path: File to write inside the package.

        Returns:
            The path, or None if this step has no curve.
        """
        if self.curve is None:
            return None
        save_curve(path, self.curve)
        return path


@dataclass(frozen=True, slots=True)
class FittedChain:
    """A fitted chain, callable like the applier the service builds from a manifest.

    Attributes:
        steps: Fitted steps in application order.
    """

    steps: tuple[FittedStep, ...]

    def __call__(self, values: Loads) -> Loads:
        """Applies every step in order through the registry."""
        return self.applier()(values)

    def applier(self) -> Applier:
        """Builds the same callable the service builds from a manifest."""
        return make_chain([(step.name, step.params, step.curve) for step in self.steps])


def _fit_stateless(_predictions: Loads, _labels: Loads, params: Mapping[str, float]) -> Learned:
    return dict(params), None


def _fit_snap_high(predictions: Loads, labels: Loads, params: Mapping[str, float]) -> Learned:
    if "threshold" in params:
        return {"threshold": float(params["threshold"])}, None
    threshold = _best_value(
        predictions,
        labels,
        _SNAP_HIGH_GRID,
        lambda values, edge: np.where(values >= edge, MAX_LOAD, values),
    )
    return {"threshold": threshold}, None


def _fit_snap_low(predictions: Loads, labels: Loads, params: Mapping[str, float]) -> Learned:
    if "threshold" in params:
        return {"threshold": float(params["threshold"])}, None
    threshold = _best_value(
        predictions,
        labels,
        _SNAP_LOW_GRID,
        lambda values, edge: np.where(values <= edge, MIN_LOAD, values),
    )
    return {"threshold": threshold}, None


def _fit_linear_expand(predictions: Loads, labels: Loads, _params: Mapping[str, float]) -> Learned:
    centre = float(predictions.mean())
    factor = _best_value(
        predictions, labels, _EXPAND_GRID, lambda values, k: centre + k * (values - centre)
    )
    return {"centre": centre, "factor": factor}, None


def _fit_isotonic(predictions: Loads, labels: Loads, _params: Mapping[str, float]) -> Learned:
    """Learns a monotone curve that undoes the pull of a fitted head towards the middle."""
    if np.unique(predictions).size < 2:
        raise ValueError("isotonic needs at least two distinct predictions")
    model = IsotonicRegression(y_min=MIN_LOAD, y_max=MAX_LOAD, out_of_bounds="clip")
    model.fit(predictions, labels)
    knots_in = np.unique(predictions)
    return {}, (knots_in, np.asarray(model.predict(knots_in), dtype=np.float64))


def _fit_median_bins(predictions: Loads, labels: Loads, params: Mapping[str, float]) -> Learned:
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
    return {"bins": float(bins)}, (np.array(centres), knots_out)


FITTERS: Final[Mapping[str, Fitter]] = {
    "clip": _fit_stateless,
    "round_to": _fit_stateless,
    "snap_high": _fit_snap_high,
    "snap_low": _fit_snap_low,
    "linear_expand": _fit_linear_expand,
    "isotonic": _fit_isotonic,
    "median_bins": _fit_median_bins,
}


def fit_step(
    name: str, predictions: Loads, labels: Loads, params: Mapping[str, float] | None = None
) -> FittedStep:
    """Fits one step by registry name.

    Args:
        name: Registry key.
        predictions: Predictions for photos the model had not seen.
        labels: Reference loads for the same photos.
        params: Numeric parameters given by the experiment; learned ones are added.

    Returns:
        The fitted step.

    Raises:
        KeyError: If the name is not in the registry.
        ValueError: If the inputs disagree or a parameter is invalid.
    """
    raw = clip(predictions)
    target = np.asarray(labels, dtype=np.float64)
    if raw.shape != target.shape:
        raise ValueError(f"shapes differ: {raw.shape} and {target.shape}")
    learned, curve = FITTERS[name](raw, target, params or {})
    if name in CURVE_STEPS and curve is None:
        raise ValueError(f"{name} must produce a curve")
    return FittedStep(name=name, params=learned, curve=curve)


def fit_chain(
    steps: Sequence[tuple[str, Mapping[str, float]]] | Sequence[str],
    predictions: Loads,
    labels: Loads,
) -> FittedChain:
    """Fits several steps in order, each on the output of the ones before it.

    Args:
        steps: Registry names, or pairs of name and parameters.
        predictions: Predictions for photos the model had not seen.
        labels: Reference loads for the same photos.

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
        one = fit_step(name, current, target, params)
        fitted.append(one)
        current = one.apply(current)
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
