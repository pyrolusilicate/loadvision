"""Registry of postprocessing steps applied to raw predictions.

Every step has the same shape: it is fitted on predictions a model made for photos it had not
seen, and afterwards it maps loads to loads. Steps compose into a chain, so a package can say
"isotonic, then snap high, then round" without the service knowing what any of them does.

Stateless steps ignore the labels; fitted steps learn their numbers from them. Both come back
as the same kind of callable, so `fit_chain` treats them alike.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.training.calibration import IsotonicCalibrator

type Loads = NDArray[np.float64]
type Fitted = Callable[[Loads], Loads]
type Fitter = Callable[[Loads, Loads, Mapping[str, float]], Fitted]

MIN_LOAD: Final = 0.0
MAX_LOAD: Final = 100.0

# Candidate thresholds a snap step searches over, in percentage points.
_SNAP_GRID: Final = np.arange(60.0, 100.0, 1.0)
_EXPAND_GRID: Final = np.arange(1.0, 2.51, 0.02)
_MIN_BIN_SIZE: Final = 5


def _clip(values: Loads) -> Loads:
    return np.clip(values, MIN_LOAD, MAX_LOAD)


def fit_clip(_predictions: Loads, _labels: Loads, _params: Mapping[str, float]) -> Fitted:
    """Keeps predictions inside the allowed range."""
    return _clip


def fit_round_to(_predictions: Loads, _labels: Loads, params: Mapping[str, float]) -> Fitted:
    """Rounds to a step, because experts only ever write round numbers.

    Args:
        params: `step`, such as 5.
    """
    step = float(params.get("step", 5.0))
    if step <= 0:
        raise ValueError(f"round_to step must be positive, got {step}")
    # A half always goes up, matching the service predictor. numpy sends a half to the even
    # neighbour, which would turn 92.5 into 90 here and into 95 there.
    return lambda values: _clip(np.floor(values / step + 0.5) * step)


def fit_snap_high(predictions: Loads, labels: Loads, params: Mapping[str, float]) -> Fitted:
    """Sends confident high predictions to exactly 100.

    A fifth of the dataset is labelled exactly 100 and a regression practically never reaches
    it, so a threshold is worth learning. It is searched on the photos given here, which during
    evaluation are the training folds only.

    Args:
        predictions: Predictions the threshold is searched on.
        labels: Reference loads for the same photos.
        params: `threshold` to use as is; omitted means search for the best one.
    """
    if "threshold" in params:
        threshold = float(params["threshold"])
    else:
        threshold = _best_threshold(
            predictions,
            labels,
            _SNAP_GRID,
            lambda values, edge: np.where(values >= edge, MAX_LOAD, values),
        )
    return lambda values: _clip(np.where(values >= threshold, MAX_LOAD, values))


def fit_snap_low(predictions: Loads, labels: Loads, params: Mapping[str, float]) -> Fitted:
    """Sends confident low predictions to exactly 0.

    Args:
        predictions: Predictions the threshold is searched on.
        labels: Reference loads for the same photos.
        params: `threshold` to use as is; omitted means search for the best one.
    """
    grid = np.arange(0.0, 20.0, 0.5)
    if "threshold" in params:
        threshold = float(params["threshold"])
    else:
        threshold = _best_threshold(
            predictions,
            labels,
            grid,
            lambda values, edge: np.where(values <= edge, MIN_LOAD, values),
        )
    return lambda values: _clip(np.where(values <= threshold, MIN_LOAD, values))


def fit_linear_expand(predictions: Loads, labels: Loads, _params: Mapping[str, float]) -> Fitted:
    """Pushes predictions away from their mean, undoing a hedging head with one number."""
    centre = float(predictions.mean())
    best = _best_threshold(
        predictions, labels, _EXPAND_GRID, lambda values, k: centre + k * (values - centre)
    )
    return lambda values: _clip(centre + best * (values - centre))


def fit_isotonic(predictions: Loads, labels: Loads, _params: Mapping[str, float]) -> Fitted:
    """Learns a monotone curve from raw prediction to corrected load."""
    curve = IsotonicCalibrator.fit(predictions, labels)
    return curve.apply


def fit_median_bins(predictions: Loads, labels: Loads, params: Mapping[str, float]) -> Fitted:
    """Maps each band of predictions to the median label of that band.

    The metric is the absolute error, and the value that minimizes it is the median, not the
    mean an isotonic fit produces.

    Args:
        predictions: Predictions the bands are built from.
        labels: Reference loads for the same photos.
        params: `bins`, the number of equally filled bands.
    """
    bins = int(params.get("bins", 16))
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
    knots_in = np.array(centres)
    knots_out = np.maximum.accumulate(np.array(medians))
    return lambda values: _clip(np.interp(values, knots_in, knots_out))


POSTPROCESSORS: Final[Mapping[str, Fitter]] = {
    "clip": fit_clip,
    "round_to": fit_round_to,
    "snap_high": fit_snap_high,
    "snap_low": fit_snap_low,
    "linear_expand": fit_linear_expand,
    "isotonic": fit_isotonic,
    "median_bins": fit_median_bins,
}


def names() -> tuple[str, ...]:
    """Registry keys of every postprocessing step, sorted."""
    return tuple(sorted(POSTPROCESSORS))


def fit_step(
    name: str, predictions: Loads, labels: Loads, params: Mapping[str, float] | None = None
) -> Fitted:
    """Fits one step by registry name.

    Args:
        name: Registry key.
        predictions: Predictions for photos the model had not seen.
        labels: Reference loads for the same photos.
        params: Numeric parameters of the step.

    Returns:
        Callable that maps loads to loads.

    Raises:
        KeyError: If the name is not in the registry.
        ValueError: If the inputs disagree or a parameter is invalid.
    """
    if predictions.shape != labels.shape:
        raise ValueError(f"shapes differ: {predictions.shape} and {labels.shape}")
    return POSTPROCESSORS[name](predictions, labels, params or {})


def fit_chain(
    steps: Sequence[tuple[str, Mapping[str, float]]] | Sequence[str],
    predictions: Loads,
    labels: Loads,
) -> Fitted:
    """Fits several steps in order, each on the output of the ones before it.

    Args:
        steps: Registry names, or pairs of name and parameters.
        predictions: Predictions for photos the model had not seen.
        labels: Reference loads for the same photos.

    Returns:
        Callable that applies the whole chain.

    Raises:
        KeyError: If a name is not in the registry.
        ValueError: If the inputs disagree or a parameter is invalid.
    """
    fitted: list[Fitted] = []
    current = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    for step in steps:
        name, params = (step, {}) if isinstance(step, str) else step
        transform = fit_step(name, current, target, params)
        fitted.append(transform)
        current = transform(current)

    def apply(values: Loads) -> Loads:
        result = np.asarray(values, dtype=np.float64)
        for transform in fitted:
            result = transform(result)
        return result

    return apply


def _best_threshold(
    predictions: Loads,
    labels: Loads,
    grid: NDArray[Any],
    make: Callable[[Loads, float], Loads],
) -> float:
    errors = [
        float(np.abs(labels - _clip(make(predictions, float(value)))).mean()) for value in grid
    ]
    return float(grid[int(np.argmin(errors))])
