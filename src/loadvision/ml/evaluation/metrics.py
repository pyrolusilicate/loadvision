"""Quality metrics of load predictions."""

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

type Vector = NDArray[np.floating[Any]] | Sequence[float]

# The task is graded by MAE, but a single average hides where the model fails, so every report
# also splits the photos by their true load. Ranges are half-open, the last one closed.
RANGES: Final = (("0-20", 0.0, 20.0), ("20-80", 20.0, 80.0), ("80-100", 80.0, 100.0))

DEFAULT_TOLERANCE: Final = 10.0

_MAX_LOAD: Final = 100.0


def mae(y_true: Vector, y_pred: Vector) -> float:
    """Mean absolute error in percentage points, the metric of the task.

    Args:
        y_true: Reference loads.
        y_pred: Predicted loads.

    Returns:
        Mean of `abs(y_true - y_pred)`.

    Raises:
        ValueError: If the inputs have different lengths or are empty.
    """
    true, pred = _as_pair(y_true, y_pred)
    return float(np.abs(true - pred).mean())


def within_tolerance(y_true: Vector, y_pred: Vector, tolerance: float = DEFAULT_TOLERANCE) -> float:
    """Share of photos predicted within `tolerance` percentage points.

    Args:
        y_true: Reference loads.
        y_pred: Predicted loads.
        tolerance: Largest error still counted as a hit.

    Returns:
        Share from 0 to 1.

    Raises:
        ValueError: If the inputs have different lengths or are empty.
    """
    true, pred = _as_pair(y_true, y_pred)
    return float((np.abs(true - pred) <= tolerance).mean())


def mae_by_range(y_true: Vector, y_pred: Vector) -> dict[str, float]:
    """MAE inside each load range, keyed by the range name.

    Photos are placed by their true load, so the numbers answer "how wrong are we on nearly
    empty trucks" rather than "how wrong are we when we guess low". A range without photos is
    left out instead of reported as zero.

    Args:
        y_true: Reference loads.
        y_pred: Predicted loads.

    Returns:
        Range name to MAE.

    Raises:
        ValueError: If the inputs have different lengths or are empty.
    """
    true, pred = _as_pair(y_true, y_pred)
    result: dict[str, float] = {}
    for name, low, high in RANGES:
        inside = (true >= low) & (true < high) if high < _MAX_LOAD else (true >= low)
        if inside.any():
            result[name] = float(np.abs(true[inside] - pred[inside]).mean())
    return result


def mae_by_fold(y_true: Vector, y_pred: Vector, folds: Sequence[int]) -> dict[int, float]:
    """MAE inside each fold, keyed by fold number.

    A large spread between folds means the metric itself is noisy, so a small improvement
    between two model versions cannot be trusted.

    Args:
        y_true: Reference loads.
        y_pred: Predicted loads.
        folds: Fold number of every photo.

    Returns:
        Fold number to MAE.

    Raises:
        ValueError: If the inputs have different lengths or are empty.
    """
    true, pred = _as_pair(y_true, y_pred)
    numbers = np.asarray(folds)
    if numbers.shape != true.shape:
        raise ValueError(f"folds has {numbers.shape} entries, predictions have {true.shape}")
    return {
        int(fold): float(np.abs(true[numbers == fold] - pred[numbers == fold]).mean())
        for fold in sorted(set(numbers.tolist()))
    }


def summary(
    y_true: Vector,
    y_pred: Vector,
    folds: Sequence[int] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, float]:
    """Everything a run reports about one set of predictions, as flat named numbers.

    Args:
        y_true: Reference loads.
        y_pred: Predicted loads.
        folds: Fold number of every photo; adds per-fold numbers when given.
        tolerance: Largest error counted as a hit by `within_tolerance`.

    Returns:
        Flat mapping ready for `summary.json`, with `mae`, `within_10`, `mae_0-20` and so on.

    Raises:
        ValueError: If the inputs have different lengths or are empty.
    """
    true, pred = _as_pair(y_true, y_pred)
    report: dict[str, float] = {
        "mae": mae(true, pred),
        f"within_{tolerance:g}": within_tolerance(true, pred, tolerance),
        "bias": float((pred - true).mean()),
        "p95_error": float(np.percentile(np.abs(true - pred), 95)),
    }
    report.update({f"mae_{name}": value for name, value in mae_by_range(true, pred).items()})
    if folds is not None:
        per_fold = mae_by_fold(true, pred, folds)
        report.update({f"mae_fold{fold}": value for fold, value in per_fold.items()})
        report["mae_fold_spread"] = max(per_fold.values()) - min(per_fold.values())
    return report


def paired_difference(errors: Vector, baseline: Vector) -> tuple[float, float]:
    """Mean difference of two models' errors on the same photos, with two standard errors.

    Pairing photo by photo cancels how hard each photo is, which is most of the spread: the
    same 0.3 that is noise between two unpaired averages can be a clear result between two
    paired ones. A difference beyond two standard errors is what every decision here needs.

    Args:
        errors: Absolute errors of the model under test.
        baseline: Absolute errors of the model it is compared with, on the same photos.

    Returns:
        Mean of `errors - baseline`, below zero when the model under test is better, and two
        standard errors of that mean; zero spread for a single photo.

    Raises:
        ValueError: If the inputs have different lengths or are empty.
    """
    ours, theirs = _as_pair(baseline, errors)
    difference = theirs - ours
    spread = (
        float(2 * difference.std(ddof=1) / np.sqrt(difference.size)) if difference.size > 1 else 0.0
    )
    return float(difference.mean()), spread


def _as_pair(y_true: Vector, y_pred: Vector) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"shapes differ: {true.shape} and {pred.shape}")
    if true.size == 0:
        raise ValueError("no predictions to score")
    if not np.isfinite(pred).all():
        raise ValueError("predictions must be finite")
    return true, pred
