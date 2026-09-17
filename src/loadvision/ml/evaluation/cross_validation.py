"""Out-of-fold predictions: every photo scored by a model that never saw it."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.evaluation.metrics import summary

type Predictor = Callable[[NDArray[Any]], NDArray[np.float64]]
type FitFunction = Callable[[NDArray[Any], NDArray[np.float64]], Predictor]


@dataclass(frozen=True, slots=True, kw_only=True)
class OutOfFold:
    """Predictions of the fold models and what they add up to.

    Attributes:
        predictions: One prediction per photo, each made by the model of another fold.
        folds: Fold number of every photo, in the same order.
        metrics: Flat metrics of the predictions, ready for `summary.json`.
    """

    predictions: NDArray[np.float64]
    folds: tuple[int, ...]
    metrics: dict[str, float]


def out_of_fold(
    features: NDArray[Any],
    labels: NDArray[Any],
    folds: Sequence[int],
    fit: FitFunction,
) -> OutOfFold:
    """Trains one model per fold and predicts the fold it was not trained on.

    This is the only honest way to measure a model on the data it was built from: a photo is
    always scored by weights fitted without it, and without any photo of its truck, because the
    folds keep whole scenes together.

    Args:
        features: Embeddings of shape (photos, features).
        labels: Reference loads, one per photo.
        folds: Fold number of every photo.
        fit: Called with the training features and labels of one fold, returns a predictor.

    Returns:
        Predictions covering every photo exactly once, with their metrics.

    Raises:
        ValueError: If the shapes disagree, if there are fewer than two folds, or if a fold is
            empty on either side.
    """
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    numbers = np.asarray(folds, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != target.size or numbers.size != target.size:
        raise ValueError(
            f"features {matrix.shape}, labels {target.shape} and folds {numbers.shape} disagree"
        )
    unique = sorted(set(numbers.tolist()))
    if len(unique) < 2:
        raise ValueError(f"need at least two folds, got {unique}")

    predictions = np.full(target.shape, np.nan, dtype=np.float64)
    for fold in unique:
        validation = numbers == fold
        training = ~validation
        if not training.any():
            raise ValueError(f"fold {fold} leaves no photos to train on")
        predict = fit(matrix[training], target[training])
        predictions[validation] = predict(matrix[validation])

    if not np.isfinite(predictions).all():
        raise ValueError("some photos were left without a prediction")
    return OutOfFold(
        predictions=predictions,
        folds=tuple(int(value) for value in numbers.tolist()),
        metrics=summary(target, predictions, numbers.tolist()),
    )
