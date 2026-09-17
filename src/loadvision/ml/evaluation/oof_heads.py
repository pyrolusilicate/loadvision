"""Heads compared out of fold: the shipped ridge and a kNN baseline to judge it against."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.evaluation.cross_validation import FitFunction, Predictor
from loadvision.ml.training.head_trainer import FittedHead


def ridge_predictor(alpha: float) -> FitFunction:
    """Returns a fit function for the ridge head the model package ships.

    Args:
        alpha: Ridge penalty.
    """

    def fit(features: NDArray[Any], labels: NDArray[np.float64]) -> Predictor:
        return FittedHead.fit(features, labels, alpha=alpha).predict

    return fit


def knn_predictor(k: int) -> FitFunction:
    """Returns a fit function that averages the labels of the k nearest training photos.

    It trains nothing, so it shows how much of the result comes from the embeddings alone.

    Args:
        k: Number of neighbours.
    """

    def fit(features: NDArray[Any], labels: NDArray[np.float64]) -> Predictor:
        unit = features / np.linalg.norm(features, axis=1, keepdims=True)

        def predict(batch: NDArray[Any]) -> NDArray[np.float64]:
            query = batch / np.linalg.norm(batch, axis=1, keepdims=True)
            nearest = np.argsort(-(query @ unit.T), axis=1)[:, :k]
            return np.asarray(labels[nearest].mean(axis=1), dtype=np.float64)

        return predict

    return fit
