"""Fitting the linear head that turns an embedding into a load percent."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

# The file the service reads; `ml.inference.predictor.RidgeHead` expects exactly these arrays.
HEAD_KEYS: Final = ("mean", "scale", "weights", "intercept")

DEFAULT_ALPHA: Final = 1.0

# A feature that never varies would divide by zero when standardized; one keeps it at zero.
_MIN_SCALE: Final = 1e-12

_MIN_LOAD: Final = 0.0
_MAX_LOAD: Final = 100.0


@dataclass(frozen=True, slots=True)
class FittedHead:
    """A fitted linear head, in the form the model package stores.

    The standardization is part of the head: the service gets raw embeddings and must not be
    told separately how to scale them.

    Attributes:
        mean: Mean of every feature in the training photos.
        scale: Standard deviation of every feature; never zero.
        weights: Weight of every standardized feature.
        intercept: Value predicted for an average photo.
    """

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    weights: NDArray[np.float64]
    intercept: float

    def predict(self, features: NDArray[Any]) -> NDArray[np.float64]:
        """Predicts the load of every photo, clipped to the allowed range.

        Args:
            features: Embeddings of shape (photos, features).

        Returns:
            Loads from 0 to 100, one per photo.

        Raises:
            ValueError: If the number of features differs from the fitted one.
        """
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.size:
            raise ValueError(f"head was fitted on {self.mean.size} features, got {matrix.shape}")
        standardized = (matrix - self.mean) / self.scale
        return np.clip(standardized @ self.weights + self.intercept, _MIN_LOAD, _MAX_LOAD)

    def save(self, path: Path) -> None:
        """Writes the head as the `.npz` file a model package carries."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
            intercept=np.float64(self.intercept),
        )

    @classmethod
    def fit(
        cls, features: NDArray[Any], labels: NDArray[Any], *, alpha: float = DEFAULT_ALPHA
    ) -> Self:
        """Fits a ridge regression on standardized embeddings.

        Ridge and not plain least squares: 768 features on a few hundred photos fit the noise
        exactly, and `alpha` trades a little fitting accuracy for weights that survive on
        photos the model has not seen.

        Args:
            features: Embeddings of shape (photos, features).
            labels: Reference loads, one per photo.
            alpha: Strength of the penalty on large weights.

        Returns:
            The fitted head.

        Raises:
            ValueError: If the shapes disagree or there is nothing to fit.
        """
        matrix = np.asarray(features, dtype=np.float64)
        target = np.asarray(labels, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != target.size or target.size == 0:
            raise ValueError(f"cannot fit on features {matrix.shape} and labels {target.shape}")

        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < _MIN_SCALE] = 1.0
        model = Ridge(alpha=alpha)
        model.fit((matrix - mean) / scale, target)
        return cls(
            mean=mean,
            scale=scale,
            weights=np.asarray(model.coef_, dtype=np.float64).ravel(),
            intercept=float(model.intercept_),
        )
