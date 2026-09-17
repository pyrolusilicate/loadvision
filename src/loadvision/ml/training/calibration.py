"""Isotonic calibration: undoing the pull of a fitted head towards the middle."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

CALIBRATION_KEYS: Final = ("knots_in", "knots_out")

_MIN_LOAD: Final = 0.0
_MAX_LOAD: Final = 100.0
_MIN_POINTS: Final = 2


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """A monotone curve that maps a raw prediction to a corrected one.

    A head fitted by least squares hedges: it answers near the middle when unsure, so empty
    trucks come out too full and full trucks too empty. The curve is learned from out-of-fold
    predictions, which is the only place where that hedging is visible without cheating.

    Stored as two arrays instead of a pickled estimator, so a model package stays readable and
    loads without executing code.

    Attributes:
        knots_in: Raw predictions, ascending.
        knots_out: Corrected values, non-decreasing.
    """

    knots_in: NDArray[np.float64]
    knots_out: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validates the curve.

        Raises:
            ValueError: If the arrays disagree, are too short, or are not sorted.
        """
        if self.knots_in.shape != self.knots_out.shape:
            raise ValueError(f"knots disagree: {self.knots_in.shape} and {self.knots_out.shape}")
        if self.knots_in.size < _MIN_POINTS:
            raise ValueError("a calibration curve needs at least two points")
        if np.any(np.diff(self.knots_in) < 0) or np.any(np.diff(self.knots_out) < 0):
            raise ValueError("calibration knots must be non-decreasing")

    def apply(self, values: NDArray[Any]) -> NDArray[np.float64]:
        """Corrects predictions, clipping outside the fitted range to its ends.

        Args:
            values: Raw predictions.

        Returns:
            Corrected loads from 0 to 100.
        """
        corrected = np.interp(np.asarray(values, dtype=np.float64), self.knots_in, self.knots_out)
        return np.clip(corrected, _MIN_LOAD, _MAX_LOAD)

    def save(self, path: Path) -> None:
        """Writes the curve as an `.npz` file a model package can carry."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, knots_in=self.knots_in, knots_out=self.knots_out)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reads a curve written by `save`.

        Args:
            path: File to read.

        Returns:
            The calibrator.

        Raises:
            ValueError: If a key is missing or the curve is invalid.
        """
        with np.load(path, allow_pickle=False) as stored:
            missing = [key for key in CALIBRATION_KEYS if key not in stored]
            if missing:
                raise ValueError(f"{path.name}: missing {', '.join(missing)}")
            return cls(
                knots_in=np.asarray(stored["knots_in"], dtype=np.float64),
                knots_out=np.asarray(stored["knots_out"], dtype=np.float64),
            )

    @classmethod
    def fit(cls, predictions: NDArray[Any], labels: NDArray[Any]) -> Self:
        """Learns the curve from predictions a model made on photos it had not seen.

        Args:
            predictions: Out-of-fold predictions.
            labels: Reference loads for the same photos.

        Returns:
            The fitted calibrator.

        Raises:
            ValueError: If the inputs disagree or hold fewer than two distinct predictions.
        """
        raw = np.asarray(predictions, dtype=np.float64)
        target = np.asarray(labels, dtype=np.float64)
        if raw.shape != target.shape:
            raise ValueError(f"shapes differ: {raw.shape} and {target.shape}")
        if np.unique(raw).size < _MIN_POINTS:
            raise ValueError("calibration needs at least two distinct predictions")

        model = IsotonicRegression(y_min=_MIN_LOAD, y_max=_MAX_LOAD, out_of_bounds="clip")
        model.fit(raw, target)
        knots_in = np.unique(raw)
        return cls(
            knots_in=knots_in,
            knots_out=np.asarray(model.predict(knots_in), dtype=np.float64),
        )
