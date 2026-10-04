"""Candidate postprocessing chains and the honest way to measure them."""

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.postprocess import Facts
from loadvision.ml.training.postprocess_fit import fit_chain

type Chain = Sequence[tuple[str, Mapping[str, float]]]

# Две популяции фото различаются пропорциями кадра: у 0.70-0.80 доля меток «ровно 100» втрое
# выше. Границы замерены и записаны в docs/ml/experiments.md.
_BY_ASPECT: Final[Mapping[str, float]] = {"bins": 16.0, "low": 0.7, "high": 0.8}

# Chains worth comparing; the winner goes into an experiment config as the postprocess chain.
CHAINS: Final[Mapping[str, Chain]] = {
    "none": [("clip", {})],
    "round5": [("round_to", {"step": 5.0})],
    "expand": [("linear_expand", {})],
    "isotonic": [("isotonic", {})],
    "median16": [("median_bins", {"bins": 16.0})],
    "snap_high": [("snap_high", {})],
    "iso+snap_high": [("isotonic", {}), ("snap_high", {})],
    "iso+snap_both": [("isotonic", {}), ("snap_high", {}), ("snap_low", {})],
    "iso+round5": [("isotonic", {}), ("round_to", {"step": 5.0})],
    "iso+snap_high+round5": [
        ("isotonic", {}),
        ("snap_high", {}),
        ("round_to", {"step": 5.0}),
    ],
    "median16+snap_high": [("median_bins", {"bins": 16.0}), ("snap_high", {})],
    "median16+round5": [("median_bins", {"bins": 16.0}), ("round_to", {"step": 5.0})],
    # Условные шаги: по кривой на популяцию фото. Это то, что стоит в пакете.
    "median16_by_aspect": [("median_bins_by_aspect", dict(_BY_ASPECT))],
    "median16_by_aspect+round5": [
        ("median_bins_by_aspect", dict(_BY_ASPECT)),
        ("round_to", {"step": 5.0}),
    ],
    "median16+snap_high+round5": [
        ("median_bins", {"bins": 16.0}),
        ("snap_high", {}),
        ("round_to", {"step": 5.0}),
    ],
    "iso+round10": [("isotonic", {}), ("round_to", {"step": 10.0})],
    "expand+snap_high": [("linear_expand", {}), ("snap_high", {})],
}


def cross_fitted(
    chain: Chain,
    predictions: NDArray[np.float64],
    labels: NDArray[np.float64],
    folds: Sequence[int],
    facts: Facts | None = None,
) -> NDArray[np.float64]:
    """Applies a chain fitted on the other folds, so it never sees the photos it is judged on.

    A threshold or a curve fitted on all photos would look better than it is; here every fold
    is corrected by a chain fitted without it.

    Args:
        chain: Steps with their parameters.
        predictions: Raw out-of-fold predictions.
        labels: Reference loads.
        folds: Fold number of every photo.
        facts: What is known about each photo before the prediction, such as the proportions
            of the frame. Conditional steps keep one curve per population and refuse to run
            without them, so a sweep that omits the facts cannot measure the chain the package
            actually ships.

    Returns:
        Corrected predictions, one per photo.

    Raises:
        KeyError: If a step is not in the postprocessing registry.
        ValueError: If a step is conditional and `facts` does not carry what it reads.
    """
    corrected = np.empty_like(predictions)
    numbers = np.asarray(folds)
    for fold in sorted(set(folds)):
        validation = numbers == fold
        apply = fit_chain(
            chain,
            predictions[~validation],
            labels[~validation],
            _pick(facts, ~validation),
        )
        corrected[validation] = apply(predictions[validation], _pick(facts, validation))
    return corrected


def _pick(facts: Facts | None, mask: NDArray[np.bool_]) -> dict[str, NDArray[np.float64]] | None:
    return None if facts is None else {name: values[mask] for name, values in facts.items()}
