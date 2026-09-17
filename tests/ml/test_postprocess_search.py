import numpy as np
import pytest

from loadvision.ml.evaluation.postprocess_search import CHAINS, cross_fitted
from loadvision.ml.postprocess import POSTPROCESSORS, fit_chain

FOLDS = [0, 0, 0, 1, 1, 1]


@pytest.mark.ml
def test_every_candidate_chain_names_known_steps() -> None:
    steps = {step for chain in CHAINS.values() for step, _ in chain}
    assert steps <= set(POSTPROCESSORS)


@pytest.mark.ml
def test_a_fold_is_corrected_by_a_chain_fitted_without_it() -> None:
    # Every fold is corrected by a chain fitted on the other one, never on itself.
    raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    labels = np.array([20.0, 30.0, 40.0, 40.0, 50.0, 60.0])

    chain = [("isotonic", {})]

    corrected = cross_fitted(chain, raw, labels, FOLDS)

    assert corrected[:3] == pytest.approx(fit_chain(chain, raw[3:], labels[3:])(raw[:3]))
    assert corrected[3:] == pytest.approx(fit_chain(chain, raw[:3], labels[:3])(raw[3:]))


@pytest.mark.ml
def test_rounding_chain_moves_every_prediction_onto_the_grid() -> None:
    raw = np.array([11.0, 22.0, 33.0, 44.0, 56.0, 67.0])
    labels = np.array([10.0, 20.0, 35.0, 45.0, 55.0, 65.0])

    corrected = cross_fitted([("round_to", {"step": 5.0})], raw, labels, FOLDS)

    assert np.allclose(corrected % 5.0, 0.0)


@pytest.mark.ml
def test_unknown_step_is_refused() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.raises(KeyError):
        cross_fitted([("polish", {})], raw, raw, FOLDS)
