from pathlib import Path

import numpy as np
import pytest

from loadvision.ml.postprocess import load_curves, make_chain
from loadvision.ml.training.postprocess_fit import FITTERS, fit_chain, fit_step


def hedged(photos: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Round labels over the whole range and predictions pulled towards the middle."""
    rng = np.random.default_rng(0)
    labels = rng.choice(np.arange(0, 101, 5).astype(float), size=photos)
    predictions = 50 + 0.6 * (labels - 50) + rng.normal(scale=4.0, size=photos)
    return predictions, labels


@pytest.mark.ml
def test_every_registry_step_can_be_fitted() -> None:
    from loadvision.ml.postprocess import APPLIERS

    assert set(FITTERS) == set(APPLIERS)


@pytest.mark.ml
def test_isotonic_reaches_the_ends_the_head_never_reached() -> None:
    predictions, labels = hedged()

    step = fit_step("isotonic", predictions, labels)
    corrected = step.apply(predictions)

    assert np.abs(labels - corrected).mean() < np.abs(labels - predictions).mean()
    assert corrected.max() > predictions.max()
    assert step.curves is not None


@pytest.mark.ml
def test_median_bins_lands_on_the_median_of_each_band() -> None:
    predictions = np.concatenate([np.full(50, 10.0), np.full(50, 90.0)]) + np.linspace(0, 1, 100)
    labels = np.concatenate([np.full(50, 5.0), np.full(50, 100.0)])

    corrected = fit_step("median_bins", predictions, labels, {"bins": 2.0}).apply(predictions)

    assert corrected[:50].mean() == pytest.approx(5.0, abs=1.0)
    assert corrected[50:].mean() == pytest.approx(100.0, abs=1.0)


@pytest.mark.ml
def test_snap_high_learns_a_threshold_and_stores_it_in_params() -> None:
    predictions = np.concatenate([np.full(50, 30.0), np.full(50, 93.0)])
    labels = np.concatenate([np.full(50, 30.0), np.full(50, 100.0)])

    step = fit_step("snap_high", predictions, labels)
    corrected = step.apply(predictions)

    assert "threshold" in step.params
    assert step.curves is None
    assert corrected[-1] == 100.0
    assert corrected[0] == 30.0


@pytest.mark.ml
def test_snap_low_learns_a_threshold() -> None:
    predictions = np.concatenate([np.full(50, 6.0), np.full(50, 70.0)])
    labels = np.concatenate([np.full(50, 0.0), np.full(50, 70.0)])

    corrected = fit_step("snap_low", predictions, labels).apply(predictions)

    assert corrected[0] == 0.0
    assert corrected[-1] == 70.0


@pytest.mark.ml
def test_linear_expand_learns_a_centre_and_a_factor() -> None:
    predictions, labels = hedged()

    step = fit_step("linear_expand", predictions, labels)
    corrected = step.apply(predictions)

    assert set(step.params) == {"centre", "factor"}
    assert step.params["factor"] > 1.0
    assert np.abs(labels - corrected).mean() < np.abs(labels - predictions).mean()


@pytest.mark.ml
def test_a_given_threshold_is_kept_instead_of_searched() -> None:
    predictions, labels = hedged()

    step = fit_step("snap_high", predictions, labels, {"threshold": 88.0})

    assert step.params["threshold"] == 88.0


@pytest.mark.ml
def test_a_chain_fits_each_step_on_the_output_of_the_previous_one() -> None:
    predictions, labels = hedged()

    chain = fit_chain([("isotonic", {}), ("round_to", {"step": 5.0})], predictions, labels)
    corrected = chain(predictions)

    assert [step.name for step in chain.steps] == ["isotonic", "round_to"]
    assert np.all(np.isclose(corrected % 5.0, 0.0))
    assert np.abs(labels - corrected).mean() < np.abs(labels - predictions).mean()


@pytest.mark.ml
def test_a_fitted_chain_and_the_service_chain_agree(tmp_path: Path) -> None:
    predictions, labels = hedged()
    steps = [("median_bins", {"bins": 8.0}), ("round_to", {"step": 5.0})]
    chain = fit_chain(steps, predictions, labels)

    # What the package would carry: params in the manifest, curves as files.
    rebuilt = []
    for index, step in enumerate(chain.steps):
        path = step.save_curves_to(tmp_path / f"step_{index}.npz")
        rebuilt.append((step.name, step.params, load_curves(path) if path else None))

    assert np.allclose(chain(predictions), make_chain(rebuilt)(predictions, {}))


@pytest.mark.ml
def test_a_chain_of_names_without_parameters_works() -> None:
    predictions, labels = hedged()

    corrected = fit_chain(["isotonic", "clip"], predictions, labels)(predictions)

    assert corrected.min() >= 0.0
    assert corrected.max() <= 100.0


@pytest.mark.ml
def test_unknown_step_is_rejected() -> None:
    with pytest.raises(KeyError):
        fit_step("magic", np.zeros(3), np.zeros(3))


@pytest.mark.ml
def test_disagreeing_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        fit_step("isotonic", np.zeros(3), np.zeros(4))


@pytest.mark.ml
@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("round_to", {"step": 0.0}, "step must be positive"),
        ("median_bins", {"bins": 1.0}, "at least two bins"),
    ],
)
def test_invalid_parameters_are_rejected(name: str, params: dict[str, float], message: str) -> None:
    predictions, labels = hedged()

    with pytest.raises(ValueError, match=message):
        fit_step(name, predictions, labels, params).apply(predictions)


@pytest.mark.ml
def test_isotonic_needs_more_than_one_distinct_prediction() -> None:
    with pytest.raises(ValueError, match="at least two distinct"):
        fit_step("isotonic", np.full(10, 50.0), np.arange(10, dtype=float))
