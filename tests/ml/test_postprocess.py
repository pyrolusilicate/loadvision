import numpy as np
import pytest

from loadvision.ml.postprocess import POSTPROCESSORS, fit_chain, fit_step, names


def hedged(photos: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Labels over the whole range and predictions pulled towards the middle, as a head does."""
    rng = np.random.default_rng(0)
    labels = rng.choice(np.arange(0, 101, 5).astype(float), size=photos)
    predictions = 50 + 0.6 * (labels - 50) + rng.normal(scale=4.0, size=photos)
    return predictions, labels


@pytest.mark.ml
def test_registry_lists_every_step() -> None:
    assert set(names()) == set(POSTPROCESSORS)
    assert names() == tuple(sorted(names()))


@pytest.mark.ml
def test_clip_keeps_loads_inside_the_range() -> None:
    step = fit_step("clip", np.zeros(3), np.zeros(3))

    assert list(step(np.array([-10.0, 50.0, 150.0]))) == [0.0, 50.0, 100.0]


@pytest.mark.ml
def test_round_to_moves_to_the_nearest_step() -> None:
    step = fit_step("round_to", np.zeros(3), np.zeros(3), {"step": 5.0})

    assert list(step(np.array([12.0, 13.0, 97.6]))) == [10.0, 15.0, 100.0]


@pytest.mark.ml
def test_isotonic_reaches_the_ends_the_head_never_reached() -> None:
    predictions, labels = hedged()

    step = fit_step("isotonic", predictions, labels)
    corrected = step(predictions)

    assert np.abs(labels - corrected).mean() < np.abs(labels - predictions).mean()
    assert corrected.max() > predictions.max()


@pytest.mark.ml
def test_median_bins_lands_on_the_median_of_each_band() -> None:
    predictions = np.concatenate([np.full(50, 10.0), np.full(50, 90.0)]) + np.linspace(0, 1, 100)
    labels = np.concatenate([np.full(50, 5.0), np.full(50, 100.0)])

    step = fit_step("median_bins", predictions, labels, {"bins": 2.0})
    corrected = step(predictions)

    assert corrected[:50].mean() == pytest.approx(5.0, abs=1.0)
    assert corrected[50:].mean() == pytest.approx(100.0, abs=1.0)


@pytest.mark.ml
def test_snap_high_learns_a_threshold_that_helps() -> None:
    predictions = np.concatenate([np.full(50, 30.0), np.full(50, 93.0)])
    labels = np.concatenate([np.full(50, 30.0), np.full(50, 100.0)])

    step = fit_step("snap_high", predictions, labels)
    corrected = step(predictions)

    assert np.abs(labels - corrected).mean() < np.abs(labels - predictions).mean()
    assert corrected[-1] == 100.0
    assert corrected[0] == 30.0


@pytest.mark.ml
def test_snap_low_learns_a_threshold_that_helps() -> None:
    predictions = np.concatenate([np.full(50, 6.0), np.full(50, 70.0)])
    labels = np.concatenate([np.full(50, 0.0), np.full(50, 70.0)])

    corrected = fit_step("snap_low", predictions, labels)(predictions)

    assert corrected[0] == 0.0
    assert corrected[-1] == 70.0


@pytest.mark.ml
def test_linear_expand_pushes_predictions_apart() -> None:
    predictions, labels = hedged()

    corrected = fit_step("linear_expand", predictions, labels)(predictions)

    assert corrected.std() > predictions.std()
    assert np.abs(labels - corrected).mean() < np.abs(labels - predictions).mean()


@pytest.mark.ml
def test_a_chain_applies_its_steps_in_order() -> None:
    predictions, labels = hedged()

    chain = fit_chain([("isotonic", {}), ("round_to", {"step": 5.0})], predictions, labels)
    corrected = chain(predictions)

    assert np.all(np.isclose(corrected % 5.0, 0.0))
    assert np.abs(labels - corrected).mean() < np.abs(labels - predictions).mean()


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
        fit_step(name, predictions, labels, params)
