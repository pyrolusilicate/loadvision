import numpy as np
import pytest

from loadvision.ml.evaluation.cross_validation import out_of_fold
from loadvision.ml.training.head_trainer import FittedHead


def constant_fit(seen: list[int]):
    """Fit function that remembers how many photos it was trained on and predicts their mean."""

    def fit(_features: np.ndarray, labels: np.ndarray):
        seen.append(len(labels))
        mean = float(labels.mean())
        return lambda batch: np.full(batch.shape[0], mean)

    return fit


@pytest.mark.ml
def test_every_photo_is_predicted_by_a_model_that_did_not_see_it() -> None:
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    labels = np.arange(10, dtype=np.float64)
    folds = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    seen: list[int] = []

    result = out_of_fold(features, labels, folds, constant_fit(seen))

    assert result.predictions.shape == labels.shape
    assert np.isfinite(result.predictions).all()
    # Five models, each trained on the other four folds.
    assert seen == [8, 8, 8, 8, 8]


@pytest.mark.ml
def test_metrics_cover_the_whole_set_and_every_fold() -> None:
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    labels = np.arange(10, dtype=np.float64)
    folds = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    result = out_of_fold(features, labels, folds, constant_fit([]))

    assert result.folds == tuple(folds)
    assert "mae" in result.metrics
    assert {f"mae_fold{fold}" for fold in range(5)} <= set(result.metrics)


@pytest.mark.ml
def test_a_real_head_beats_predicting_the_mean() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(100, 4))
    labels = np.clip(50 + 25 * features[:, 0], 0, 100)
    folds = [index % 5 for index in range(100)]

    trained = out_of_fold(features, labels, folds, lambda a, b: FittedHead.fit(a, b).predict)
    baseline = out_of_fold(features, labels, folds, constant_fit([]))

    assert trained.metrics["mae"] < baseline.metrics["mae"]


@pytest.mark.ml
def test_one_fold_is_not_a_split() -> None:
    features = np.zeros((4, 2))
    labels = np.zeros(4)

    with pytest.raises(ValueError, match="at least two folds"):
        out_of_fold(features, labels, [0, 0, 0, 0], constant_fit([]))


@pytest.mark.ml
def test_disagreeing_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="disagree"):
        out_of_fold(np.zeros((4, 2)), np.zeros(3), [0, 1, 0, 1], constant_fit([]))


@pytest.mark.ml
def test_a_predictor_that_skips_photos_is_caught() -> None:
    features = np.zeros((4, 2))
    labels = np.zeros(4)

    def broken(_features: np.ndarray, _labels: np.ndarray):
        return lambda batch: np.full(batch.shape[0], np.nan)

    with pytest.raises(ValueError, match="without a prediction"):
        out_of_fold(features, labels, [0, 0, 1, 1], broken)
