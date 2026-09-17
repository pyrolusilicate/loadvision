from pathlib import Path

import numpy as np
import pytest

from loadvision.ml.inference.predictor import RidgeHead
from loadvision.ml.training.head_trainer import FittedHead


def linear_data(photos: int = 200, features: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings whose first feature carries the load, plus noise in the rest."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(photos, features))
    labels = np.clip(50 + 25 * matrix[:, 0], 0, 100)
    return matrix, labels


@pytest.mark.ml
def test_fitted_head_recovers_a_linear_signal() -> None:
    matrix, labels = linear_data()

    head = FittedHead.fit(matrix, labels, alpha=0.1)

    assert float(np.abs(head.predict(matrix) - labels).mean()) < 1.0


@pytest.mark.ml
def test_predictions_stay_inside_the_allowed_range() -> None:
    matrix, labels = linear_data()
    head = FittedHead.fit(matrix, labels, alpha=0.1)
    extreme = np.full((3, matrix.shape[1]), 50.0)

    predictions = head.predict(extreme)

    assert predictions.min() >= 0.0
    assert predictions.max() <= 100.0


@pytest.mark.ml
def test_a_constant_feature_does_not_divide_by_zero() -> None:
    matrix, labels = linear_data()
    matrix[:, 1] = 7.0

    head = FittedHead.fit(matrix, labels)

    assert np.isfinite(head.predict(matrix)).all()
    assert head.scale[1] == 1.0


@pytest.mark.ml
def test_stronger_penalty_shrinks_the_weights() -> None:
    matrix, labels = linear_data()

    weak = FittedHead.fit(matrix, labels, alpha=0.1)
    strong = FittedHead.fit(matrix, labels, alpha=1000.0)

    assert float(np.abs(strong.weights).sum()) < float(np.abs(weak.weights).sum())


@pytest.mark.ml
def test_saved_head_is_readable_by_the_service(tmp_path: Path) -> None:
    matrix, labels = linear_data()
    head = FittedHead.fit(matrix, labels)
    path = tmp_path / "head.npz"

    head.save(path)
    loaded = RidgeHead.load(path)

    assert np.allclose(loaded.mean, head.mean)
    assert np.allclose(loaded.scale, head.scale)
    assert np.allclose(loaded.weights, head.weights)
    assert loaded.intercept == pytest.approx(head.intercept)


@pytest.mark.ml
def test_feature_count_mismatch_is_rejected() -> None:
    matrix, labels = linear_data()
    head = FittedHead.fit(matrix, labels)

    with pytest.raises(ValueError, match="was fitted on 8 features"):
        head.predict(np.zeros((2, 5)))


@pytest.mark.ml
def test_fitting_on_disagreeing_shapes_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot fit"):
        FittedHead.fit(np.zeros((4, 3)), np.zeros(5))
