import numpy as np
import pytest

from loadvision.ml.evaluation.oof_heads import knn_predictor, ridge_predictor


@pytest.fixture
def features() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])


@pytest.mark.ml
def test_ridge_head_learns_a_linear_relation() -> None:
    features = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    labels = 100.0 * features.ravel()

    predict = ridge_predictor(0.001)(features, labels)

    assert predict(np.array([[0.5]]))[0] == pytest.approx(50.0, abs=1.0)


@pytest.mark.ml
def test_stronger_penalty_pulls_the_ridge_head_towards_the_mean() -> None:
    features = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    labels = 100.0 * features.ravel()
    query = np.array([[1.0]])

    weak = ridge_predictor(0.001)(features, labels)(query)[0]
    strong = ridge_predictor(1000.0)(features, labels)(query)[0]

    assert labels.mean() < strong < weak


@pytest.mark.ml
def test_knn_head_averages_the_labels_of_the_nearest_photos(features: np.ndarray) -> None:
    labels = np.array([10.0, 20.0, 80.0, 90.0])

    predict = knn_predictor(2)(features, labels)

    assert predict(np.array([[1.0, 0.05]]))[0] == pytest.approx(15.0)
    assert predict(np.array([[0.05, 1.0]]))[0] == pytest.approx(85.0)


@pytest.mark.ml
def test_knn_head_ignores_the_length_of_a_vector(features: np.ndarray) -> None:
    labels = np.array([10.0, 20.0, 80.0, 90.0])
    predict = knn_predictor(1)(features, labels)

    assert predict(np.array([[3.0, 0.0]]))[0] == predict(np.array([[0.01, 0.0]]))[0]
