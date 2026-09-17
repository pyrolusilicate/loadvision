import threading
import time
from collections.abc import Sequence

import pytest

from loadvision.application.ports import LoadPredictor, Prediction
from loadvision.domain.assessment import LoadPercent
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage
from loadvision.ml.inference.active_model import ActiveModel
from tests.builders import make_image
from tests.fakes.predictor import FakePredictor


class SlowPredictor:
    """Records how many predictions ran at the same time."""

    def __init__(self) -> None:
        self.running = 0
        self.most_at_once = 0
        self._guard = threading.Lock()

    @property
    def version(self) -> ModelVersionId:
        return ModelVersionId("slow-1")

    def predict(self, images: Sequence[PreparedImage]) -> list[Prediction]:
        with self._guard:
            self.running += 1
            self.most_at_once = max(self.most_at_once, self.running)
        time.sleep(0.02)
        with self._guard:
            self.running -= 1
        return [Prediction(LoadPercent(50.0)) for _ in images]


@pytest.mark.unit
def test_no_predictor_before_a_model_is_loaded() -> None:
    model = ActiveModel()
    assert (model.current(), model.failure) == (None, None)


@pytest.mark.unit
def test_activated_model_predicts_with_its_version() -> None:
    model = ActiveModel()
    model.activate(FakePredictor(version="fake-7"))
    predictor = model.current()
    assert predictor is not None
    assert predictor.version == ModelVersionId("fake-7")
    assert len(predictor.predict([make_image()])) == 1


@pytest.mark.unit
def test_background_loading_activates_the_model() -> None:
    model = ActiveModel()
    model.load_in_background(lambda: FakePredictor(version="fake-3")).join(timeout=5)
    current = model.current()
    assert current is not None
    assert current.version == ModelVersionId("fake-3")


@pytest.mark.unit
def test_failed_loading_keeps_the_previous_model_and_the_reason() -> None:
    def broken() -> LoadPredictor:
        raise FileNotFoundError("storage/models/v9/manifest.json")

    model = ActiveModel()
    model.activate(FakePredictor(version="fake-1"))
    model.load_in_background(broken).join(timeout=5)
    current = model.current()
    assert current is not None
    assert current.version == ModelVersionId("fake-1")
    assert model.failure == "FileNotFoundError: storage/models/v9/manifest.json"


@pytest.mark.unit
def test_predictions_run_one_at_a_time() -> None:
    slow = SlowPredictor()
    model = ActiveModel()
    model.activate(slow)
    predictors = [model.current() for _ in range(4)]
    threads = [
        threading.Thread(target=predictor.predict, args=([make_image()],))
        for predictor in predictors
        if predictor is not None
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert slow.most_at_once == 1


@pytest.mark.unit
def test_request_keeps_the_model_it_took_when_another_is_activated() -> None:
    model = ActiveModel()
    model.activate(FakePredictor(version="fake-1"))
    taken = model.current()
    model.activate(FakePredictor(version="fake-2"))
    assert taken is not None
    assert taken.version == ModelVersionId("fake-1")
