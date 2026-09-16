"""Behavior every LoadPredictor implementation must have: the fake now, model packages later."""

import pytest

from loadvision.application.ports import LoadPredictor
from loadvision.imaging.prepared_image import PreparedImage
from tests.builders import make_image


class LoadPredictorContract:
    @pytest.fixture
    def photos(self) -> list[PreparedImage]:
        return [make_image(b"photo-a"), make_image(b"photo-b")]

    @pytest.fixture
    def predictor(self, photos: list[PreparedImage]) -> LoadPredictor:
        """Predictor whose loads for `photos` differ, so a swapped order is visible."""
        raise NotImplementedError

    def test_empty_batch_gives_no_predictions(self, predictor: LoadPredictor) -> None:
        assert predictor.predict([]) == []

    def test_batch_matches_single_photo_predictions_in_order(
        self, predictor: LoadPredictor, photos: list[PreparedImage]
    ) -> None:
        singles = [predictor.predict([photo])[0].load_pct.value for photo in photos]
        batch = [prediction.load_pct.value for prediction in predictor.predict(photos)]
        reversed_batch = [
            prediction.load_pct.value for prediction in predictor.predict(photos[::-1])
        ]
        assert singles[0] != singles[1], "the fixture must give the photos different loads"
        assert batch == pytest.approx(singles, abs=1e-3)
        assert reversed_batch == pytest.approx(singles[::-1], abs=1e-3)
