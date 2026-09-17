import pytest

from tests.api.service import PHOTO_SHA256, Service, make_service
from tests.fakes.predictor import FakePredictor


@pytest.fixture
def service() -> Service:
    return make_service(FakePredictor(loads={PHOTO_SHA256: 85.25}))
