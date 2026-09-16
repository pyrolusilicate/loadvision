import pytest

from loadvision.application.ports import ImageStorage, LoadPredictor, UnitOfWorkFactory
from loadvision.imaging.prepared_image import PreparedImage
from tests.fakes.image_storage import InMemoryImageStorage
from tests.fakes.in_memory_db import InMemoryStore
from tests.fakes.predictor import FakePredictor
from tests.port_contracts.image_storage import ImageStorageContract
from tests.port_contracts.load_predictor import LoadPredictorContract
from tests.port_contracts.unit_of_work import UnitOfWorkContract


@pytest.mark.unit
class TestInMemoryUnitOfWork(UnitOfWorkContract):
    @pytest.fixture
    def uow_factory(self) -> UnitOfWorkFactory:
        return InMemoryStore().unit_of_work


@pytest.mark.unit
class TestInMemoryImageStorage(ImageStorageContract):
    @pytest.fixture
    def storage(self) -> ImageStorage:
        return InMemoryImageStorage()


@pytest.mark.unit
class TestFakePredictor(LoadPredictorContract):
    @pytest.fixture
    def predictor(self, photos: list[PreparedImage]) -> LoadPredictor:
        return FakePredictor(loads={photos[0].sha256: 10.0, photos[1].sha256: 90.0})
