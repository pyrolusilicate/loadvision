"""Behavior every ImageStorage implementation must have."""

import pytest

from loadvision.application.ports import ImageStorage
from tests.builders import make_image


class ImageStorageContract:
    @pytest.fixture
    def storage(self) -> ImageStorage:
        raise NotImplementedError

    def test_unknown_photo_is_absent(self, storage: ImageStorage) -> None:
        assert storage.load("0" * 64) is None

    def test_saved_photo_loads_back_by_hash(self, storage: ImageStorage) -> None:
        image = make_image()
        storage.save(image)
        assert storage.load(image.sha256) == image.raw

    def test_saving_the_same_photo_again_changes_nothing(self, storage: ImageStorage) -> None:
        image = make_image()
        storage.save(image)
        storage.save(make_image())
        assert storage.load(image.sha256) == image.raw

    def test_photos_are_kept_apart(self, storage: ImageStorage) -> None:
        first = make_image(b"photo-a")
        second = make_image(b"photo-b")
        storage.save(first)
        storage.save(second)
        assert storage.load(first.sha256) == first.raw
        assert storage.load(second.sha256) == second.raw
