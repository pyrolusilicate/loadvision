import os
from pathlib import Path

import pytest

from loadvision.application.ports import ImageStorage
from loadvision.infrastructure.storage.local_storage import LocalImageStorage
from tests.builders import make_image
from tests.port_contracts.image_storage import ImageStorageContract


def files_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


@pytest.mark.unit
class TestLocalImageStorage(ImageStorageContract):
    @pytest.fixture
    def storage(self, tmp_path: Path) -> ImageStorage:
        return LocalImageStorage(tmp_path / "images")


@pytest.mark.unit
def test_photo_is_stored_under_its_hash_in_a_two_character_folder(tmp_path: Path) -> None:
    image = make_image()
    LocalImageStorage(tmp_path).save(image)
    path = tmp_path / image.sha256[:2] / image.sha256
    assert files_under(tmp_path) == [path]
    assert path.read_bytes() == image.raw


@pytest.mark.unit
def test_existing_photo_file_is_not_written_again(tmp_path: Path) -> None:
    storage = LocalImageStorage(tmp_path)
    image = make_image()
    storage.save(image)
    path = storage.path(image.sha256)
    os.utime(path, (0, 0))
    storage.save(image)
    assert path.stat().st_mtime == 0


@pytest.mark.unit
def test_failed_write_leaves_neither_photo_nor_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(self: Path, target: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail)
    storage = LocalImageStorage(tmp_path)
    image = make_image()
    with pytest.raises(OSError, match="disk full"):
        storage.save(image)
    assert files_under(tmp_path) == []
    assert storage.load(image.sha256) is None


@pytest.mark.unit
@pytest.mark.parametrize("sha256", ["../../etc/passwd", "A" * 64, "a" * 63, "a" * 65, ""])
def test_rejects_a_value_that_is_not_a_hash(tmp_path: Path, sha256: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        LocalImageStorage(tmp_path).load(sha256)
