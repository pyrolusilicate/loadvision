from pathlib import Path

import pytest

from loadvision.application.ports import ExplanationKind
from loadvision.domain.model_version import ModelVersionId
from loadvision.infrastructure.storage.heatmap_store import HeatmapStore

SHA256 = "a" * 64
V1 = ModelVersionId("v1")
V2 = ModelVersionId("v2")
PNG = b"\x89PNG drawn"


@pytest.mark.unit
def test_a_stored_map_comes_back(tmp_path: Path) -> None:
    store = HeatmapStore(tmp_path)

    store.write(SHA256, V1, ExplanationKind.ATTENTION, PNG)

    assert store.read(SHA256, V1, ExplanationKind.ATTENTION) == PNG


@pytest.mark.unit
def test_a_map_nobody_drew_is_none(tmp_path: Path) -> None:
    assert HeatmapStore(tmp_path).read(SHA256, V1, ExplanationKind.ATTENTION) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("version", "kind"),
    [(V2, ExplanationKind.ATTENTION), (V1, ExplanationKind.CONTRIBUTION)],
    ids=["another model version", "another kind of map"],
)
def test_every_version_and_kind_keeps_its_own_map(
    tmp_path: Path, version: ModelVersionId, kind: ExplanationKind
) -> None:
    store = HeatmapStore(tmp_path)
    store.write(SHA256, V1, ExplanationKind.ATTENTION, PNG)

    assert store.read(SHA256, version, kind) is None


@pytest.mark.unit
def test_a_map_of_the_same_photo_is_replaced_not_doubled(tmp_path: Path) -> None:
    store = HeatmapStore(tmp_path)

    store.write(SHA256, V1, ExplanationKind.ATTENTION, PNG)
    store.write(SHA256, V1, ExplanationKind.ATTENTION, b"drawn again")

    assert store.read(SHA256, V1, ExplanationKind.ATTENTION) == b"drawn again"
    assert len([path for path in tmp_path.rglob("*") if path.is_file()]) == 1


@pytest.mark.unit
@pytest.mark.parametrize("sha256", ["0" * 63, "A" * 64, "../../etc/passwd"])
def test_a_hash_that_is_not_a_hash_is_refused(tmp_path: Path, sha256: str) -> None:
    store = HeatmapStore(tmp_path)

    with pytest.raises(ValueError, match="64 lowercase hex characters"):
        store.read(sha256, V1, ExplanationKind.ATTENTION)
