from pathlib import Path

import pytest

from loadvision.application.ports import ExplanationKind
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.validator import ImageLimits, prepare_image
from loadvision.infrastructure.storage.heatmap_cache import CachedExplainer
from tests import image_files

LIMITS = ImageLimits(
    max_bytes=2_000_000, max_pixels=1_000_000, min_side=256, media_types=frozenset(MediaType)
)
PHOTO = prepare_image(image_files.jpeg(), LIMITS)


class CountingExplainer:
    """Draws a different picture for every call and counts them."""

    def __init__(self, version: str = "v1") -> None:
        self._version = ModelVersionId(version)
        self.calls = 0

    @property
    def version(self) -> ModelVersionId:
        return self._version

    def explain(self, image: PreparedImage, kind: ExplanationKind) -> bytes:
        self.calls += 1
        return f"{self._version.value}-{kind.value}-{self.calls}".encode()


@pytest.mark.unit
def test_a_map_is_drawn_once_and_read_afterwards(tmp_path: Path) -> None:
    inner = CountingExplainer()
    cached = CachedExplainer(inner, tmp_path)

    first = cached.explain(PHOTO, ExplanationKind.ATTENTION)
    second = cached.explain(PHOTO, ExplanationKind.ATTENTION)

    assert (first, second) == (b"v1-attention-1", b"v1-attention-1")
    assert inner.calls == 1


@pytest.mark.unit
def test_every_kind_and_every_version_keeps_its_own_map(tmp_path: Path) -> None:
    attention = CachedExplainer(CountingExplainer(), tmp_path).explain(
        PHOTO, ExplanationKind.ATTENTION
    )
    contribution = CachedExplainer(CountingExplainer(), tmp_path).explain(
        PHOTO, ExplanationKind.CONTRIBUTION
    )
    newer = CachedExplainer(CountingExplainer("v2"), tmp_path).explain(
        PHOTO, ExplanationKind.ATTENTION
    )

    assert attention == b"v1-attention-1"
    assert contribution == b"v1-contribution-1"
    assert newer == b"v2-attention-1"


@pytest.mark.unit
def test_a_failed_drawing_leaves_nothing_behind(tmp_path: Path) -> None:
    class Failing(CountingExplainer):
        def explain(self, image: PreparedImage, kind: ExplanationKind) -> bytes:
            raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        CachedExplainer(Failing(), tmp_path).explain(PHOTO, ExplanationKind.ATTENTION)

    assert list(tmp_path.rglob("*")) == []


@pytest.mark.unit
@pytest.mark.parametrize("sha256", ["0" * 63, "A" * 64, "../../etc/passwd"])
def test_a_hash_that_is_not_a_hash_is_refused(tmp_path: Path, sha256: str) -> None:
    cached = CachedExplainer(CountingExplainer(), tmp_path)

    with pytest.raises(ValueError, match="64 lowercase hex characters"):
        cached.path(sha256, ExplanationKind.ATTENTION)
