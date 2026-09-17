from pathlib import Path
from typing import Any

import numpy as np
import pytest

from loadvision.ml.data.manifest import Sample
from loadvision.ml.evaluation import ensemble as ensemble_module
from loadvision.ml.evaluation.ensemble import Member, ensemble_out_of_fold

FOLDS = [0, 0, 0, 0, 1, 1, 1, 1]
LABELS = np.array([0.0, 20.0, 40.0, 60.0, 10.0, 30.0, 50.0, 70.0])


class FakeBackbone:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dim = 1


def sample(index: int) -> Sample:
    return Sample(
        image_id=f"img_{index}",
        path=Path(f"img_{index}.jpg"),
        sha256="a" * 64,
        label=float(LABELS[index]),
        group_id=f"grp_{index}",
    )


@pytest.fixture
def embeddings(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replaces the encoders: every member sees the features the test gives it."""
    asked: dict[str, Any] = {"devices": [], "cache_dirs": [], "encoders": []}

    def load_backbone(name: str, device: str) -> FakeBackbone:
        asked["encoders"].append(name)
        asked["devices"].append(device)
        return FakeBackbone(name)

    def embed_samples(
        samples: list[Sample], backbone: FakeBackbone, *, cache_dir: Path, device: str
    ) -> np.ndarray:
        asked["cache_dirs"].append(cache_dir)
        # The members see different features, not a shifted copy of the same one: the head
        # standardizes its input, so a shift or a scale would give identical predictions.
        values = (
            [float(index) for index, _ in enumerate(samples)]
            if backbone.name == "first"
            else [float((index * 7) % 5) for index, _ in enumerate(samples)]
        )
        return np.array([[value] for value in values], dtype=np.float32)

    monkeypatch.setattr(ensemble_module, "load_backbone", load_backbone)
    monkeypatch.setattr(ensemble_module, "embed_samples", embed_samples)
    return asked


@pytest.mark.ml
@pytest.mark.parametrize(
    ("text", "member"),
    [
        ("dinov3_vitb16:200", Member(encoder="dinov3_vitb16", alpha=200.0)),
        ("siglip2_vitb16:0.5", Member(encoder="siglip2_vitb16", alpha=0.5)),
    ],
)
def test_member_is_read_from_encoder_and_penalty(text: str, member: Member) -> None:
    assert Member.parse(text) == member
    assert str(Member.parse(text)) == text


@pytest.mark.ml
@pytest.mark.parametrize("text", ["dinov3_vitb16", ":200", "dinov3_vitb16:alpha"])
def test_malformed_member_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        Member.parse(text)


@pytest.mark.ml
def test_ensemble_averages_its_members_and_reports_each_of_them(
    embeddings: dict[str, Any],
) -> None:
    samples = [sample(index) for index in range(len(LABELS))]
    members = [Member(encoder="first", alpha=1.0), Member(encoder="second", alpha=1.0)]

    result = ensemble_out_of_fold(
        samples, LABELS, FOLDS, members, device="cpu", cache_dir=Path("cache")
    )

    assert [part.member for part in result.members] == members
    assert embeddings["encoders"] == ["first", "second"]
    assert embeddings["devices"] == ["cpu", "cpu"]
    assert embeddings["cache_dirs"] == [Path("cache"), Path("cache")]
    first, second = (part.predictions for part in result.members)
    assert not np.allclose(first, second), "the members must disagree, or the test proves nothing"
    assert result.predictions == pytest.approx((first + second) / 2)
    assert all(part.mae >= 0 for part in result.members)


@pytest.mark.ml
def test_an_ensemble_needs_a_member(embeddings: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least one member"):
        ensemble_out_of_fold([], LABELS, FOLDS, [], device="cpu", cache_dir=Path("cache"))
