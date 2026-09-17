from pathlib import Path

import numpy as np
import pytest

from loadvision.ml.data.manifest import Sample
from loadvision.ml.evaluation.probe import knn_mae

NEIGHBOURS = (1, 2)


def sample(image_id: str, label: float | None, group: str) -> Sample:
    return Sample(
        image_id=image_id,
        path=Path(f"{image_id}.jpg"),
        sha256="a" * 64,
        label=label,
        group_id=group,
    )


@pytest.mark.ml
def test_probe_averages_the_labels_of_the_nearest_photos() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]], dtype=np.float32)
    samples = [
        sample("a", 10.0, "grp_1"),
        sample("b", 20.0, "grp_2"),
        sample("c", 80.0, "grp_3"),
        sample("d", 90.0, "grp_4"),
    ]

    scores = knn_mae(vectors, samples, NEIGHBOURS)

    # Each photo is nearest to the one of the other pair member: 10 vs 20, 80 vs 90.
    assert scores["mae_k1"] == pytest.approx(10.0)


@pytest.mark.ml
def test_photos_of_one_group_never_vote_for_each_other() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    together = [sample("a", 10.0, "grp_1"), sample("b", 10.0, "grp_1"), sample("c", 90.0, "grp_2")]
    apart = [sample("a", 10.0, "grp_1"), sample("b", 10.0, "grp_2"), sample("c", 90.0, "grp_3")]

    assert knn_mae(vectors, together, (1,))["mae_k1"] == pytest.approx(80.0)
    assert knn_mae(vectors, apart, (1,))["mae_k1"] == pytest.approx(80.0 / 3)


@pytest.mark.ml
def test_probe_refuses_photos_without_labels() -> None:
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="labelled"):
        knn_mae(vectors, [sample("a", None, "grp_1")], NEIGHBOURS)
