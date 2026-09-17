"""kNN probe: how much load information frozen embeddings already carry, without a head."""

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.data.manifest import Sample

type Vectors = NDArray[np.float32] | NDArray[np.float64]


def knn_mae(
    vectors: Vectors, samples: list[Sample], neighbours: tuple[int, ...]
) -> dict[str, float]:
    """Mean absolute error of averaging the labels of the k nearest photos of other groups.

    A photo may not vote for itself, and neither may another photo of the same truck: the split
    rule keeps whole groups together, so the probe respects it too.

    Args:
        vectors: Embedding per photo, in the order of `samples`.
        samples: Labelled photos with their groups.
        neighbours: Values of k to report.

    Returns:
        `mae_k<k>` for every k.

    Raises:
        ValueError: If a sample has no label.
    """
    if any(sample.label is None for sample in samples):
        raise ValueError("the probe needs labelled photos")
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = np.array([sample.label for sample in samples], dtype=np.float64)
    groups = np.array([sample.group_id for sample in samples])

    similarity = unit @ unit.T
    similarity[groups[:, None] == groups[None, :]] = -1.0
    ranked = np.argsort(-similarity, axis=1)
    return {
        f"mae_k{k}": float(np.abs(labels - labels[ranked[:, :k]].mean(axis=1)).mean())
        for k in neighbours
    }
