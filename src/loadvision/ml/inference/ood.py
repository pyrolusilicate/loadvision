"""How far a photo is from the ones the model was trained on.

A service that answers confidently on a scene it has never seen is worse than one that says so.
The score is the mean distance to the nearest training embeddings, and the reference file holds
those embeddings together with the name and width of the encoder that produced them, so a
package that pairs the reference with another member fails to load instead of scoring nonsense.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

REFERENCE_KEYS: Final = ("vectors", "backbone", "dim")

_MIN_NEIGHBOURS: Final = 1


@dataclass(frozen=True, slots=True)
class OodReference:
    """Training embeddings a photo is compared against.

    Attributes:
        backbone: Registry key of the encoder that produced them.
        vectors: Unit-length embeddings, one row per training photo.
    """

    backbone: str
    vectors: NDArray[np.float32]

    def score(self, embedding: NDArray[Any], neighbours: int) -> float:
        """Mean cosine distance from a photo to its nearest training photos.

        Args:
            embedding: Embedding of the photo, from the same encoder.
            neighbours: How many nearest training photos to average over.

        Returns:
            Score from 0 for a photo just like the training ones, upwards.

        Raises:
            ValueError: If the embedding has another width or `neighbours` is not positive.
        """
        if embedding.shape[-1] != self.vectors.shape[1]:
            raise ValueError(
                f"embedding is {embedding.shape[-1]} wide, reference is {self.vectors.shape[1]}"
            )
        if neighbours < _MIN_NEIGHBOURS:
            raise ValueError(f"neighbours must be positive, got {neighbours}")
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        unit = vector / norm if norm > 0 else vector
        similarity = self.vectors @ unit
        taken = min(neighbours, similarity.size)
        nearest = np.partition(similarity, -taken)[-taken:]
        return float(np.clip(1.0 - nearest.mean(), 0.0, 2.0))

    def save(self, path: Path) -> None:
        """Writes the reference as the `.npz` file a package carries."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=self.vectors,
            backbone=np.array(self.backbone),
            dim=np.array(self.vectors.shape[1]),
        )

    @classmethod
    def load(cls, path: Path, backbone: str) -> Self:
        """Reads a reference and checks it belongs to the member that will use it.

        Args:
            path: File inside a model package.
            backbone: Registry key of the member named by `ood.member`.

        Returns:
            The reference.

        Raises:
            ValueError: If a key is missing, the file is empty, or it was built with another
                encoder than the member it is paired with.
        """
        with np.load(path, allow_pickle=False) as stored:
            missing = [key for key in REFERENCE_KEYS if key not in stored]
            if missing:
                raise ValueError(f"{path.name}: missing {', '.join(missing)}")
            vectors = np.asarray(stored["vectors"], dtype=np.float32)
            written = str(stored["backbone"])
            dim = int(stored["dim"])
        if vectors.ndim != 2 or vectors.size == 0:
            raise ValueError(f"{path.name}: reference holds no embeddings")
        if vectors.shape[1] != dim:
            raise ValueError(f"{path.name}: says {dim} wide, holds {vectors.shape[1]}")
        if written != backbone:
            raise ValueError(f"{path.name}: built with {written!r}, paired with {backbone!r}")
        return cls(backbone=written, vectors=vectors)

    @classmethod
    def build(cls, backbone: str, embeddings: NDArray[Any]) -> Self:
        """Normalizes training embeddings into a reference.

        Args:
            backbone: Registry key of the encoder that produced them.
            embeddings: One row per training photo.

        Returns:
            The reference, stored as float16 rows of unit length; a megabyte per encoder.

        Raises:
            ValueError: If there are no embeddings.
        """
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError("a reference needs at least one embedding")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        unit = matrix / np.where(norms > 0, norms, 1.0)
        return cls(backbone=backbone, vectors=unit.astype(np.float16).astype(np.float32))
