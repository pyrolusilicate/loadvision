"""Relative depth of a photo from Depth Anything V2, pooled into a coarse grid."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
import torch
from numpy.typing import NDArray

from loadvision.imaging.prepared_image import PreparedImage
from loadvision.imaging.preprocessing import preprocess
from loadvision.ml.data.manifest import Sample
from loadvision.ml.training.embedding_cache import read_image

# Apache-2.0; the Base and Large checkpoints are CC-BY-NC and may not ship.
DEPTH_MODEL: Final = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_LICENSE: Final = "Apache-2.0"
DEPTH_ORIGIN: Final = "Depth Anything V2 Small, TikTok and HKU"
GRID_ROWS: Final = 16
GRID_COLS: Final = 12
DEPTH_DIM: Final = GRID_ROWS * GRID_COLS

# The DINOv2 box and normalization the checkpoint was trained with, 3:4 like the photos.
_PARAMS: Final = {
    "height": 518,
    "width": 392,
    "mean_r": 0.485,
    "mean_g": 0.456,
    "mean_b": 0.406,
    "std_r": 0.229,
    "std_g": 0.224,
    "std_b": 0.225,
}
_BATCH: Final = 16

# ponytail: one process-wide cache, so five folds and the test split pay for each photo once.
_CACHE: dict[Path, NDArray[np.float32]] = {}


class DepthEncoder:
    """Depth Anything V2 Small, frozen, turning a photo into a coarse grid of relative depth.

    The model gives depth up to an unknown scale, so every grid is divided by its own maximum:
    only the shape of the depth within the photo carries meaning.

    Args:
        model: Loaded depth model in eval mode.
        device: Torch device it runs on.
    """

    def __init__(self, model: Any, device: str) -> None:
        self._model = model
        self._device = device

    @classmethod
    def load(cls, source: str | Path = DEPTH_MODEL, device: str = "cpu") -> Self:
        """Loads the depth model from the hub or from a folder a package carries.

        Args:
            source: Hub name or a local folder with the checkpoint.
            device: Torch device.

        Returns:
            The encoder.
        """
        # Imported here: transformers comes with the ML group, and a package of plain torch
        # members must load without it.
        from transformers import AutoModelForDepthEstimation

        model = AutoModelForDepthEstimation.from_pretrained(str(source)).to(device).eval()
        return cls(model, device)

    def grids(self, images: Sequence[PreparedImage]) -> NDArray[np.float32]:
        """Returns one grid per photo, in the order given.

        Args:
            images: Decoded photos.

        Returns:
            Array of shape (photos, `DEPTH_DIM`), values 0 to 1, larger means nearer.
        """
        rows: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for start in range(0, len(images), _BATCH):
                batch = np.stack(
                    [
                        preprocess("resize_pad", image, _PARAMS)
                        for image in images[start : start + _BATCH]
                    ]
                )
                depth = self._model(
                    pixel_values=torch.from_numpy(batch).to(self._device)
                ).predicted_depth
                # Pooling into a grid that does not divide the input evenly runs on the CPU:
                # MPS has no kernel for it.
                grid = torch.nn.functional.adaptive_avg_pool2d(
                    depth.float().cpu().unsqueeze(1), (GRID_ROWS, GRID_COLS)
                ).flatten(1)
                rows.append((grid / grid.amax(dim=1, keepdim=True).clamp_min(1e-6)).numpy())
        return np.concatenate(rows).astype(np.float32)

    def grid(self, image: PreparedImage) -> NDArray[np.float32]:
        """Returns the grid of one photo, the way the service asks for it."""
        return self.grids([image])[0]


def depth_grids(samples: Sequence[Sample], device: str) -> NDArray[np.float32]:
    """Returns the depth grid of every training photo, computing only the ones not seen yet.

    Args:
        samples: Photos of the dataset.
        device: Torch device the depth model runs on.

    Returns:
        Array of shape (photos, `DEPTH_DIM`), rows in the order of `samples`.
    """
    missing = [sample.path for sample in samples if sample.path not in _CACHE]
    if missing:
        encoder = DepthEncoder.load(device=device)
        for start in range(0, len(missing), _BATCH):
            paths = missing[start : start + _BATCH]
            computed = encoder.grids([read_image(path) for path in paths])
            _CACHE.update(dict(zip(paths, computed, strict=True)))
        del encoder
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return np.stack([_CACHE[sample.path] for sample in samples])


def mirror_grids(grids: torch.Tensor) -> torch.Tensor:
    """Reverses the columns of flattened grids, for a photo that was mirrored."""
    return grids.reshape(-1, GRID_ROWS, GRID_COLS).flip(-1).reshape(grids.shape[0], -1)
