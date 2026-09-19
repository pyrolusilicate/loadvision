"""Relative depth of a photo from Depth Anything V2, pooled into a coarse grid."""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray

from loadvision.imaging.preprocessing import preprocess
from loadvision.ml.data.manifest import Sample
from loadvision.ml.training.embedding_cache import read_image

# Apache-2.0; the Base and Large checkpoints are CC-BY-NC and may not ship.
DEPTH_MODEL: Final = "depth-anything/Depth-Anything-V2-Small-hf"
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


def depth_grids(samples: Sequence[Sample], device: str) -> NDArray[np.float32]:
    """Returns the depth grid of every photo, computing only the ones not seen yet.

    Every grid is divided by its own maximum: the model gives depth up to an unknown scale,
    so only its shape within the photo carries meaning.

    Args:
        samples: Photos.
        device: Torch device the depth model runs on.

    Returns:
        Array of shape (photos, `DEPTH_DIM`), rows in the order of `samples`, values 0 to 1,
        where larger means nearer to the camera.
    """
    missing = [sample.path for sample in samples if sample.path not in _CACHE]
    if missing:
        from transformers import AutoModelForDepthEstimation

        model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
        with torch.inference_mode():
            for start in range(0, len(missing), _BATCH):
                paths = missing[start : start + _BATCH]
                batch = np.stack([preprocess("resize_pad", read_image(p), _PARAMS) for p in paths])
                depth = model(pixel_values=torch.from_numpy(batch).to(device)).predicted_depth
                grid = torch.nn.functional.adaptive_avg_pool2d(
                    depth.float().cpu().unsqueeze(1), (GRID_ROWS, GRID_COLS)
                ).flatten(1)
                grid = grid / grid.amax(dim=1, keepdim=True).clamp_min(1e-6)
                for path, row in zip(paths, grid.numpy(), strict=True):
                    _CACHE[path] = row.astype(np.float32)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return np.stack([_CACHE[sample.path] for sample in samples])


def mirror_grids(grids: torch.Tensor) -> torch.Tensor:
    """Reverses the columns of flattened grids, for a photo that was mirrored."""
    return grids.reshape(-1, GRID_ROWS, GRID_COLS).flip(-1).reshape(grids.shape[0], -1)
