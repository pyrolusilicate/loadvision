"""How high the cargo is stacked, measured with the depth model a package already carries.

The share of the body's height a stack reaches is the one factor of a weight that a photo can
answer. It is read off the depth map in `ml/inference/cargo.py`; this is the thin piece that
loads that map for a model version and keeps it between requests, the way the explainer does.

The depth model is about 95 MB, far less than an encoder, and it is built when an operator
first asks for a weight rather than at startup.
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage
from loadvision.ml.inference.cargo import stack_share
from loadvision.ml.packaging.model_package import ModelPackage

_logger = logging.getLogger(__name__)


class StackMeter:
    """Measures the stack of one photo with the depth model of one package.

    Args:
        depth: Loaded depth encoder of that package.
        rows: Rows of the grid the package pools depth into.
        cols: Columns of that grid.
        version: Version of the package the measurement belongs to.
    """

    def __init__(self, depth: Any, rows: int, cols: int, version: ModelVersionId) -> None:
        self._depth = depth
        self._rows = rows
        self._cols = cols
        self._version = version

    @property
    def version(self) -> ModelVersionId:
        """Version of the model package the stack is measured with."""
        return self._version

    def measure(self, image: PreparedImage) -> float:
        """Returns the share of the body's height the cargo reaches, 0 when nothing stands out."""
        grid = self._depth.grid(image, self._rows, self._cols)
        return stack_share(grid, self._rows, self._cols)


def load_stack_meter(models_dir: Path, device: str, version: ModelVersionId) -> StackMeter | None:
    """Builds the stack meter of a model package.

    Args:
        models_dir: Folder with the model packages.
        device: Torch device the depth model runs on.
        version: Version to measure with.

    Returns:
        The meter, or None when that package carries no depth model and the stack cannot be
        read off a photo at all.

    Raises:
        ModelPackageError: If the package is missing or does not load.
    """
    # Imported here and not at the top: the module pulls in torch, and a service that weighs
    # nothing must start without paying for it.
    from loadvision.ml.models.depth import DepthEncoder

    package = ModelPackage.load(models_dir / version.value)
    config = package.manifest.depth
    if config is None:
        return None
    depth = DepthEncoder.load(package.root / config.model, device)
    return StackMeter(depth, config.rows, config.cols, package.version)


class ActiveStackMeter:
    """Holds the stack meter of one model version and builds it on the first weighing.

    Measuring runs one at a time, behind a lock of its own: an operator weighing a photo must
    not hold up assessments.

    Args:
        load: Builds the meter of a model version, or returns None when it has no depth model.
    """

    def __init__(self, load: Callable[[ModelVersionId], StackMeter | None]) -> None:
        self._load = load
        self._build_lock = threading.Lock()
        self._measure_lock = threading.Lock()
        self._current: StackMeter | None = None

    def measure(self, image: PreparedImage, version: ModelVersionId) -> float | None:
        """Returns the share of the body's height the cargo reaches on this photo.

        Args:
            image: Decoded photo, the same pixels the prediction was made on.
            version: Version of the model that answered for it.

        Returns:
            The share, or None when the package carries no depth model.

        Raises:
            Exception: Whatever loading the package raises; the caller reports it.
        """
        with self._build_lock:
            if self._current is None or self._current.version != version:
                _logger.info("loading the stack meter of %s", version.value)
                self._current = self._load(version)
            meter = self._current
        if meter is None:
            return None
        with self._measure_lock:
            return meter.measure(image)
