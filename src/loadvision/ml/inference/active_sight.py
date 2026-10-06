"""The model that names the kind of cargo on a photo, built when an operator first asks for it."""

import logging
import threading
from collections.abc import Callable

from loadvision.domain.cargo_weight import CargoKind
from loadvision.imaging.prepared_image import PreparedImage
from loadvision.ml.inference.cargo import CargoSight

_logger = logging.getLogger(__name__)


class ActiveCargoSight:
    """Holds the cargo-kind model and builds it on the first weighing that asks for it.

    The model is half a billion parameters and answers in seconds, so a service whose operators
    never ask for the kind to be read off a photo never pays for it. It reads one photo at a
    time, behind a lock of its own: an operator weighing must not hold up assessments.

    Args:
        load: Builds the model.
    """

    def __init__(self, load: Callable[[], CargoSight]) -> None:
        self._load = load
        self._build_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._current: CargoSight | None = None

    def kind(self, image: PreparedImage) -> CargoKind:
        """Returns which kind of cargo the photo shows.

        Raises:
            Exception: Whatever loading or running the model raises; the caller reports it.
        """
        with self._build_lock:
            if self._current is None:
                _logger.info("loading the cargo-kind model")
                self._current = self._load()
            sight = self._current
        with self._read_lock:
            return sight.kind(image)
