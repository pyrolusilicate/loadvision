"""The explainer of the active model: built when a person first asks to see a map."""

import logging
import threading
from collections.abc import Callable

from loadvision.application.ports import ExplanationKind, LoadExplainer
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage

_logger = logging.getLogger(__name__)


class _OneAtATime:
    """`LoadExplainer` that lets a single map be drawn at a time."""

    def __init__(self, explainer: LoadExplainer, lock: threading.Lock) -> None:
        self._explainer = explainer
        self._lock = lock

    @property
    def version(self) -> ModelVersionId:
        return self._explainer.version

    def explain(self, image: PreparedImage, kind: ExplanationKind) -> bytes:
        with self._lock:
            return self._explainer.explain(image, kind)


class ActiveExplainer:
    """Holds the explainer of one model version and builds it on the first request.

    The explainer carries its own copy of the encoder, so a service whose maps nobody opens
    never pays for it. Drawing runs one at a time for the same reason predictions do, but
    behind a lock of its own: a person looking at a map must not hold up assessments.

    Args:
        load: Builds the explainer of a model version; the caller decides where packages live.
    """

    def __init__(self, load: Callable[[ModelVersionId], LoadExplainer]) -> None:
        self._load = load
        self._build_lock = threading.Lock()
        self._draw_lock = threading.Lock()
        self._current: LoadExplainer | None = None

    def of(self, version: ModelVersionId) -> LoadExplainer:
        """Returns the explainer of that model version, building it if it is not the current one.

        Args:
            version: Version of the model that made the prediction being explained.

        Returns:
            The explainer, which draws one map at a time.

        Raises:
            Exception: Whatever loading the package raises; the caller reports it.
        """
        with self._build_lock:
            if self._current is None or self._current.version != version:
                _logger.info("loading the explainer of %s", version.value)
                self._current = self._load(version)
        return _OneAtATime(self._current, self._draw_lock)
