"""The model the service answers with: loaded in the background, used by one request at a time."""

import logging
import threading
from collections.abc import Callable, Sequence

from loadvision.application.ports import LoadPredictor, Prediction
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage

_logger = logging.getLogger(__name__)


class _OneAtATime:
    """`LoadPredictor` that lets a single prediction run at a time."""

    def __init__(self, predictor: LoadPredictor, lock: threading.Lock) -> None:
        self._predictor = predictor
        self._lock = lock

    @property
    def version(self) -> ModelVersionId:
        return self._predictor.version

    def predict(self, images: Sequence[PreparedImage]) -> list[Prediction]:
        with self._lock:
            return self._predictor.predict(images)


class ActiveModel:
    """Holds the predictor the service answers with.

    Predictions run one at a time: a torch model is not written for concurrent calls, and a
    single prediction gets every CPU core. A batcher (task 3.8) can later take this place
    behind the same `LoadPredictor` port.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: LoadPredictor | None = None
        self._failure: str | None = None

    def current(self) -> LoadPredictor | None:
        """Returns the predictor for one request, or None while no model is loaded.

        The caller keeps the returned predictor for the whole request, so the version stored
        with an assessment is the version that predicted, even if another model is activated
        meanwhile.
        """
        predictor = self._current
        return None if predictor is None else _OneAtATime(predictor, self._lock)

    @property
    def failure(self) -> str | None:
        """Why the last loading failed, or None."""
        return self._failure

    def activate(self, predictor: LoadPredictor) -> None:
        """Makes the predictor current; requests that already took the previous one finish on it.

        Args:
            predictor: Loaded model.
        """
        self._current = predictor
        self._failure = None

    def load_in_background(self, load: Callable[[], LoadPredictor]) -> threading.Thread:
        """Starts loading a model in a background thread.

        The service answers while the model loads. A failure is logged and kept in `failure`;
        the current model, if any, stays.

        Args:
            load: Builds the predictor, for example from a model package.

        Returns:
            The started thread.
        """
        thread = threading.Thread(target=self._load, args=(load,), name="model-loader", daemon=True)
        thread.start()
        return thread

    def _load(self, load: Callable[[], LoadPredictor]) -> None:
        try:
            predictor = load()
        except Exception as error:
            self._failure = f"{type(error).__name__}: {error}"
            _logger.exception("model loading failed")
            return
        self.activate(predictor)
