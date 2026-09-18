"""Keeping the served model on the version the database names active."""

import logging
import threading
import time
from collections.abc import Callable

from loadvision.application.load_served_model import LoadServedModel
from loadvision.domain.model_version import ModelVersionId
from loadvision.ml.inference.active_model import ActiveModel

_logger = logging.getLogger(__name__)


class ModelFollower:
    """Loads the version the database names active whenever the service answers with another.

    The database alone decides which version answers; a command, the API, the pages and the
    worker change it there. Every service process checks it every `check_seconds` and loads
    the new version in the background, so all of them follow without a restart. A version
    whose loading failed is tried again after `retry_seconds`, not on every check: a broken
    package is not reread over and over, and a package put in place later is still picked up.

    Args:
        model: The model the service answers with.
        load: Loads the active version.
        check_seconds: Pause between checks.
        retry_seconds: Pause before a version whose loading failed is tried again.
        clock: Monotonic seconds; tests pass their own.
    """

    def __init__(
        self,
        model: ActiveModel,
        load: LoadServedModel,
        *,
        check_seconds: float,
        retry_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model = model
        self._load = load
        self._check_seconds = check_seconds
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._guard = threading.Lock()
        self._loading: threading.Thread | None = None
        self._tried: tuple[ModelVersionId, float] | None = None
        self._stop = threading.Event()
        self._checks: threading.Thread | None = None

    def check(self) -> threading.Thread | None:
        """Starts loading the active version if the service does not answer with it.

        Returns:
            The loading thread, or None when there is nothing to load: the version already
            answers, another loading is running, or this version failed less than
            `retry_seconds` ago.

        Raises:
            Exception: Whatever the database raises when it cannot be read.
        """
        wanted = self._load.wanted_version()
        with self._guard:
            current = self._model.current()
            if current is not None and current.version == wanted:
                return None
            if self._loading is not None and self._loading.is_alive():
                return None
            now = self._clock()
            if self._recently_failed(wanted, now):
                return None
            self._tried = (wanted, now)
            _logger.info("loading model version %s", wanted.value)
            self._loading = self._model.load_in_background(self._load)
            return self._loading

    def start(self) -> None:
        """Checks at once, then every `check_seconds` in a background thread until `stop`."""
        self._checks = threading.Thread(target=self._follow, name="model-follower", daemon=True)
        self._checks.start()

    def stop(self) -> None:
        """Ends the checks; a loading that already runs finishes on its own."""
        self._stop.set()
        if self._checks is not None:
            self._checks.join()

    def _follow(self) -> None:
        while True:
            try:
                self.check()
            except Exception:
                # A database restart must not end the checks: the next one tries again.
                _logger.exception("model check failed")
            if self._stop.wait(self._check_seconds):
                return

    def _recently_failed(self, wanted: ModelVersionId, now: float) -> bool:
        # A version tried before and still not answering is a version whose loading failed.
        if self._tried is None:
            return False
        version, at = self._tried
        return version == wanted and now - at < self._retry_seconds
