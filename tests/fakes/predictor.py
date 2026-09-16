"""LoadPredictor for tests that needs no model."""

from collections.abc import Mapping, Sequence

from loadvision.application.ports import Prediction
from loadvision.domain.assessment import LoadPercent
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage


class FakePredictor:
    """Returns preset or hash-derived loads and records every batch it receives.

    Args:
        version: Version tag reported by the predictor.
        loads: Load per photo SHA-256; other photos get a load derived from their hash.
        error: Exception raised for every batch, to test model failures.
    """

    def __init__(
        self,
        *,
        version: str = "fake-1",
        loads: Mapping[str, float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._version = ModelVersionId(version)
        self._loads = dict(loads or {})
        self._error = error
        self.batches: list[list[str]] = []

    @property
    def version(self) -> ModelVersionId:
        return self._version

    def predict(self, images: Sequence[PreparedImage]) -> list[Prediction]:
        self.batches.append([image.sha256 for image in images])
        if self._error is not None:
            raise self._error
        return [Prediction(LoadPercent(self._load_for(image.sha256))) for image in images]

    def _load_for(self, sha256: str) -> float:
        if sha256 in self._loads:
            return self._loads[sha256]
        # Derived from the hash: different photos get different loads that repeat across runs.
        return float(int(sha256[:8], 16) % 101)
