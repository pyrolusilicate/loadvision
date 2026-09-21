"""LoadPredictor for tests that needs no model."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from loadvision.application.ports import Prediction
from loadvision.domain.assessment import LoadPercent, ReviewReason
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage

# The two ends of the scale of a package that measures distances at all: where its training
# photos end and where frames with nothing in them land. Tests that care set their own.
OOD_THRESHOLD: Final = 0.2
OOD_UNUSABLE: Final = 1.0


def band_line(share: float) -> float:
    """Returns where a line at `share` of the fake model's band stands."""
    return OOD_THRESHOLD + (OOD_UNUSABLE - OOD_THRESHOLD) * share


class FakePredictor:
    """Returns preset or hash-derived loads and records every batch it receives.

    Args:
        version: Version tag reported by the predictor.
        loads: Load per photo SHA-256; other photos get a load derived from their hash.
        ood_score: Distance to the training photos reported for every photo, or None when the
            model does not measure it.
        ood_threshold: Distance from which this model calls a scene unusual; the near end of
            its scale, as a real package carries it in its manifest.
        ood_unusable: Distance at which frames with nothing in them land; the far end, which
            a real package measures on itself when it loads.
        review_reasons: Reasons the model reports with every prediction.
        error: Exception raised for every batch, to test model failures.
    """

    def __init__(
        self,
        *,
        version: str = "fake-1",
        loads: Mapping[str, float] | None = None,
        ood_score: float | None = None,
        ood_threshold: float | None = OOD_THRESHOLD,
        ood_unusable: float | None = OOD_UNUSABLE,
        review_reasons: Iterable[ReviewReason] = (),
        error: Exception | None = None,
    ) -> None:
        self._version = ModelVersionId(version)
        self._loads = dict(loads or {})
        self._ood_score = ood_score
        self._ood_threshold = ood_threshold
        self._ood_unusable = ood_unusable
        self._reasons = frozenset(review_reasons)
        self._error = error
        self.batches: list[list[str]] = []

    @property
    def version(self) -> ModelVersionId:
        return self._version

    def predict(self, images: Sequence[PreparedImage]) -> list[Prediction]:
        self.batches.append([image.sha256 for image in images])
        if self._error is not None:
            raise self._error
        return [
            Prediction(
                LoadPercent(self._load_for(image.sha256)),
                ood_score=self._ood_score,
                ood_threshold=self._ood_threshold,
                ood_unusable=self._ood_unusable,
                review_reasons=self._reasons,
            )
            for image in images
        ]

    def _load_for(self, sha256: str) -> float:
        if sha256 in self._loads:
            return self._loads[sha256]
        # Derived from the hash: different photos get different loads that repeat across runs.
        return float(int(sha256[:8], 16) % 101)
