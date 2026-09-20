"""Drawing maps of what the model saw, without a model."""

import io

from PIL import Image

from loadvision.application.ports import ExplanationKind
from loadvision.domain.model_version import ModelVersionId


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (200, 40, 40)).save(buffer, "PNG")
    return buffer.getvalue()


PNG = _png()


class FakeHeatmaps:
    """Returns one picture for every map and records what was asked for."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, ExplanationKind, ModelVersionId]] = []

    def __call__(self, sha256: str, kind: ExplanationKind, version: ModelVersionId) -> bytes:
        self.asked.append((sha256, kind, version))
        return PNG
