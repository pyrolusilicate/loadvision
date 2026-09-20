"""LoadExplainer implementation backed by a model package.

The drawing itself lives in `ml/inference/explain.py`; this is the thin piece that gives it the
shape the application asks for and names the model version the maps belong to.
"""

from pathlib import Path
from typing import Any

from loadvision.application.ports import ExplanationKind, LoadExplainer
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage
from loadvision.ml.packaging.model_package import ModelPackage


class PackageExplainer:
    """Draws the maps of one model package.

    Args:
        explainer: The drawing itself, built from the same package the predictor runs.
        version: Version of that package.
    """

    def __init__(self, explainer: Any, version: ModelVersionId) -> None:
        self._explainer = explainer
        self._version = version

    @property
    def version(self) -> ModelVersionId:
        """Version of the model package the maps are drawn with."""
        return self._version

    def explain(self, image: PreparedImage, kind: ExplanationKind) -> bytes:
        """Returns the photo with the map drawn over it, as PNG bytes."""
        return bytes(self._explainer.picture(image, kind.value))


def load_explainer(models_dir: Path, device: str, version: ModelVersionId) -> LoadExplainer:
    """Builds the explainer of a model package.

    It keeps a copy of the encoder of its own, about 350 MB, so it is built when a person first
    asks to see a map and not at startup.

    Args:
        models_dir: Folder with the model packages.
        device: Torch device the encoder runs on.
        version: Version to draw with.

    Returns:
        The explainer.

    Raises:
        ModelPackageError: If the package is missing or does not load.
    """
    # Imported here and not at the top: the module pulls in torch, and a service that draws no
    # maps must start without paying for it.
    # The ignore goes away with the merge of the branch that brings `explain.py`; until then
    # this build has no drawing, and a map request fails with MODEL_INFERENCE_FAILED.
    from loadvision.ml.inference.explain import (  # pyright: ignore[reportMissingImports]
        Explainer,
    )

    package = ModelPackage.load(models_dir / version.value)
    return PackageExplainer(Explainer.from_package(package, device), package.version)
