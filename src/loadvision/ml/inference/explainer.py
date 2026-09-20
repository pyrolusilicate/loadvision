"""LoadExplainer implementation backed by a model package.

The drawing itself lives in `ml/inference/explain.py`; this is the thin piece that gives it the
shape the application asks for and names the model version the maps belong to.
"""

import io
from pathlib import Path
from typing import Any, Final

from PIL import Image

from loadvision.application.ports import ExplanationKind, LoadExplainer
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage
from loadvision.ml.packaging.model_package import ModelPackage

# A map is a photo with a coloured layer over it, and the pages scale it to the screen anyway.
# Measured on a map of a 1440x1920 photo: PNG as drawn is 5352 KB, this is 538 KB and costs
# 91 ms of re-encoding, which the store pays once.
MAP_MAX_SIDE: Final = 1600
MAP_QUALITY: Final = 85


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
        """Returns the photo with the map drawn over it, as JPEG bytes."""
        return _for_a_page(bytes(self._explainer.picture(image, kind.value)))


def _for_a_page(picture: bytes) -> bytes:
    """Fits a drawn map into a size a browser is happy to download."""
    with Image.open(io.BytesIO(picture)) as drawn:
        drawn.thumbnail((MAP_MAX_SIDE, MAP_MAX_SIDE))
        buffer = io.BytesIO()
        drawn.convert("RGB").save(buffer, "JPEG", quality=MAP_QUALITY, progressive=True)
    return buffer.getvalue()


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
    from loadvision.ml.inference.explain import Explainer

    package = ModelPackage.load(models_dir / version.value)
    return PackageExplainer(Explainer.from_package(package, device), package.version)
