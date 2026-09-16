import importlib

import pytest

LAYER_PACKAGES = [
    "loadvision.domain",
    "loadvision.application",
    "loadvision.infrastructure",
    "loadvision.ml",
    "loadvision.api",
    "loadvision.cli",
]


@pytest.mark.unit
@pytest.mark.parametrize("package", LAYER_PACKAGES)
def test_layer_package_is_importable(package: str) -> None:
    importlib.import_module(package)
