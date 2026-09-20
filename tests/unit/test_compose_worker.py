"""The worker container must see every folder a run reads from the repository."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from loadvision.ml.pipeline.stages.build_package import LICENSES_DIR

COMPOSE = Path(__file__).parents[2] / "compose.yaml"


def worker_mounts() -> set[str]:
    """Paths inside the worker container that come from the repository."""
    compose: dict[str, Any] = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    volumes: list[str] = compose["services"]["worker"]["volumes"]
    return {volume.split(":")[1] for volume in volumes}


@pytest.mark.unit
def test_the_worker_sees_the_licence_texts_a_package_must_ship() -> None:
    # Without them the build stops at the last stage, after all the counting is done.
    assert f"/app/{LICENSES_DIR}" in worker_mounts()


@pytest.mark.unit
@pytest.mark.parametrize("folder", ["data", "runs", "storage", "configs"])
def test_the_worker_sees_the_folders_a_run_works_with(folder: str) -> None:
    assert f"/app/{folder}" in worker_mounts()
