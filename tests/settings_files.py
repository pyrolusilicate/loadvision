"""Settings file for tests: the repository config with the folders moved into a temp folder."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from loadvision.settings import CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH


def use_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **sections: dict[str, Any]
) -> Path:
    """Writes a settings file and points `LOADVISION_CONFIG` at it.

    Args:
        tmp_path: Folder of the test.
        monkeypatch: Fixture that sets the environment variable.
        **sections: Sections to replace, such as `storage` or `runs`.

    Returns:
        Path of the written file.
    """
    settings = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    settings.update(sections)
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(path))
    return path
