"""Settings file for tests: the repository config with the folders moved into a temp folder."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from loadvision.settings import CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH, Settings


def read_settings(**sections: dict[str, Any]) -> Settings:
    """Returns the repository settings with some values changed.

    Args:
        **sections: Values to change by section, such as `storage` or `training`; keys a
            section does not name keep the repository values.
    """
    return Settings.model_validate(_changed(sections))


def use_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **sections: dict[str, Any]
) -> Path:
    """Writes a settings file and points `LOADVISION_CONFIG` at it.

    Args:
        tmp_path: Folder of the test.
        monkeypatch: Fixture that sets the environment variable.
        **sections: Values to change by section, such as `storage` or `runs`; keys a section
            does not name keep the repository values.

    Returns:
        Path of the written file.
    """
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(_changed(sections)), encoding="utf-8")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(path))
    return path


def _changed(sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # A test changes the values it is about; a setting added later keeps its repository value.
    settings = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for name, values in sections.items():
        settings[name] = {**settings.get(name, {}), **values}
    return settings
