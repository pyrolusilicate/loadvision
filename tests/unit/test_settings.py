from pathlib import Path

import pytest
from pydantic import ValidationError

from loadvision.imaging.prepared_image import MediaType
from loadvision.settings import (
    CONFIG_PATH_ENV,
    DATABASE_URL_ENV,
    DEFAULT_CONFIG_PATH,
    database_url,
    load_settings,
)

MEBIBYTE = 1024 * 1024


@pytest.mark.unit
def test_repository_config_loads_with_limits_in_validator_units() -> None:
    settings = load_settings(DEFAULT_CONFIG_PATH)
    limits = settings.images.limits()
    assert limits.max_bytes == settings.images.max_decoded_mb * MEBIBYTE
    assert limits.max_pixels == settings.images.max_megapixels * 1_000_000
    assert limits.min_side == settings.images.min_side_px
    assert limits.media_types == frozenset(MediaType)
    assert settings.api.max_body_bytes == settings.api.max_body_mb * MEBIBYTE
    assert settings.model_dir == settings.storage.models_dir / settings.model.version


@pytest.mark.unit
def test_config_path_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moved = tmp_path / "app.yaml"
    moved.write_text(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace("baseline-v0", "v7"),
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_PATH_ENV, str(moved))
    assert load_settings().model.version == "v7"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ("max_body_mb: 20", "max_body_mb: 20\n  max_body_gb: 1", "max_body_gb"),
        ("max_megapixels: 50", "max_megapixels: 0", "max_megapixels"),
        ("allowed_formats: [jpeg, png, webp]", "allowed_formats: [jpeg, heic]", "allowed_formats"),
        ("allowed_formats: [jpeg, png, webp]", "allowed_formats: []", "allowed_formats"),
    ],
)
def test_wrong_setting_is_rejected(tmp_path: Path, old: str, new: str, field: str) -> None:
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert old in text
    broken = tmp_path / "app.yaml"
    broken.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ValidationError, match=field):
        load_settings(broken)


@pytest.mark.unit
def test_database_url_must_be_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)
    with pytest.raises(RuntimeError, match=DATABASE_URL_ENV):
        database_url()
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql+psycopg://u@h/db")
    assert database_url() == "postgresql+psycopg://u@h/db"
