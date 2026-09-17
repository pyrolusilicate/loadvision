"""Typed application settings loaded from YAML and environment variables."""

import os
from pathlib import Path
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from loadvision.imaging.prepared_image import MediaType
from loadvision.imaging.validator import ImageLimits

# Connection string of the service database, such as postgresql+psycopg://user:password@host/db;
# kept in the environment, never in a committed file.
DATABASE_URL_ENV: Final = "DATABASE_URL"
# Path of the settings file, for a deployment that keeps it outside the repository.
CONFIG_PATH_ENV: Final = "LOADVISION_CONFIG"
DEFAULT_CONFIG_PATH: Final = Path("configs/app.yaml")

_MEBIBYTE: Final = 1024 * 1024
_MEGAPIXEL: Final = 1_000_000
_FORMATS: Final = {"jpeg": MediaType.JPEG, "png": MediaType.PNG, "webp": MediaType.WEBP}

type ImageFormat = Literal["jpeg", "png", "webp"]


class _Section(BaseModel):
    # A misspelled key is an error, not a silently ignored setting.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiSettings(_Section):
    """HTTP API settings.

    Attributes:
        max_body_mb: Largest accepted request body, in mebibytes.
        request_id_header: Header that carries the request id both ways.
    """

    max_body_mb: int = Field(gt=0)
    request_id_header: str = Field(min_length=1)

    @property
    def max_body_bytes(self) -> int:
        """Largest accepted request body, in bytes."""
        return self.max_body_mb * _MEBIBYTE


class ImageSettings(_Section):
    """Limits of uploaded photos.

    Attributes:
        max_decoded_mb: Largest photo file after base64 decoding, in mebibytes.
        max_megapixels: Largest width times height, in millions of pixels.
        min_side_px: Smallest shorter side, in pixels.
        allowed_formats: Accepted formats.
    """

    max_decoded_mb: int = Field(gt=0)
    max_megapixels: int = Field(gt=0)
    min_side_px: int = Field(gt=0)
    allowed_formats: tuple[ImageFormat, ...] = Field(min_length=1)

    def limits(self) -> ImageLimits:
        """Returns the limits in the units the image validator works with."""
        return ImageLimits(
            max_bytes=self.max_decoded_mb * _MEBIBYTE,
            max_pixels=self.max_megapixels * _MEGAPIXEL,
            min_side=self.min_side_px,
            media_types=frozenset(_FORMATS[name] for name in self.allowed_formats),
        )


class StorageSettings(_Section):
    """Folders of the files the service keeps.

    Attributes:
        images_dir: Uploaded photos by SHA-256.
        models_dir: Model packages, one folder per version.
    """

    images_dir: Path
    models_dir: Path


class LoggingSettings(_Section):
    """Log output.

    Attributes:
        format: `json` for machines, `console` for people.
        level: Lowest level written.
    """

    format: Literal["json", "console"]
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class RunsSettings(_Section):
    """Pipeline runs.

    Attributes:
        dir: Folder with one subfolder per run.
    """

    dir: Path


class ModelSettings(_Section):
    """Model the service starts with.

    Attributes:
        version: Version the service activates while the database has no active version; once
            one is active, the database decides. The package is the folder of that name in the
            models folder.
    """

    version: str = Field(min_length=1)


class Settings(_Section):
    """All settings of `configs/app.yaml`."""

    api: ApiSettings
    images: ImageSettings
    storage: StorageSettings
    logging: LoggingSettings
    runs: RunsSettings
    model: ModelSettings


def load_settings(path: Path | None = None) -> Settings:
    """Reads the settings file.

    Args:
        path: Settings file; by default the path from `LOADVISION_CONFIG`, then
            `configs/app.yaml`.

    Returns:
        Validated settings.

    Raises:
        OSError: If the file cannot be read.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If a setting is missing, unknown or out of range.
    """
    config_path = path or Path(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))
    with config_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return Settings.model_validate(data)


def database_url() -> str:
    """Returns the database connection string from the environment.

    Raises:
        RuntimeError: If `DATABASE_URL` is not set.
    """
    url = os.environ.get(DATABASE_URL_ENV)
    if not url:
        raise RuntimeError(f"{DATABASE_URL_ENV} is not set; see .env.example")
    return url
