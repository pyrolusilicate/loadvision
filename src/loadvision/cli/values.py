"""Values commands read from the command line and print back."""

from datetime import datetime

from loadvision.cli.errors import CommandError
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.timestamps import format_utc


def version_of(tag: str) -> ModelVersionId:
    """Reads a model version tag given on the command line or in the settings.

    Raises:
        CommandError: If the tag is not a valid version.
    """
    try:
        return ModelVersionId(tag)
    except ValueError as error:
        raise CommandError(f"invalid version {tag!r}: {error}") from error


def short_utc(value: datetime) -> str:
    """Writes a time for a table: UTC to the second, microseconds would not fit a column."""
    return format_utc(value.replace(microsecond=0))
