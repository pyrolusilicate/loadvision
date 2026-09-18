"""Values commands read from the command line."""

from loadvision.cli.errors import CommandError
from loadvision.domain.model_version import ModelVersionId


def version_of(tag: str) -> ModelVersionId:
    """Reads a model version tag given on the command line or in the settings.

    Raises:
        CommandError: If the tag is not a valid version.
    """
    try:
        return ModelVersionId(tag)
    except ValueError as error:
        raise CommandError(f"invalid version {tag!r}: {error}") from error
