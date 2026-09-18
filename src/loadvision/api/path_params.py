"""Values the API takes from a URL path."""

from uuid import UUID

from loadvision.domain.model_version import ModelVersionId
from loadvision.errors import ErrorCode, LoadvisionError


def uuid_from_path(text: str, not_found: ErrorCode) -> UUID:
    """Reads a record id from a URL path.

    Only the canonical form, in any letter case, is accepted: `UUID` also parses braces and a
    ``urn:uuid:`` prefix, which would give one record several addresses.

    Args:
        text: Path segment as received.
        not_found: Code to answer with; no record has an id that is not a canonical UUID.

    Returns:
        The id.

    Raises:
        LoadvisionError: With `not_found` for anything else.
    """
    try:
        value = UUID(text)
    except ValueError:
        value = None
    if value is None or str(value) != text.lower():
        raise LoadvisionError(not_found, "malformed id")
    return value


def version_from_path(text: str) -> ModelVersionId:
    """Reads a model version tag from a URL path.

    Raises:
        LoadvisionError: `MODEL_VERSION_NOT_FOUND` for a malformed tag; no version has one.
    """
    try:
        return ModelVersionId(text)
    except ValueError as error:
        raise LoadvisionError(ErrorCode.MODEL_VERSION_NOT_FOUND, "malformed version") from error
