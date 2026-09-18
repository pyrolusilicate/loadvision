"""Values the API takes from a URL path."""

from uuid import UUID

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
