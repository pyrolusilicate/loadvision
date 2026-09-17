"""XML response class."""

from collections.abc import Mapping
from typing import Final

from lxml import etree
from starlette.responses import Response

XML_MEDIA_TYPE: Final = "application/xml; charset=utf-8"


class XmlResponse(Response):
    """Response whose body is an XML document in UTF-8."""

    media_type = XML_MEDIA_TYPE


def xml_response(
    document: etree._Element, status_code: int = 200, headers: Mapping[str, str] | None = None
) -> XmlResponse:
    """Serializes a document with an XML declaration.

    Args:
        document: Root element of the response.
        status_code: HTTP status.
        headers: Extra response headers.

    Returns:
        The response.
    """
    body = etree.tostring(document, xml_declaration=True, encoding="UTF-8")
    return XmlResponse(body, status_code=status_code, headers=headers)
