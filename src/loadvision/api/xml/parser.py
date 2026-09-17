"""Hardened XML parsing and XSD validation."""

import importlib.resources
import re
import threading
from functools import cache
from typing import Final

from lxml import etree

from loadvision.errors import ErrorCode, LoadvisionError

NAMESPACE: Final = "urn:loadvision:api:v1"
XML_CONTENT_TYPE: Final = "application/xml"

_XSD_RESOURCE: Final = "xsd/loadvision-v1.xsd"
# Schema errors on these elements have a code more precise than INVALID_XML.
_ELEMENT_CODES: Final = {
    "ShipmentId": ErrorCode.INVALID_SHIPMENT_ID,
    "Image": ErrorCode.INVALID_ENCODING,
}
# libxml2 starts a schema error with the element it is about.
_ERROR_ELEMENT_RE: Final = re.compile(r"Element '(?:\{[^}]*\})?([A-Za-z]+)'")

# A compiled schema must not validate two documents at once, so every thread compiles its own.
_thread_schema = threading.local()


@cache
def schema_bytes() -> bytes:
    """Returns the XSD served at `GET /api/v1/schema.xsd` and used for validation."""
    return importlib.resources.files("loadvision.api").joinpath(_XSD_RESOURCE).read_bytes()


def check_content_type(content_type: str | None) -> None:
    """Accepts only `application/xml`, with no charset or UTF-8.

    Args:
        content_type: Value of the Content-Type header.

    Raises:
        LoadvisionError: `INVALID_CONTENT_TYPE` otherwise.
    """
    media_type, *params = (part.strip() for part in (content_type or "").split(";"))
    charsets = [
        value.strip().strip('"').lower()
        for name, _, value in (param.partition("=") for param in params)
        if name.strip().lower() == "charset"
    ]
    if media_type.lower() != XML_CONTENT_TYPE or any(c not in ("utf-8", "utf8") for c in charsets):
        raise LoadvisionError(ErrorCode.INVALID_CONTENT_TYPE, f"content type {content_type!r}")


def parse_document(body: bytes, root_name: str) -> etree._Element:
    """Parses a request body without entities, DTD or network, and validates it against the XSD.

    Args:
        body: Request body.
        root_name: Expected root element, such as ``AssessmentRequest``.

    Returns:
        The root element of a valid document.

    Raises:
        LoadvisionError: `INVALID_XML` if the body is not well-formed, has a DOCTYPE, has
            another root or breaks the schema; `INVALID_SHIPMENT_ID` or `INVALID_ENCODING`
            if every schema error is about the shipment id or about the photo.
    """
    try:
        root = etree.fromstring(body, _parser())
    except etree.XMLSyntaxError as error:
        raise LoadvisionError(ErrorCode.INVALID_XML, f"not well-formed: {error}") from error
    if root.getroottree().docinfo.doctype:
        raise LoadvisionError(ErrorCode.INVALID_XML, "DOCTYPE is not accepted")
    name = etree.QName(root)
    if (name.namespace, name.localname) != (NAMESPACE, root_name):
        raise LoadvisionError(ErrorCode.INVALID_XML, f"expected {root_name}, got {name.text}")
    schema = _schema()
    if not schema.validate(root):
        raise _schema_error(schema.error_log)
    return root


def _parser() -> etree.XMLParser:
    # A parser per document: lxml parsers must not be shared between threads.
    return etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)


def _schema() -> etree.XMLSchema:
    schema: etree.XMLSchema | None = getattr(_thread_schema, "schema", None)
    if schema is None:
        schema = etree.XMLSchema(etree.fromstring(schema_bytes(), _parser()))
        _thread_schema.schema = schema
    return schema


def _schema_error(log: etree._ListErrorLog) -> LoadvisionError:
    # Only element names go into the detail: a message can quote the value, and the value of
    # Image is the whole photo in base64.
    elements = sorted(
        {match.group(1) for entry in log if (match := _ERROR_ELEMENT_RE.match(entry.message))}
    )
    codes = {_ELEMENT_CODES.get(element, ErrorCode.INVALID_XML) for element in elements}
    code = codes.pop() if len(codes) == 1 else ErrorCode.INVALID_XML
    return LoadvisionError(code, f"schema errors in: {', '.join(elements) or 'document'}")
