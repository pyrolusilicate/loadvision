"""XML messages of API v1: reading requests and building responses as the XSD describes them."""

import base64
import binascii
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from lxml import etree

from loadvision.api.xml.parser import NAMESPACE
from loadvision.application.assess_photo import AssessPhotoResult
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.timestamps import format_utc
from loadvision.errors import ERROR_CATALOG, ErrorCode, LoadvisionError

LOAD_PCT_PLACES: Final = 1
SIGNAL_PLACES: Final = 3

# Java, openssl and mail encoders wrap base64 at 76 characters; the breaks carry no data.
_BASE64_WHITESPACE_RE: Final = re.compile(r"[ \t\r\n]+")


@dataclass(frozen=True, slots=True)
class AssessmentRequestMessage:
    """Fields of `AssessmentRequest`.

    Attributes:
        shipment_id: Shipment number as sent; the use case validates it.
        image: Photo file bytes decoded from base64.
    """

    shipment_id: str
    image: bytes


def read_assessment_request(root: etree._Element) -> AssessmentRequestMessage:
    """Reads a request that already passed the XSD.

    Args:
        root: Root element `AssessmentRequest`.

    Returns:
        The request fields.

    Raises:
        LoadvisionError: `INVALID_ENCODING` if the photo is not strict base64.
    """
    return AssessmentRequestMessage(
        shipment_id=_text(root, "ShipmentId"), image=decode_base64(_text(root, "Image"))
    )


def decode_base64(text: str) -> bytes:
    """Decodes base64 strictly, skipping only spaces and line breaks.

    The XSD check of base64 in libxml2 is loose and lets a ``data:image/jpeg;base64,`` prefix
    through, so the decoding here is what rejects foreign characters and bad padding.

    Raises:
        LoadvisionError: `INVALID_ENCODING` for any other character or bad padding.
    """
    try:
        return base64.b64decode(_BASE64_WHITESPACE_RE.sub("", text), validate=True)
    except (binascii.Error, ValueError) as error:
        raise LoadvisionError(ErrorCode.INVALID_ENCODING, "photo is not valid base64") from error


def assessment_document(result: AssessPhotoResult) -> etree._Element:
    """Builds `Assessment` for a new or reused assessment."""
    assessment = result.assessment
    root = etree.Element(
        _name("Assessment"),
        nsmap={None: NAMESPACE},
        id=str(assessment.id),
        shipmentId=assessment.shipment_id.value,
        reused=_boolean(result.reused),
    )
    etree.SubElement(root, _name("LoadPct")).text = format_decimal(
        assessment.load_pct.value, LOAD_PCT_PLACES
    )
    # Review is written only when there are reasons: until the review policy exists, an empty
    # set means "not checked", and needed="false" would claim more than the service knows.
    if assessment.review_reasons:
        review = etree.SubElement(root, _name("Review"), needed="true")
        for reason in sorted(assessment.review_reasons):
            etree.SubElement(review, _name("Reason")).text = reason.value
    signals = assessment.signals
    known = {
        name: format_decimal(value, SIGNAL_PLACES)
        for name, value in (
            ("uncertainty", signals.uncertainty),
            ("oodScore", signals.ood_score),
            ("brightness", signals.brightness),
            ("sharpness", signals.sharpness),
        )
        if value is not None
    }
    if known:
        etree.SubElement(root, _name("Signals"), known)
    etree.SubElement(
        root,
        _name("Image"),
        sha256=result.image.sha256,
        mediaType=result.image.media_type.value,
        width=str(result.image.width),
        height=str(result.image.height),
    )
    etree.SubElement(root, _name("Model"), version=assessment.model_version.value)
    etree.SubElement(root, _name("ProcessingMs")).text = str(assessment.processing_ms)
    etree.SubElement(root, _name("CreatedAt")).text = format_utc(assessment.created_at)
    return root


def health_document(version: ModelVersionId) -> etree._Element:
    """Builds `Health` for a service with a loaded model."""
    return etree.Element(_name("Health"), nsmap={None: NAMESPACE}, modelVersion=version.value)


def error_document(code: ErrorCode, request_id: str) -> etree._Element:
    """Builds `Error` with the message and hint of the catalog."""
    spec = ERROR_CATALOG[code]
    root = etree.Element(
        _name("Error"), nsmap={None: NAMESPACE}, code=code.value, requestId=request_id
    )
    etree.SubElement(root, _name("Message")).text = spec.message
    etree.SubElement(root, _name("Hint")).text = spec.hint
    return root


def format_decimal(value: float, places: int) -> str:
    """Writes a number as `xs:decimal` rounded half up to the given places.

    Python's `round` and `format` round halves to even on the binary value; here the shortest
    decimal form of the number is rounded, so 85.25 becomes 85.3.

    Args:
        value: Finite number.
        places: Digits after the point.

    Returns:
        Text without an exponent, such as ``85.3``.
    """
    quantum = Decimal(1).scaleb(-places)
    return format(Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP), "f")


def _name(local: str) -> str:
    return f"{{{NAMESPACE}}}{local}"


def _text(root: etree._Element, local: str) -> str:
    return root.findtext(_name(local)) or ""


def _boolean(value: bool) -> str:
    return "true" if value else "false"
