"""XML messages of API v1: reading requests and building responses as the XSD describes them."""

import base64
import binascii
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final
from uuid import UUID

from lxml import etree

from loadvision.api.xml.parser import NAMESPACE
from loadvision.application.get_shipment import ShipmentHistory
from loadvision.application.labelled_assessments import LabeledAssessment
from loadvision.application.ports import ImageRecord
from loadvision.domain.assessment import Assessment, LoadPercent
from loadvision.domain.confirmed_label import ConfirmedLabel
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


@dataclass(frozen=True, slots=True)
class LabelRequestMessage:
    """Fields of `LabelRequest`.

    Attributes:
        load_pct: Confirmed load, or None if the request says `Unusable`.
        author: Author as sent, or None if absent.
    """

    load_pct: LoadPercent | None
    author: str | None


def read_label_request(root: etree._Element) -> LabelRequestMessage:
    """Reads a request that already passed the XSD, so it has either `LoadPct` or `Unusable`.

    Args:
        root: Root element `LabelRequest`.

    Returns:
        The request fields.
    """
    text = root.findtext(qualified("LoadPct"))
    # The XSD accepts "-0" as zero; adding 0.0 turns -0.0 into 0.0, which is written as "0.0".
    load_pct = None if text is None else LoadPercent(float(text) + 0.0)
    return LabelRequestMessage(load_pct=load_pct, author=root.findtext(qualified("Author")))


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


def assessment_document(
    assessment: Assessment,
    image: ImageRecord,
    *,
    label: ConfirmedLabel | None = None,
    reused: bool | None = None,
) -> etree._Element:
    """Builds `Assessment`.

    Args:
        assessment: The assessment.
        image: Stored metadata of the assessed photo.
        label: Label of the photo, if any.
        reused: For `POST`, whether the stored assessment was returned; None elsewhere, where
            the attribute is not written.
    """
    attributes = {"id": str(assessment.id), "shipmentId": assessment.shipment_id.value}
    if reused is not None:
        attributes["reused"] = _boolean(reused)
    root = etree.Element(qualified("Assessment"), attributes, nsmap={None: NAMESPACE})
    etree.SubElement(root, qualified("LoadPct")).text = format_decimal(
        assessment.load_pct.value, LOAD_PCT_PLACES
    )
    # Review is written only when there are reasons: a package without an `ood` section checks
    # nothing, so an empty set means "not checked" and needed="false" would claim more than the
    # service knows.
    if assessment.review_reasons:
        review = etree.SubElement(root, qualified("Review"), needed="true")
        for reason in sorted(assessment.review_reasons):
            etree.SubElement(review, qualified("Reason")).text = reason.value
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
        etree.SubElement(root, qualified("Signals"), known)
    if label is not None:
        etree.SubElement(root, qualified("ConfirmedLabel"), _label_attributes(label))
    etree.SubElement(
        root,
        qualified("Image"),
        sha256=image.sha256,
        mediaType=image.media_type.value,
        width=str(image.width),
        height=str(image.height),
    )
    etree.SubElement(root, qualified("Model"), version=assessment.model_version.value)
    etree.SubElement(root, qualified("ProcessingMs")).text = str(assessment.processing_ms)
    etree.SubElement(root, qualified("CreatedAt")).text = format_utc(assessment.created_at)
    return root


def shipment_document(history: ShipmentHistory) -> etree._Element:
    """Builds `Shipment` with the current assessment and the history, newest first."""
    root = etree.Element(
        qualified("Shipment"), nsmap={None: NAMESPACE}, id=history.shipment_id.value
    )
    etree.SubElement(root, qualified("Current"), _summary(history.current))
    past = etree.SubElement(root, qualified("History"), count=str(len(history.assessments)))
    for item in history.assessments:
        etree.SubElement(past, qualified("Assessment"), _summary(item))
    return root


def label_document(label: ConfirmedLabel, assessment_id: UUID) -> etree._Element:
    """Builds `Label` for the label stored through an assessment."""
    root = etree.Element(
        qualified("Label"),
        nsmap={None: NAMESPACE},
        assessmentId=str(assessment_id),
        imageSha256=label.image_sha256,
    )
    if label.load_pct is None:
        etree.SubElement(root, qualified("Unusable"))
    else:
        etree.SubElement(root, qualified("LoadPct")).text = format_decimal(
            label.load_pct.value, LOAD_PCT_PLACES
        )
    if label.author is not None:
        etree.SubElement(root, qualified("Author")).text = label.author
    etree.SubElement(root, qualified("UpdatedAt")).text = format_utc(label.updated_at)
    return root


def health_document(version: ModelVersionId) -> etree._Element:
    """Builds `Health` for a service with a loaded model."""
    return etree.Element(qualified("Health"), nsmap={None: NAMESPACE}, modelVersion=version.value)


def error_document(code: ErrorCode, request_id: str) -> etree._Element:
    """Builds `Error` with the message and hint of the catalog."""
    spec = ERROR_CATALOG[code]
    root = etree.Element(
        qualified("Error"), nsmap={None: NAMESPACE}, code=code.value, requestId=request_id
    )
    etree.SubElement(root, qualified("Message")).text = spec.message
    etree.SubElement(root, qualified("Hint")).text = spec.hint
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


def _summary(item: LabeledAssessment) -> dict[str, str]:
    assessment = item.assessment
    attributes = {
        "id": str(assessment.id),
        "loadPct": format_decimal(assessment.load_pct.value, LOAD_PCT_PLACES),
        "modelVersion": assessment.model_version.value,
        "createdAt": format_utc(assessment.created_at),
    }
    # The same rule as for `Review`: without a review policy, no reasons means "not checked".
    if assessment.review_reasons:
        attributes["needsReview"] = "true"
    if item.label is not None and item.label.load_pct is not None:
        attributes["confirmedLoadPct"] = format_decimal(item.label.load_pct.value, LOAD_PCT_PLACES)
    if item.label is not None and item.label.unusable:
        attributes["photoUnusable"] = "true"
    return attributes


def _label_attributes(label: ConfirmedLabel) -> dict[str, str]:
    attributes = {"updatedAt": format_utc(label.updated_at)}
    if label.load_pct is not None:
        attributes["loadPct"] = format_decimal(label.load_pct.value, LOAD_PCT_PLACES)
    else:
        attributes["unusable"] = "true"
    if label.author is not None:
        attributes["author"] = label.author
    return attributes


def qualified(local: str) -> str:
    """Returns the element name in the API namespace."""
    return f"{{{NAMESPACE}}}{local}"


def _text(root: etree._Element, local: str) -> str:
    return root.findtext(qualified(local)) or ""


def _boolean(value: bool) -> str:
    return "true" if value else "false"
