"""Route that serves the XSD of the API."""

from fastapi import APIRouter

from loadvision.api.xml.parser import schema_bytes
from loadvision.api.xml.response import XmlResponse

router = APIRouter()


@router.get("/api/v1/schema.xsd")
def schema() -> XmlResponse:
    """Returns the XML schema requests are validated against."""
    return XmlResponse(schema_bytes())
