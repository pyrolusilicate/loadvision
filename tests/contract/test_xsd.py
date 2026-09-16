import importlib.resources
from pathlib import Path

import pytest
from lxml import etree

from loadvision.domain.assessment import ReviewReason
from loadvision.domain.model_version import MODEL_VERSION_PATTERN, ModelVersionStatus
from loadvision.domain.shipment import SHIPMENT_ID_PATTERN
from loadvision.errors import ERROR_CATALOG, ErrorCode
from loadvision.imaging.prepared_image import MediaType

XS = "http://www.w3.org/2001/XMLSchema"
API = "urn:loadvision:api:v1"
FIXTURES = Path(__file__).parents[1] / "fixtures" / "xml"
EXAMPLES = sorted(FIXTURES.glob("*.xml"))
# The fixtures are trusted, but they are read the way the API reads requests: no entities,
# no network, no DTD.
PARSER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)


@pytest.fixture(scope="module")
def schema_root() -> etree._Element:
    resource = importlib.resources.files("loadvision.api").joinpath("xsd/loadvision-v1.xsd")
    return etree.fromstring(resource.read_bytes(), PARSER)


@pytest.fixture(scope="module")
def schema(schema_root: etree._Element) -> etree.XMLSchema:
    return etree.XMLSchema(schema_root)


def parse_example(name: str) -> etree._Element:
    return etree.parse(str(FIXTURES / name), PARSER).getroot()


def with_value(example: str, path: str, value: str) -> etree._Element:
    """Returns an example with one text or attribute replaced; `path` is like ``Image/@sha256``."""
    root = parse_example(example)
    element_path, _, attribute = path.partition("@")
    steps = [f"{{{API}}}{step}" for step in element_path.strip("/").split("/") if step]
    target = root.find("/".join(steps)) if steps else root
    assert target is not None, path
    if attribute:
        target.set(attribute, value)
    else:
        target.text = value
    return target.getroottree().getroot()


def enumeration(schema_root: etree._Element, type_name: str) -> list[str]:
    return sorted(facets(schema_root, type_name, "enumeration"))


def facets(schema_root: etree._Element, type_name: str, facet: str) -> list[str]:
    values = schema_root.xpath(
        f"xs:simpleType[@name=$name]/xs:restriction/xs:{facet}/@value",
        namespaces={"xs": XS},
        name=type_name,
    )
    return [str(value) for value in values]


@pytest.mark.contract
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_example_is_valid(schema: etree.XMLSchema, example: Path) -> None:
    schema.assertValid(etree.parse(str(example), PARSER))


@pytest.mark.contract
def test_every_message_has_an_example(schema_root: etree._Element) -> None:
    messages = {element.get("name") for element in schema_root.iterchildren(f"{{{XS}}}element")}
    examples = {etree.QName(parse_example(path.name)).localname for path in EXAMPLES}
    assert messages == examples


@pytest.mark.contract
@pytest.mark.parametrize(
    ("type_name", "pattern"),
    [("ShipmentIdType", SHIPMENT_ID_PATTERN), ("ModelVersionType", MODEL_VERSION_PATTERN)],
)
def test_pattern_is_the_one_in_code(
    schema_root: etree._Element, type_name: str, pattern: str
) -> None:
    assert facets(schema_root, type_name, "pattern") == [pattern]


@pytest.mark.contract
@pytest.mark.parametrize(
    ("type_name", "values"),
    [
        (
            "ErrorCodeType",
            [code.value for code in ErrorCode if ERROR_CATALOG[code].http_status is not None],
        ),
        ("ReviewReasonType", [reason.value for reason in ReviewReason]),
        ("ModelStatusType", [status.value for status in ModelVersionStatus]),
        ("MediaTypeType", [media_type.value for media_type in MediaType]),
    ],
)
def test_enumeration_is_the_one_in_code(
    schema_root: etree._Element, type_name: str, values: list[str]
) -> None:
    assert enumeration(schema_root, type_name) == sorted(values)


@pytest.mark.contract
def test_error_example_uses_catalog_texts() -> None:
    root = parse_example("error.xml")
    spec = ERROR_CATALOG[ErrorCode(root.get("code"))]
    assert root.findtext(f"{{{API}}}Message") == spec.message
    assert root.findtext(f"{{{API}}}Hint") == spec.hint


@pytest.mark.contract
@pytest.mark.parametrize(
    ("example", "path", "value"),
    [
        ("assessment_request.xml", "ShipmentId", "RP 1"),
        ("assessment_request.xml", "ShipmentId", ""),
        ("assessment_request.xml", "ShipmentId", "R" * 65),
        ("assessment_request.xml", "ShipmentId", "RP-1\n"),
        ("assessment_request.xml", "Image", ""),
        # libxml2 checks base64 loosely (it lets a "data:image/jpeg;base64," prefix through),
        # so the API decodes strictly on its own; the schema only catches gross errors.
        ("assessment_request.xml", "Image", "/9j/4AAQSkZJRgABAQ="),
        ("assessment_request.xml", "DeclaredLoadPct", "100.5"),
        ("assessment_created.xml", "LoadPct", "NaN"),
        ("assessment_created.xml", "LoadPct", "INF"),
        ("assessment_created.xml", "LoadPct", "1e2"),
        ("assessment_created.xml", "LoadPct", "-0.1"),
        ("assessment_created.xml", "LoadPct", "85.25"),
        ("assessment_created.xml", "Declared/@deviation", "-100.5"),
        ("assessment_created.xml", "CreatedAt", "2026-09-20T10:15:03"),
        ("assessment_created.xml", "CreatedAt", "2026-09-20T13:15:03+03:00"),
        ("assessment_created.xml", "@id", "3F6C2A4E-7B1D-4C55-9A0E-5D2F8E41C7AA"),
        ("assessment_created.xml", "Image/@sha256", "abc"),
        ("assessment_created.xml", "Image/@mediaType", "image/heic"),
        ("assessment_created.xml", "Review/Reason", "looks_odd"),
        ("assessment_created.xml", "Signals/@oodScore", "-0.1"),
        ("assessment_created.xml", "Model/@version", ".hidden"),
        ("shipment.xml", "History/@count", "0"),
        ("label_request.xml", "Author", ""),
        ("models.xml", "Model/@status", "deleted"),
        ("error.xml", "@code", "STAGE_CHECK_FAILED"),
        ("error.xml", "@requestId", "req 1"),
    ],
)
def test_rejects_invalid_value(
    schema: etree.XMLSchema, example: str, path: str, value: str
) -> None:
    assert not schema.validate(with_value(example, path, value))


@pytest.mark.contract
@pytest.mark.parametrize(
    ("example", "path", "value"),
    [
        ("assessment_request.xml", "ShipmentId", "R" * 64),
        ("assessment_request.xml", "DeclaredLoadPct", "0"),
        ("assessment_created.xml", "LoadPct", "100.0"),
        ("assessment_created.xml", "Declared/@deviation", "-100"),
        ("assessment_created.xml", "CreatedAt", "2026-09-20T10:15:03.123456Z"),
        ("error.xml", "@requestId", "0b7c6f2e-1d7a-4c1e-9f55-3a2b1c0d9e8f"),
    ],
)
def test_accepts_boundary_value(
    schema: etree.XMLSchema, example: str, path: str, value: str
) -> None:
    schema.assertValid(with_value(example, path, value))


@pytest.mark.contract
def test_rejects_unknown_element(schema: etree.XMLSchema) -> None:
    root = parse_example("assessment_request.xml")
    etree.SubElement(root, f"{{{API}}}Comment").text = "fragile"
    assert not schema.validate(root)


@pytest.mark.contract
def test_rejects_document_without_namespace(schema: etree.XMLSchema) -> None:
    root = etree.fromstring(b'<Health modelVersion="2026-09-20.1"/>', PARSER)
    assert not schema.validate(root)
