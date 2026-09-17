import base64

import pytest

from loadvision.api.xml.parser import check_content_type
from loadvision.api.xml.schemas import decode_base64, format_decimal
from loadvision.errors import ErrorCode, LoadvisionError


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "places", "text"),
    [
        (85.25, 1, "85.3"),
        (85.0, 1, "85.0"),
        (0.0, 1, "0.0"),
        (100.0, 1, "100.0"),
        (12.35, 1, "12.4"),
        (7.4499, 1, "7.4"),
        (1e-05, 3, "0.000"),
        (0.12345, 3, "0.123"),
        (2.5e-4, 3, "0.000"),
        (0.0005, 3, "0.001"),
    ],
)
def test_decimal_is_rounded_half_up_without_an_exponent(
    value: float, places: int, text: str
) -> None:
    assert format_decimal(value, places) == text


@pytest.mark.unit
def test_base64_skips_spaces_and_line_breaks() -> None:
    data = base64.b64encode(b"photo bytes " * 10).decode()
    wrapped = " \r\n".join(data[start : start + 20] for start in range(0, len(data), 20))
    assert decode_base64(wrapped) == b"photo bytes " * 10


@pytest.mark.unit
@pytest.mark.parametrize(
    "text", ["data:image/jpeg;base64,/9j/4AAQ", "/9j/4AAQ=", "@@@@", "/9j/4A-_"]
)
def test_base64_rejects_other_characters_and_bad_padding(text: str) -> None:
    with pytest.raises(LoadvisionError) as caught:
        decode_base64(text)
    assert caught.value.code is ErrorCode.INVALID_ENCODING


@pytest.mark.unit
@pytest.mark.parametrize(
    "content_type",
    [
        "application/xml",
        "APPLICATION/XML",
        "application/xml; charset=utf-8",
        'application/xml;charset="UTF8"',
    ],
)
def test_xml_content_type_is_accepted(content_type: str) -> None:
    check_content_type(content_type)


@pytest.mark.unit
@pytest.mark.parametrize(
    "content_type",
    [
        None,
        "",
        "text/xml",
        "application/xml+soap",
        "application/xml; charset=cp1251",
        "application/json",
    ],
)
def test_other_content_type_is_rejected(content_type: str | None) -> None:
    with pytest.raises(LoadvisionError) as caught:
        check_content_type(content_type)
    assert caught.value.code is ErrorCode.INVALID_CONTENT_TYPE
