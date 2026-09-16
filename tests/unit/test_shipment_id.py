import pytest

from loadvision.domain.shipment import ShipmentId
from loadvision.errors import ErrorCode, LoadvisionError


@pytest.mark.unit
@pytest.mark.parametrize("value", ["RP-2026-000123", "a", "abc_DEF-09", "A" * 64])
def test_accepts_valid_id(value: str) -> None:
    assert ShipmentId(value).value == value


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "",
        "A" * 65,
        "рп-123",
        # The id goes into a URL path, where these characters change the route.
        "RP/123",
        "RP-1?x",
        "RP 123",
        # re.match with "^...$" accepts a trailing newline; fullmatch must not.
        "RP-1\n",
    ],
)
def test_rejects_invalid_id(value: str) -> None:
    with pytest.raises(LoadvisionError) as error:
        ShipmentId(value)
    assert error.value.code is ErrorCode.INVALID_SHIPMENT_ID


@pytest.mark.unit
def test_equal_ids_work_as_the_same_dict_key() -> None:
    assert {ShipmentId("RP-1"): 1}[ShipmentId("RP-1")] == 1
