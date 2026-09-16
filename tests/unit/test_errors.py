import pickle

import pytest

from loadvision.errors import ERROR_CATALOG, ErrorCode, LoadvisionError


@pytest.mark.unit
def test_every_code_has_a_catalog_entry() -> None:
    assert set(ERROR_CATALOG) == set(ErrorCode)


@pytest.mark.unit
@pytest.mark.parametrize("code", list(ErrorCode))
def test_code_value_is_its_name(code: ErrorCode) -> None:
    # Clients and the XSD see the value; StrEnum with auto() would lowercase it.
    assert code.value == code.name


@pytest.mark.unit
@pytest.mark.parametrize("code", list(ErrorCode))
def test_catalog_entry_is_a_complete_error_response(code: ErrorCode) -> None:
    spec = ERROR_CATALOG[code]
    assert spec.http_status is None or 400 <= spec.http_status <= 599
    assert spec.message.strip()
    assert spec.hint.strip()


@pytest.mark.unit
def test_catalog_cannot_be_changed_at_runtime() -> None:
    with pytest.raises(TypeError):
        ERROR_CATALOG[ErrorCode.INTERNAL_ERROR] = ERROR_CATALOG[ErrorCode.MODEL_NOT_READY]  # pyright: ignore[reportIndexIssue]


@pytest.mark.unit
def test_error_exposes_code_spec_and_detail() -> None:
    error = LoadvisionError(ErrorCode.CORRUPTED_IMAGE, "truncated JPEG")
    assert error.code is ErrorCode.CORRUPTED_IMAGE
    assert error.spec is ERROR_CATALOG[ErrorCode.CORRUPTED_IMAGE]
    assert str(error) == "CORRUPTED_IMAGE: truncated JPEG"


@pytest.mark.unit
def test_error_without_detail_prints_only_the_code() -> None:
    assert str(LoadvisionError(ErrorCode.INTERNAL_ERROR)) == "INTERNAL_ERROR"


@pytest.mark.unit
@pytest.mark.parametrize("detail", [None, "truncated JPEG"])
def test_error_survives_pickling_between_processes(detail: str | None) -> None:
    restored = pickle.loads(pickle.dumps(LoadvisionError(ErrorCode.CORRUPTED_IMAGE, detail)))
    assert restored.code is ErrorCode.CORRUPTED_IMAGE
    assert restored.detail == detail
    assert str(restored) == str(LoadvisionError(ErrorCode.CORRUPTED_IMAGE, detail))
