import math
from dataclasses import fields
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest

from loadvision.domain.assessment import Assessment, LoadPercent, ReviewReason, Signals
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.shipment import ShipmentId

SIGNAL_NAMES = [signal.name for signal in fields(Signals)]


def make_assessment(**overrides: Any) -> Assessment:
    values: dict[str, Any] = {
        "id": UUID("3f6c2a4e-7b1d-4c55-9a0e-5d2f8e41c7aa"),
        "shipment_id": ShipmentId("RP-2026-000123"),
        "image_sha256": "a" * 64,
        "load_pct": LoadPercent(85.0),
        "model_version": ModelVersionId("2026-09-20.1"),
        "processing_ms": 184,
        "created_at": datetime(2026, 9, 20, 10, 15, 3, tzinfo=UTC),
    }
    values.update(overrides)
    return Assessment(**values)


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, 0.0, 57.25, 100, 100.0])
def test_load_percent_keeps_raw_value(value: float) -> None:
    assert LoadPercent(value).value == value


@pytest.mark.unit
@pytest.mark.parametrize("value", [-0.001, 100.001, math.nan, math.inf, -math.inf])
def test_load_percent_rejects_value_outside_range(value: float) -> None:
    with pytest.raises(ValueError, match="load percent"):
        LoadPercent(value)


@pytest.mark.unit
def test_signal_names_match_the_xml_contract() -> None:
    assert SIGNAL_NAMES == ["uncertainty", "ood_score", "brightness", "sharpness"]


@pytest.mark.unit
def test_signals_are_absent_by_default() -> None:
    assert all(getattr(Signals(), name) is None for name in SIGNAL_NAMES)


@pytest.mark.unit
@pytest.mark.parametrize("name", SIGNAL_NAMES)
@pytest.mark.parametrize("value", [0.0, 0.31, 7.4])
def test_signal_accepts_non_negative_value(name: str, value: float) -> None:
    assert getattr(Signals(**{name: value}), name) == value


@pytest.mark.unit
@pytest.mark.parametrize("name", SIGNAL_NAMES)
@pytest.mark.parametrize("value", [-0.01, math.nan, math.inf])
def test_signal_rejects_negative_or_non_finite_value(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        Signals(**{name: value})


@pytest.mark.unit
def test_review_reason_values_match_the_xml_contract() -> None:
    assert [reason.value for reason in ReviewReason] == [
        "low_confidence",
        "unusual_scene",
        "too_dark",
        "blurry",
    ]


@pytest.mark.unit
def test_assessment_defaults_need_no_review() -> None:
    assessment = make_assessment()
    assert assessment.signals == Signals()
    assert assessment.review_reasons == frozenset()
    assert assessment.request_id is None
    assert assessment.needs_review is False


@pytest.mark.unit
def test_assessment_with_a_reason_needs_review() -> None:
    assessment = make_assessment(review_reasons=frozenset({ReviewReason.TOO_DARK}))
    assert assessment.needs_review is True


@pytest.mark.unit
@pytest.mark.parametrize("image_sha256", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64])
def test_assessment_rejects_malformed_image_hash(image_sha256: str) -> None:
    with pytest.raises(ValueError, match="image_sha256"):
        make_assessment(image_sha256=image_sha256)


@pytest.mark.unit
def test_assessment_accepts_zero_processing_time() -> None:
    assert make_assessment(processing_ms=0).processing_ms == 0


@pytest.mark.unit
def test_assessment_rejects_negative_processing_time() -> None:
    with pytest.raises(ValueError, match="processing_ms"):
        make_assessment(processing_ms=-1)


@pytest.mark.unit
def test_assessment_rejects_naive_time() -> None:
    with pytest.raises(ValueError, match="created_at"):
        make_assessment(created_at=datetime(2026, 9, 20, 10, 15, 3))
