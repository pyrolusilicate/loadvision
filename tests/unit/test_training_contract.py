from pathlib import Path

import pytest

from loadvision.application.ports import (
    LabeledImage,
    TrainingMode,
    TrainingOutcome,
    TrainingRequest,
    TrainingStatus,
)
from loadvision.domain.assessment import LoadPercent
from loadvision.domain.model_version import ModelVersionId
from loadvision.errors import ErrorCode

SHA_A = "a" * 64
SHA_B = "b" * 64
PACKAGE = Path("storage/models/2026-09-20.1")


def make_label(sha256: str = SHA_A, load_pct: float = 85.0) -> LabeledImage:
    return LabeledImage(image_sha256=sha256, load_pct=LoadPercent(load_pct))


@pytest.mark.unit
def test_labeled_image_rejects_a_hash_that_is_not_lowercase_hex() -> None:
    with pytest.raises(ValueError, match="image_sha256"):
        make_label("A" * 64)


@pytest.mark.unit
def test_request_defaults_to_no_labels_and_no_base_version() -> None:
    request = TrainingRequest(mode=TrainingMode.FAST)
    assert (request.labels, request.base_version) == ((), None)


@pytest.mark.unit
def test_request_keeps_labels_of_different_photos() -> None:
    request = TrainingRequest(
        mode=TrainingMode.FULL,
        labels=(make_label(SHA_A), make_label(SHA_B, 10.0)),
        base_version=ModelVersionId("2026-09-20.1"),
    )
    assert [label.image_sha256 for label in request.labels] == [SHA_A, SHA_B]


@pytest.mark.unit
def test_request_rejects_two_labels_of_one_photo() -> None:
    with pytest.raises(ValueError, match="repeat a photo"):
        TrainingRequest(mode=TrainingMode.FAST, labels=(make_label(SHA_A), make_label(SHA_A, 10.0)))


@pytest.mark.unit
def test_rejected_candidate_is_a_successful_run() -> None:
    outcome = TrainingOutcome(
        run_id="run_20260918_140155_a3f9c2",
        status=TrainingStatus.SUCCEEDED,
        package_path=PACKAGE,
        gate_passed=False,
        gate_report={"mae_candidate": 12.4, "mae_active": 11.9},
    )
    assert outcome.gate_passed is False
    assert outcome.error_code is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": TrainingStatus.SUCCEEDED}, "succeeded run"),
        (
            {
                "status": TrainingStatus.SUCCEEDED,
                "package_path": PACKAGE,
                "error_code": ErrorCode.STAGE_CHECK_FAILED,
            },
            "succeeded run",
        ),
        ({"status": TrainingStatus.FAILED}, "failed run"),
        (
            {"status": TrainingStatus.FAILED, "package_path": PACKAGE},
            "failed run",
        ),
        (
            {
                "status": TrainingStatus.FAILED,
                "error_code": ErrorCode.STAGE_CHECK_FAILED,
                "gate_passed": True,
            },
            "cannot pass the gate",
        ),
    ],
)
def test_outcome_rejects_fields_that_disagree_with_the_status(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingOutcome(run_id="run_20260918_140155_a3f9c2", **kwargs)  # type: ignore[arg-type]
