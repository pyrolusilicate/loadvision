from datetime import UTC, datetime

import pytest

from loadvision.domain.model_version import ModelVersion, ModelVersionId, ModelVersionStatus
from loadvision.errors import ErrorCode, LoadvisionError

CANDIDATE = ModelVersionStatus.CANDIDATE
ACTIVE = ModelVersionStatus.ACTIVE
RETIRED = ModelVersionStatus.RETIRED
REJECTED = ModelVersionStatus.REJECTED

CREATED = datetime(2026, 9, 16, 10, 0, tzinfo=UTC)
ACTIVATED = datetime(2026, 9, 16, 11, 0, tzinfo=UTC)
NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)

# (status before, action, status after; None when the action is forbidden)
TRANSITIONS = [
    (CANDIDATE, "activate", ACTIVE),
    (CANDIDATE, "retire", None),
    (CANDIDATE, "reject", REJECTED),
    (ACTIVE, "activate", None),
    (ACTIVE, "retire", RETIRED),
    (ACTIVE, "reject", None),
    (RETIRED, "activate", ACTIVE),
    (RETIRED, "retire", None),
    (RETIRED, "reject", None),
    (REJECTED, "activate", None),
    (REJECTED, "retire", None),
    (REJECTED, "reject", None),
]


def make_version(status: ModelVersionStatus) -> ModelVersion:
    activated_at = ACTIVATED if status in (ACTIVE, RETIRED) else None
    return ModelVersion(
        version=ModelVersionId("2026-09-16.1"),
        status=status,
        created_at=CREATED,
        activated_at=activated_at,
    )


def apply(version: ModelVersion, action: str) -> ModelVersion:
    if action == "activate":
        return version.activate(NOW)
    if action == "retire":
        return version.retire()
    return version.reject()


@pytest.mark.unit
@pytest.mark.parametrize("value", ["2026-09-20.1", "v0", "baseline_ridge-v0", "a" * 64])
def test_version_id_accepts_valid_tag(value: str) -> None:
    assert ModelVersionId(value).value == value


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "",
        "a" * 65,
        # The tag is a URL path segment and a directory name, where dot names point elsewhere.
        ".",
        "..",
        ".hidden",
        "v1/../x",
        "v 1",
        "версия-1",
        "v1\n",
    ],
)
def test_version_id_rejects_unsafe_tag(value: str) -> None:
    with pytest.raises(ValueError, match="model version"):
        ModelVersionId(value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "activated_at"),
    [(ACTIVE, None), (RETIRED, None), (CANDIDATE, ACTIVATED), (REJECTED, ACTIVATED)],
)
def test_activation_time_must_match_status(
    status: ModelVersionStatus, activated_at: datetime | None
) -> None:
    with pytest.raises(ValueError, match="activated_at"):
        ModelVersion(
            version=ModelVersionId("v1"),
            status=status,
            created_at=CREATED,
            activated_at=activated_at,
        )


@pytest.mark.unit
def test_model_version_rejects_naive_time() -> None:
    with pytest.raises(ValueError, match="created_at"):
        ModelVersion(
            version=ModelVersionId("v1"), status=CANDIDATE, created_at=datetime(2026, 9, 16, 10)
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("before", "action", "after"), [row for row in TRANSITIONS if row[2] is not None]
)
def test_allowed_transition_changes_status(
    before: ModelVersionStatus, action: str, after: ModelVersionStatus
) -> None:
    assert apply(make_version(before), action).status is after


@pytest.mark.unit
@pytest.mark.parametrize(("before", "action"), [row[:2] for row in TRANSITIONS if row[2] is None])
def test_forbidden_transition_is_a_status_conflict(before: ModelVersionStatus, action: str) -> None:
    with pytest.raises(LoadvisionError) as error:
        apply(make_version(before), action)
    assert error.value.code is ErrorCode.MODEL_VERSION_STATUS_CONFLICT


@pytest.mark.unit
@pytest.mark.parametrize("before", [CANDIDATE, RETIRED])
def test_activation_records_the_new_time(before: ModelVersionStatus) -> None:
    assert make_version(before).activate(NOW).activated_at == NOW


@pytest.mark.unit
def test_retirement_keeps_the_activation_time() -> None:
    assert make_version(ACTIVE).retire().activated_at == ACTIVATED


@pytest.mark.unit
def test_activation_rejects_naive_time() -> None:
    with pytest.raises(ValueError, match="activated_at"):
        make_version(CANDIDATE).activate(datetime(2026, 9, 16, 12))
