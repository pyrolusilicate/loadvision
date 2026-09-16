"""Model version entity and its lifecycle statuses."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Self

from loadvision.domain.timestamps import ensure_aware
from loadvision.errors import ErrorCode, LoadvisionError

# Starts with a letter or digit: the version is a URL path segment and a directory name in the
# models storage, where "." or ".." would point to another location.
MODEL_VERSION_PATTERN: Final = "[A-Za-z0-9][A-Za-z0-9._-]{0,63}"

_MODEL_VERSION_RE: Final = re.compile(MODEL_VERSION_PATTERN)


@dataclass(frozen=True, slots=True)
class ModelVersionId:
    """Version tag of a model package, for example ``2026-09-20.1``.

    Attributes:
        value: 1 to 64 Latin letters, digits, dots, underscores or hyphens, starting with a
            letter or digit.
    """

    value: str

    def __post_init__(self) -> None:
        """Validates the format.

        Raises:
            ValueError: If `value` does not match `MODEL_VERSION_PATTERN`. The caller picks the
                error code: a bad tag in an API path is a lookup failure, a bad tag in a package
                manifest is a packaging failure.
        """
        if _MODEL_VERSION_RE.fullmatch(self.value) is None:
            raise ValueError(f"model version must match {MODEL_VERSION_PATTERN}")


class ModelVersionStatus(StrEnum):
    """Lifecycle status of a model version."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    REJECTED = "rejected"
    RETIRED = "retired"


# Retired -> active is a rollback to a version that has already served traffic.
_ALLOWED_TRANSITIONS: Final[Mapping[ModelVersionStatus, frozenset[ModelVersionStatus]]] = (
    MappingProxyType(
        {
            ModelVersionStatus.CANDIDATE: frozenset(
                {ModelVersionStatus.ACTIVE, ModelVersionStatus.REJECTED}
            ),
            ModelVersionStatus.ACTIVE: frozenset({ModelVersionStatus.RETIRED}),
            ModelVersionStatus.RETIRED: frozenset({ModelVersionStatus.ACTIVE}),
            ModelVersionStatus.REJECTED: frozenset[ModelVersionStatus](),
        }
    )
)

_EVER_ACTIVE_STATUSES: Final = frozenset({ModelVersionStatus.ACTIVE, ModelVersionStatus.RETIRED})


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelVersion:
    """Model package registered in the service.

    The domain checks the transitions of one version; the database keeps a single active
    version across all of them.

    Attributes:
        version: Version tag.
        status: Current lifecycle status.
        created_at: Registration time, timezone-aware.
        activated_at: Time of the latest activation, timezone-aware; None for a version that
            has never been active.
    """

    version: ModelVersionId
    status: ModelVersionStatus
    created_at: datetime
    activated_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validates the timestamps and their agreement with the status.

        Raises:
            ValueError: If a timestamp has no UTC offset, or `activated_at` is set for a version
                that has never been active, or missing for one that has.
        """
        ensure_aware(self.created_at, "created_at")
        if self.activated_at is not None:
            ensure_aware(self.activated_at, "activated_at")
        if (self.activated_at is not None) != (self.status in _EVER_ACTIVE_STATUSES):
            raise ValueError(f"activated_at does not match status {self.status}")

    def activate(self, at: datetime) -> Self:
        """Returns this version made active.

        Args:
            at: Activation time, timezone-aware.

        Raises:
            LoadvisionError: With code `MODEL_VERSION_STATUS_CONFLICT` if the current status
                does not allow the transition.
            ValueError: If `at` has no UTC offset.
        """
        return self._move_to(ModelVersionStatus.ACTIVE, activated_at=at)

    def retire(self) -> Self:
        """Returns this version taken out of service, keeping its activation time.

        Raises:
            LoadvisionError: With code `MODEL_VERSION_STATUS_CONFLICT` if the current status
                does not allow the transition.
        """
        return self._move_to(ModelVersionStatus.RETIRED, activated_at=self.activated_at)

    def reject(self) -> Self:
        """Returns this version rejected by the promotion gate.

        Raises:
            LoadvisionError: With code `MODEL_VERSION_STATUS_CONFLICT` if the current status
                does not allow the transition.
        """
        return self._move_to(ModelVersionStatus.REJECTED, activated_at=None)

    def _move_to(self, status: ModelVersionStatus, *, activated_at: datetime | None) -> Self:
        if status not in _ALLOWED_TRANSITIONS[self.status]:
            raise LoadvisionError(
                ErrorCode.MODEL_VERSION_STATUS_CONFLICT, f"{self.status} -> {status}"
            )
        return replace(self, status=status, activated_at=activated_at)
