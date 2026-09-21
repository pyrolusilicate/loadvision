"""TrainingRunner for tests: returns a prepared outcome and records what it was asked to do."""

from collections.abc import Callable
from pathlib import Path

from loadvision.application.ports import TrainingOutcome, TrainingRequest, TrainingStatus

RUN_ID = "run_20260918_101500_ab12"


class FakeTrainingRunner:
    """Returns the outcome a test set, or raises the error it was given.

    Args:
        outcome: What the run reports; by default a package that passed the gate.
        error: Raised instead of returning, to test a runner that crashes.

    Attributes:
        requests: Every request the runner was given.
        during: Called while the run is in progress, before it returns.
    """

    def __init__(
        self, outcome: TrainingOutcome | None = None, error: Exception | None = None
    ) -> None:
        self._outcome = outcome or succeeded()
        self._error = error
        self.requests: list[TrainingRequest] = []
        self.during: Callable[[], None] | None = None

    def run(self, request: TrainingRequest) -> TrainingOutcome:
        self.requests.append(request)
        if self.during is not None:
            # Lets a test look at the database while the job is running.
            self.during()
        if self._error is not None:
            raise self._error
        return self._outcome


# The one line a real gate writes about its decision, numbers included.
GATE_REASON = "кандидат не хуже активной на 716 общих фото: -0.021 при допуске +0.30"


def succeeded(
    *,
    package_path: Path = Path("runs/run-1/package"),
    gate_passed: bool = True,
    reason: str = GATE_REASON,
) -> TrainingOutcome:
    """Outcome of a run that produced a package."""
    return TrainingOutcome(
        run_id=RUN_ID,
        status=TrainingStatus.SUCCEEDED,
        package_path=package_path,
        gate_passed=gate_passed,
        gate_report={"mae_candidate": 7.0, "mae_active": 7.5, "reason": reason},
    )
