"""Catalog of event names written to logs and run records."""

from enum import StrEnum


class Event(StrEnum):
    """Stable event names; searching logs by an exact name finds every occurrence."""

    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    STAGE_STARTED = "stage.started"
    STAGE_FINISHED = "stage.finished"
    STAGE_FAILED = "stage.failed"
    REQUEST_FINISHED = "request.finished"
    MODEL_ACTIVATED = "model.activated"
    GATE_REJECTED = "gate.rejected"
