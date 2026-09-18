"""CLI commands for the retraining queue."""

import argparse
import logging
import uuid

from loadvision.application.training_jobs import DEFAULT_LIMIT, StartTraining
from loadvision.cli.database import unit_of_work_factory
from loadvision.cli.values import short_utc
from loadvision.domain.timestamps import utc_now
from loadvision.domain.training_job import TrainingJob, TrainingMode

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `jobs` group and its commands.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser("jobs", help="retraining queue")
    commands = parser.add_subparsers(title="commands", required=True)

    start = commands.add_parser(
        "start",
        help="queue a retraining on the confirmed labels",
        description=(
            "Puts a job into the queue; the worker picks it up. Only one retraining is queued "
            "or running at a time."
        ),
    )
    start.add_argument(
        "--mode",
        choices=[mode.value for mode in TrainingMode],
        default=TrainingMode.FAST.value,
        help="how much of the pipeline to repeat (default: %(default)s)",
    )
    start.set_defaults(run=start_job)

    listing = commands.add_parser(
        "list", help="recent jobs, newest first", description="What the queue has been doing."
    )
    listing.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="rows to show (default: %(default)s)"
    )
    listing.set_defaults(run=list_jobs)


def start_job(args: argparse.Namespace) -> int:
    """Queues a retraining.

    Returns:
        Exit code 0.

    Raises:
        LoadvisionError: `TRAINING_ALREADY_RUNNING` if a job is queued or running.
        RuntimeError: If `DATABASE_URL` is not set.
    """
    with unit_of_work_factory() as uow_factory:
        job = StartTraining(uow_factory=uow_factory, clock=utc_now, new_id=uuid.uuid4)(
            TrainingMode(args.mode)
        )
    _logger.info("queued job %s in mode %s", job.id, job.mode.value)
    _logger.info("the worker picks it up; watch it with: loadvision jobs list")
    return 0


def list_jobs(args: argparse.Namespace) -> int:
    """Prints the recent jobs, newest first.

    Returns:
        Exit code 0.

    Raises:
        RuntimeError: If `DATABASE_URL` is not set.
    """
    with unit_of_work_factory() as uow_factory, uow_factory() as uow:
        jobs = uow.training_jobs.list_recent(args.limit)
    if not jobs:
        _logger.info("the queue is empty")
        return 0
    _logger.info("%-38s%-12s%-22s%-24s%s", "job", "status", "queued", "result", "gate")
    for job in jobs:
        _logger.info(
            "%-38s%-12s%-22s%-24s%s",
            job.id,
            job.status.value,
            short_utc(job.created_at),
            _result(job),
            _gate(job),
        )
    return 0


def _result(job: TrainingJob) -> str:
    if job.result_version is not None:
        return job.result_version.value
    return job.error_code.value if job.error_code is not None else "-"


def _gate(job: TrainingJob) -> str:
    if job.gate_passed is None:
        return "-"
    return "passed" if job.gate_passed else "rejected"
