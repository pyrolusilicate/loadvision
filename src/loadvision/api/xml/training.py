"""XML messages about model versions and the retraining queue."""

from collections.abc import Sequence

from lxml import etree

from loadvision.api.xml.parser import NAMESPACE
from loadvision.api.xml.schemas import qualified
from loadvision.domain.model_version import ModelVersion
from loadvision.domain.timestamps import format_utc
from loadvision.domain.training_job import TrainingJob, TrainingMode


def read_training_job_request(root: etree._Element) -> TrainingMode:
    """Reads a request that already passed the XSD.

    Args:
        root: Root element `TrainingJobRequest`.

    Returns:
        The mode to retrain in; `fast` when the request does not name one.
    """
    mode = root.findtext(qualified("Mode"))
    return TrainingMode.FAST if mode is None else TrainingMode(mode)


def models_document(versions: Sequence[ModelVersion]) -> etree._Element:
    """Builds `Models` with every registered version."""
    root = etree.Element(qualified("Models"), nsmap={None: NAMESPACE})
    for version in versions:
        etree.SubElement(root, qualified("Model"), _model(version))
    return root


def model_document(version: ModelVersion) -> etree._Element:
    """Builds `Model` for one version."""
    return etree.Element(qualified("Model"), _model(version), nsmap={None: NAMESPACE})


def training_job_document(job: TrainingJob) -> etree._Element:
    """Builds `TrainingJob` for one job."""
    root = etree.Element(qualified("TrainingJob"), _job(job), nsmap={None: NAMESPACE})
    _add_result(root, job)
    return root


def training_jobs_document(jobs: Sequence[TrainingJob]) -> etree._Element:
    """Builds `TrainingJobs`, newest first."""
    root = etree.Element(qualified("TrainingJobs"), nsmap={None: NAMESPACE})
    for job in jobs:
        _add_result(etree.SubElement(root, qualified("TrainingJob"), _job(job)), job)
    return root


def _model(version: ModelVersion) -> dict[str, str]:
    attributes = {
        "version": version.version.value,
        "status": version.status.value,
        "createdAt": format_utc(version.created_at),
    }
    if version.activated_at is not None:
        attributes["activatedAt"] = format_utc(version.activated_at)
    return attributes


def _job(job: TrainingJob) -> dict[str, str]:
    attributes = {
        "id": str(job.id),
        "status": job.status.value,
        "mode": job.mode.value,
        "createdAt": format_utc(job.created_at),
    }
    if job.started_at is not None:
        attributes["startedAt"] = format_utc(job.started_at)
    if job.finished_at is not None:
        attributes["finishedAt"] = format_utc(job.finished_at)
    return attributes


def _add_result(element: etree._Element, job: TrainingJob) -> None:
    # A job that has not ended has nothing to report yet.
    if not job.done:
        return
    result = {}
    if job.run_id is not None:
        result["runId"] = job.run_id
    if job.base_version is not None:
        result["baseVersion"] = job.base_version.value
    if job.result_version is not None:
        result["version"] = job.result_version.value
    if job.gate_passed is not None:
        result["gatePassed"] = "true" if job.gate_passed else "false"
    if job.error_code is not None:
        result["errorCode"] = job.error_code.value
    etree.SubElement(element, qualified("Result"), result)
