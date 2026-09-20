"""What the operator pages need from the rest of the service, put together once at startup."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from starlette.requests import Request

from loadvision.application.assess_photo import AssessPhoto
from loadvision.application.get_assessment import GetAssessment
from loadvision.application.get_shipment import GetShipment
from loadvision.application.model_versions import ActivateModel, ListModelVersions, RollbackModel
from loadvision.application.operators import Authenticate, SignIn, SignOut
from loadvision.application.ports import ExplanationKind, ImageStorage
from loadvision.application.review_queue import ReviewQueue
from loadvision.application.submit_label import MarkReadable, SubmitLabel
from loadvision.application.training_jobs import (
    CountTrainingLabels,
    ListTrainingJobPage,
    StartTraining,
)
from loadvision.domain.model_version import ModelVersionId
from loadvision.ml.inference.active_model import ActiveModel

# Draws the map of what the model saw: photo hash, which map, and the version that answered.
type DrawHeatmap = Callable[[str, ExplanationKind, ModelVersionId], bytes]


@dataclass(frozen=True, slots=True, kw_only=True)
class UiServices:
    """Dependencies of the operator pages; tests build them from fakes.

    Attributes:
        sign_in: Checks a password and opens a session.
        authenticate: Finds the session of a browser.
        sign_out: Ends a session.
        assess_photo: Assesses an uploaded photo and stores it with its shipment.
        review_queue: Assessments waiting for a confirmed load.
        get_assessment: One assessment with its photo and label.
        submit_label: Stores the verdict of an operator.
        mark_readable: Cancels the verdict of the model about an unreadable photo.
        get_shipment: A shipment with its history.
        list_models: Registered model versions.
        activate_model: Makes a version the active one.
        rollback_model: Brings back the previous version.
        reload_model: Starts loading the active version now rather than at the next check.
        start_training: Queues a retraining.
        training_job_page: One page of retrainings, newest first.
        count_training_labels: Confirmed loads a retraining would learn from.
        worker_online: Whether a worker is running to take a retraining from the queue.
        storage: Photo bytes, shown on the pages.
        model: The model the service answers with, for the header.
        draw_heatmap: Draws what the model saw on a photo, or None when the service carries no
            explainer; then the pages offer no maps.
    """

    sign_in: SignIn
    authenticate: Authenticate
    sign_out: SignOut
    assess_photo: AssessPhoto
    review_queue: ReviewQueue
    get_assessment: GetAssessment
    submit_label: SubmitLabel
    mark_readable: MarkReadable
    get_shipment: GetShipment
    list_models: ListModelVersions
    activate_model: ActivateModel
    rollback_model: RollbackModel
    reload_model: Callable[[], object]
    start_training: StartTraining
    training_job_page: ListTrainingJobPage
    count_training_labels: CountTrainingLabels
    worker_online: Callable[[], bool]
    storage: ImageStorage
    model: ActiveModel
    draw_heatmap: DrawHeatmap | None = None


def ui_services_of(request: Request) -> UiServices:
    """Returns the services of the operator pages."""
    return cast(UiServices, request.app.state.ui_services)
