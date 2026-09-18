"""What the operator pages need from the rest of the service, put together once at startup."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from starlette.requests import Request

from loadvision.application.get_assessment import GetAssessment
from loadvision.application.get_shipment import GetShipment
from loadvision.application.model_versions import ActivateModel, ListModelVersions, RollbackModel
from loadvision.application.operators import Authenticate, SignIn, SignOut
from loadvision.application.ports import ImageStorage
from loadvision.application.review_queue import ReviewQueue
from loadvision.application.submit_label import SubmitLabel
from loadvision.application.training_jobs import ListTrainingJobs, StartTraining
from loadvision.ml.inference.active_model import ActiveModel


@dataclass(frozen=True, slots=True, kw_only=True)
class UiServices:
    """Dependencies of the operator pages; tests build them from fakes.

    Attributes:
        sign_in: Checks a password and opens a session.
        authenticate: Finds the session of a browser.
        sign_out: Ends a session.
        review_queue: Assessments waiting for a confirmed load.
        get_assessment: One assessment with its photo and label.
        submit_label: Stores the verdict of an operator.
        get_shipment: A shipment with its history.
        list_models: Registered model versions.
        activate_model: Makes a version the active one.
        rollback_model: Brings back the previous version.
        reload_model: Starts loading the active version now rather than at the next check.
        start_training: Queues a retraining.
        list_training_jobs: Recent retrainings.
        storage: Photo bytes, shown on the pages.
        model: The model the service answers with, for the header.
    """

    sign_in: SignIn
    authenticate: Authenticate
    sign_out: SignOut
    review_queue: ReviewQueue
    get_assessment: GetAssessment
    submit_label: SubmitLabel
    get_shipment: GetShipment
    list_models: ListModelVersions
    activate_model: ActivateModel
    rollback_model: RollbackModel
    reload_model: Callable[[], object]
    start_training: StartTraining
    list_training_jobs: ListTrainingJobs
    storage: ImageStorage
    model: ActiveModel


def ui_services_of(request: Request) -> UiServices:
    """Returns the services of the operator pages."""
    return cast(UiServices, request.app.state.ui_services)
