"""The API and the operator pages on fakes, with helpers that check every XML response."""

import base64
from dataclasses import dataclass

from fastapi.testclient import TestClient
from httpx2 import Response
from lxml import etree

from loadvision.api.app import create_app
from loadvision.api.services import ApiServices
from loadvision.api.ui.services import UiServices
from loadvision.api.xml.parser import NAMESPACE, schema_bytes
from loadvision.application.assess_photo import AssessPhoto
from loadvision.application.get_assessment import GetAssessment
from loadvision.application.get_shipment import GetShipment
from loadvision.application.model_versions import (
    ActivateModel,
    ListModelVersions,
    RollbackModel,
)
from loadvision.application.operators import Authenticate, SignIn, SignOut
from loadvision.application.ports import ImageStorage, LoadPredictor
from loadvision.application.review_queue import ReviewQueue
from loadvision.application.submit_label import SubmitLabel
from loadvision.application.training_jobs import GetTrainingJob, ListTrainingJobs, StartTraining
from loadvision.errors import ERROR_CATALOG, ErrorCode
from loadvision.imaging.prepared_image import MediaType
from loadvision.imaging.validator import ImageLimits, prepare_image
from loadvision.ml.inference.active_model import ActiveModel
from tests import image_files
from tests.builders import T0
from tests.fakes.clock import FixedClock, SequentialIds
from tests.fakes.image_storage import InMemoryImageStorage
from tests.fakes.in_memory_db import InMemoryStore

LIMITS = ImageLimits(
    max_bytes=2_000_000, max_pixels=1_000_000, min_side=256, media_types=frozenset(MediaType)
)
MAX_BODY_BYTES = 1_000_000
PHOTO = image_files.jpeg()
PHOTO_SHA256 = prepare_image(PHOTO, LIMITS).sha256
XML = {"Content-Type": "application/xml; charset=utf-8"}
SCHEMA = etree.XMLSchema(etree.fromstring(schema_bytes()))


class ReloadCounter:
    """Stands in for loading the active model in the background; counts the requests."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


@dataclass
class Service:
    client: TestClient
    store: InMemoryStore
    storage: InMemoryImageStorage
    clock: FixedClock
    model: ActiveModel
    reloads: ReloadCounter


def make_service(
    predictor: LoadPredictor | None = None,
    storage: InMemoryImageStorage | None = None,
    photos: ImageStorage | None = None,
) -> Service:
    """Builds the service on fakes, with the operator pages mounted like in production.

    Args:
        predictor: Model the service answers with, or None while it is loading.
        storage: Where assessments save photos.
        photos: Where the pages read photos from; the same storage when None.
    """
    store = InMemoryStore()
    storage = InMemoryImageStorage() if storage is None else storage
    model = ActiveModel()
    if predictor is not None:
        model.activate(predictor)
    clock = FixedClock(T0)
    services = ApiServices(
        assess_photo=AssessPhoto(
            uow_factory=store.unit_of_work,
            storage=storage,
            limits=LIMITS,
            clock=clock,
            new_id=SequentialIds(),
        ),
        get_assessment=GetAssessment(uow_factory=store.unit_of_work),
        submit_label=SubmitLabel(uow_factory=store.unit_of_work, clock=clock),
        get_shipment=GetShipment(uow_factory=store.unit_of_work),
        list_models=ListModelVersions(uow_factory=store.unit_of_work),
        activate_model=ActivateModel(uow_factory=store.unit_of_work, clock=clock),
        start_training=StartTraining(
            uow_factory=store.unit_of_work, clock=clock, new_id=SequentialIds()
        ),
        list_training_jobs=ListTrainingJobs(uow_factory=store.unit_of_work),
        get_training_job=GetTrainingJob(uow_factory=store.unit_of_work),
        model=model,
        request_id_header="X-Request-ID",
        max_body_bytes=MAX_BODY_BYTES,
    )
    reloads = ReloadCounter()
    ui = UiServices(
        sign_in=SignIn(uow_factory=store.unit_of_work, clock=clock),
        authenticate=Authenticate(uow_factory=store.unit_of_work, clock=clock),
        sign_out=SignOut(uow_factory=store.unit_of_work),
        review_queue=ReviewQueue(uow_factory=store.unit_of_work),
        get_assessment=services.get_assessment,
        submit_label=services.submit_label,
        get_shipment=services.get_shipment,
        list_models=services.list_models,
        activate_model=services.activate_model,
        rollback_model=RollbackModel(
            uow_factory=store.unit_of_work, activate=services.activate_model
        ),
        reload_model=reloads,
        start_training=services.start_training,
        list_training_jobs=services.list_training_jobs,
        storage=storage if photos is None else photos,
        model=model,
    )
    client = TestClient(create_app(services, ui=ui), raise_server_exceptions=False)
    return Service(
        client=client, store=store, storage=storage, clock=clock, model=model, reloads=reloads
    )


def request_body(shipment_id: str = "RP-1", image: str | None = None) -> bytes:
    data = base64.b64encode(PHOTO).decode() if image is None else image
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<AssessmentRequest xmlns="{NAMESPACE}">'
        f"<ShipmentId>{shipment_id}</ShipmentId><Image>{data}</Image>"
        "</AssessmentRequest>"
    ).encode()


def post(client: TestClient, body: bytes, headers: dict[str, str] | None = None) -> Response:
    return client.post(
        "/api/v1/assessments", content=body, headers=XML if headers is None else headers
    )


def document(response: Response) -> etree._Element:
    """Parses a response and checks it is XML valid against the published schema."""
    assert response.headers["content-type"] == "application/xml; charset=utf-8"
    root = etree.fromstring(response.content)
    SCHEMA.assertValid(root)
    return root


def child(root: etree._Element, name: str) -> etree._Element:
    found = root.find(f"{{{NAMESPACE}}}{name}")
    assert found is not None, name
    return found


def assert_error(response: Response, code: ErrorCode) -> None:
    root = document(response)
    spec = ERROR_CATALOG[code]
    assert response.status_code == spec.http_status
    assert root.get("code") == code.value
    assert child(root, "Message").text == spec.message
    assert child(root, "Hint").text == spec.hint
    assert root.get("requestId") == response.headers["X-Request-ID"]
