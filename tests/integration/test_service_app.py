import functools
import time
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from loadvision.api.main import build_service_app
from loadvision.api.services import ApiServices
from loadvision.api.xml.parser import NAMESPACE
from loadvision.application.model_versions import RegisterModel
from loadvision.application.operators import AddOperator
from loadvision.domain.confirmed_label import LabelSource
from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.timestamps import utc_now
from loadvision.infrastructure.db.session import PostgresUnitOfWork
from loadvision.settings import DEFAULT_CONFIG_PATH, Settings
from tests.api.service import PHOTO, PHOTO_SHA256, XML, child, document, request_body
from tests.api.ui_pages import LOGIN, PASSWORD, form_token
from tests.model_packages import build_numpy_package

VERSION = "baseline-v0"
CANDIDATE = "candidate-v1"
READY_SECONDS = 30.0
POLL_SECONDS = 0.05


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["storage"] = {"images_dir": tmp_path / "images", "models_dir": tmp_path / "models"}
    data["model"] = {"version": VERSION}
    return Settings.model_validate(data)


def build_package(
    settings: Settings, tmp_path: Path, version: str = VERSION, load: float = 55.0
) -> None:
    """Puts a package whose head predicts `load` for every photo into the models folder."""
    settings.storage.models_dir.mkdir(parents=True, exist_ok=True)
    build_numpy_package(tmp_path / f"build-{version}", load=load, version=version).rename(
        settings.storage.models_dir / version
    )


def services_of(app: FastAPI) -> ApiServices:
    services = app.state.services
    assert isinstance(services, ApiServices)
    return services


def wait_for_model(app: FastAPI, version: str | None = None) -> None:
    """Waits until a model is loaded, or `version` if given, or the loading has failed."""
    model = services_of(app).model

    def settled() -> bool:
        current = model.current()
        if model.failure is not None:
            return True
        return current is not None and (version is None or current.version.value == version)

    deadline = time.monotonic() + READY_SECONDS
    while not settled():
        assert time.monotonic() < deadline, "the model did not load in time"
        time.sleep(POLL_SECONDS)


@pytest.mark.integration
def test_service_assesses_labels_and_reports_on_postgres(
    settings: Settings, engine: Engine, tmp_path: Path
) -> None:
    build_package(settings, tmp_path)
    app = build_service_app(settings, engine)

    with TestClient(app) as client:
        wait_for_model(app)
        assert client.get("/health").status_code == 200
        created = client.post("/api/v1/assessments", content=request_body(), headers=XML)
        repeated = client.post("/api/v1/assessments", content=request_body(), headers=XML)
        assessment_id = document(created).get("id")
        label = client.put(
            f"/api/v1/assessments/{assessment_id}/label",
            content=f'<LabelRequest xmlns="{NAMESPACE}"><LoadPct>60</LoadPct></LabelRequest>',
            headers=XML,
        )
        read = client.get(f"/api/v1/assessments/{assessment_id}")
        shipment = client.get("/api/v1/shipments/RP-1")
        models = client.get("/api/v1/models")
        queued = client.post(
            "/api/v1/training-jobs",
            content=f'<TrainingJobRequest xmlns="{NAMESPACE}"/>'.encode(),
            headers=XML,
        )
        jobs = client.get("/api/v1/training-jobs")

    assert (created.status_code, repeated.status_code, label.status_code) == (201, 200, 200)
    assert document(repeated).get("id") == assessment_id
    assert child(document(created), "LoadPct").text == "55.0"
    assert child(document(read), "ConfirmedLabel").get("loadPct") == "60.0"
    assert child(document(shipment), "Current").get("confirmedLoadPct") == "60.0"
    assert (settings.storage.images_dir / PHOTO_SHA256[:2] / PHOTO_SHA256).read_bytes() == PHOTO
    # The version the service started with is registered and active, and the queue answers.
    [model] = document(models).findall(f"{{{NAMESPACE}}}Model")
    assert (model.get("version"), model.get("status")) == (VERSION, "active")
    assert queued.status_code == 201
    assert document(queued).get("status") == "queued"
    [job] = document(jobs).findall(f"{{{NAMESPACE}}}TrainingJob")
    assert job.get("id") == document(queued).get("id")
    with PostgresUnitOfWork(engine) as uow:
        active = uow.model_versions.get_active()
    assert active is not None
    assert active.version == ModelVersionId(VERSION)


@pytest.mark.integration
def test_service_without_a_package_answers_but_is_not_ready(
    settings: Settings, engine: Engine
) -> None:
    app = build_service_app(settings, engine)

    with TestClient(app) as client:
        wait_for_model(app)
        health = client.get("/health")
        shipment = client.get("/api/v1/shipments/RP-1")

    assert services_of(app).model.failure is not None
    assert health.status_code == 503
    assert shipment.status_code == 404
    with PostgresUnitOfWork(engine) as uow:
        assert uow.model_versions.list_all() == []


@pytest.mark.integration
def test_operator_labels_a_photo_and_switches_the_model_on_postgres(
    settings: Settings, engine: Engine, tmp_path: Path
) -> None:
    build_package(settings, tmp_path)
    build_package(settings, tmp_path, version=CANDIDATE, load=30.0)
    uow_factory = functools.partial(PostgresUnitOfWork, engine)
    AddOperator(uow_factory=uow_factory, clock=utc_now)(LOGIN, PASSWORD)
    RegisterModel(uow_factory=uow_factory, clock=utc_now)(ModelVersionId(CANDIDATE))
    app = build_service_app(settings, engine)

    with TestClient(app) as client:
        wait_for_model(app)
        created = client.post("/api/v1/assessments", content=request_body(), headers=XML)
        assessment_id = document(created).get("id")
        signed = client.post(
            "/ui/login", data={"login": LOGIN, "password": PASSWORD}, follow_redirects=False
        )
        token = form_token(client.get("/ui/"))
        labelled = client.post(
            f"/ui/assessments/{assessment_id}/label",
            data={"csrf": token, "action": "confirm", "load_pct": "62,5"},
            follow_redirects=False,
        )
        read = client.get(f"/api/v1/assessments/{assessment_id}")
        switched = client.post(
            f"/ui/models/{CANDIDATE}/activate", data={"csrf": token}, follow_redirects=False
        )
        # The page starts loading the new version; the service answers with it once loaded.
        wait_for_model(app, CANDIDATE)
        again = client.post("/api/v1/assessments", content=request_body("RP-2"), headers=XML)

    assert (signed.status_code, labelled.status_code, switched.status_code) == (303, 303, 303)
    label = child(document(read), "ConfirmedLabel")
    assert (label.get("loadPct"), label.get("author")) == ("62.5", LOGIN)
    assert services_of(app).model.failure is None
    assert child(document(again), "LoadPct").text == "30.0"
    assert child(document(again), "Model").get("version") == CANDIDATE
    with PostgresUnitOfWork(engine) as uow:
        active = uow.model_versions.get_active()
        stored = uow.labels.get(PHOTO_SHA256)
    assert active is not None
    assert active.version == ModelVersionId(CANDIDATE)
    assert stored is not None
    assert stored.source is LabelSource.UI
