from datetime import timedelta

import pytest
from lxml import etree

from loadvision.api.xml.parser import NAMESPACE
from loadvision.domain.model_version import ModelVersionStatus
from loadvision.errors import ErrorCode
from tests.api.service import Service, assert_error, document
from tests.builders import T0, make_model_version


def models_of(root: etree._Element) -> list[dict[str, str]]:
    return [dict(model.attrib) for model in root.iterfind(f"{{{NAMESPACE}}}Model")]


def register(service: Service, *versions: str) -> None:
    with service.store.unit_of_work() as uow:
        for number, tag in enumerate(versions):
            uow.model_versions.add(make_model_version(tag, created_at=T0 + timedelta(hours=number)))
        uow.commit()


@pytest.mark.api
def test_an_empty_registry_is_an_empty_list(service: Service) -> None:
    response = service.client.get("/api/v1/models")

    assert response.status_code == 200
    assert models_of(document(response)) == []


@pytest.mark.api
def test_versions_come_newest_registration_first(service: Service) -> None:
    register(service, "v1", "v2")

    root = document(service.client.get("/api/v1/models"))

    assert models_of(root) == [
        {"version": "v2", "status": "candidate", "createdAt": "2026-09-16T13:00:00Z"},
        {"version": "v1", "status": "candidate", "createdAt": "2026-09-16T12:00:00Z"},
    ]


@pytest.mark.api
def test_activation_answers_with_the_active_version(service: Service) -> None:
    register(service, "v1", "v2")
    service.client.post("/api/v1/models/v1/activate")

    response = service.client.post("/api/v1/models/v2/activate")

    assert response.status_code == 200
    root = document(response)
    assert etree.QName(root).localname == "Model"
    assert root.get("status") == "active"
    assert root.get("activatedAt") == "2026-09-16T12:00:00Z"
    stored = service.store.model_versions
    assert [version.status for version in stored.values()] == [
        ModelVersionStatus.RETIRED,
        ModelVersionStatus.ACTIVE,
    ]


@pytest.mark.api
@pytest.mark.parametrize("version", ["v404", "%20", ".hidden"], ids=["unknown", "space", "dot"])
def test_an_unknown_version_is_not_found(service: Service, version: str) -> None:
    register(service, "v1")

    response = service.client.post(f"/api/v1/models/{version}/activate")

    assert_error(response, ErrorCode.MODEL_VERSION_NOT_FOUND)
    assert service.store.model_versions[make_model_version("v1").version].status is (
        ModelVersionStatus.CANDIDATE
    )


@pytest.mark.api
def test_a_rejected_version_is_refused(service: Service) -> None:
    register(service, "v1")
    with service.store.unit_of_work() as uow:
        uow.model_versions.update(make_model_version("v1").reject())
        uow.commit()

    response = service.client.post("/api/v1/models/v1/activate")

    assert_error(response, ErrorCode.MODEL_VERSION_STATUS_CONFLICT)


@pytest.mark.api
def test_the_registry_is_only_read(service: Service) -> None:
    response = service.client.post("/api/v1/models")

    assert_error(response, ErrorCode.METHOD_NOT_ALLOWED)
    assert response.headers["allow"] == "GET"
