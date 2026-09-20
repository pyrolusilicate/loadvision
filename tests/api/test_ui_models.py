from datetime import timedelta
from uuid import UUID

import pytest

from loadvision.application.ports import LoadPredictor
from loadvision.domain.model_version import ModelVersionId, ModelVersionStatus
from loadvision.domain.training_job import TrainingJobStatus, TrainingMode
from loadvision.errors import ERROR_CATALOG, ErrorCode
from loadvision.ml.training.gate import DEFAULT_RULES
from tests.api.service import Service, make_service
from tests.api.ui_pages import get, page, sign_in, submit, text_of
from tests.builders import (
    FIRST_JOB_ID,
    T0,
    make_image_record,
    make_job,
    make_label,
    make_model_version,
)

NEEDED = DEFAULT_RULES.min_new_labels


def register(service: Service, *versions: tuple[str, ModelVersionStatus]) -> None:
    with service.store.unit_of_work() as uow:
        for number, (tag, status) in enumerate(versions):
            at = T0 + timedelta(hours=number)
            answered = status in (ModelVersionStatus.ACTIVE, ModelVersionStatus.RETIRED)
            activated = at if answered else None
            uow.model_versions.add(
                make_model_version(tag, status=status, created_at=at, activated_at=activated)
            )
        uow.commit()


def confirm_loads(service: Service, count: int = 1) -> None:
    """Stores photos with a confirmed load, the material of a retraining."""
    with service.store.unit_of_work() as uow:
        for number in range(count):
            raw = f"photo-{number}".encode()
            uow.images.add(make_image_record(raw))
            uow.labels.save(make_label(raw))
        uow.commit()


def status_of(service: Service, tag: str) -> ModelVersionStatus:
    return service.store.model_versions[ModelVersionId(tag)].status


@pytest.mark.api
def test_versions_are_listed_and_only_those_that_may_answer_can_be_switched_on(
    service: Service,
) -> None:
    sign_in(service)
    register(
        service,
        ("v1", ModelVersionStatus.RETIRED),
        ("v2", ModelVersionStatus.ACTIVE),
        ("v3", ModelVersionStatus.REJECTED),
        ("v4", ModelVersionStatus.CANDIDATE),
    )

    table = page(get(service, "/ui/models"))

    rows = [
        (row.xpath("string(td[1])"), row.xpath("string(td[2])"))
        for row in table.xpath("//tbody/tr")
    ]
    assert rows == [
        ("v4", "кандидат"),
        ("v3", "отвергнута гейтом"),
        ("v2", "работает"),
        ("v1", "устарела"),
    ]
    assert table.xpath("//tbody//form/@action") == [
        "/ui/models/v4/activate",
        "/ui/models/v1/activate",
    ]


@pytest.mark.api
def test_switching_a_version_on_makes_the_service_load_it(service: Service) -> None:
    token = sign_in(service)
    register(service, ("v1", ModelVersionStatus.ACTIVE), ("v2", ModelVersionStatus.CANDIDATE))

    response = submit(service, "/ui/models/v2/activate", csrf=token)

    assert (response.status_code, response.headers["location"]) == (303, "/ui/models")
    assert (status_of(service, "v1"), status_of(service, "v2")) == (
        ModelVersionStatus.RETIRED,
        ModelVersionStatus.ACTIVE,
    )
    assert service.reloads.calls == 1


@pytest.mark.api
def test_rolling_back_brings_the_previous_version_and_loads_it(service: Service) -> None:
    token = sign_in(service)
    register(service, ("v1", ModelVersionStatus.RETIRED), ("v2", ModelVersionStatus.ACTIVE))

    response = submit(service, "/ui/models/rollback", csrf=token)

    assert (response.status_code, response.headers["location"]) == (303, "/ui/models")
    assert status_of(service, "v1") is ModelVersionStatus.ACTIVE
    assert service.reloads.calls == 1


@pytest.mark.api
@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("/ui/models/rollback", ErrorCode.MODEL_VERSION_NOT_FOUND),
        ("/ui/models/v9/activate", ErrorCode.MODEL_VERSION_NOT_FOUND),
        ("/ui/models/bad%20tag/activate", ErrorCode.MODEL_VERSION_NOT_FOUND),
        ("/ui/models/v3/activate", ErrorCode.MODEL_VERSION_STATUS_CONFLICT),
    ],
)
def test_a_version_that_cannot_answer_leaves_the_model_as_it_was(
    service: Service, path: str, code: ErrorCode
) -> None:
    token = sign_in(service)
    register(service, ("v2", ModelVersionStatus.ACTIVE), ("v3", ModelVersionStatus.REJECTED))

    response = submit(service, path, csrf=token)

    spec = ERROR_CATALOG[code]
    assert response.status_code == spec.http_status
    assert spec.message in text_of(response)
    assert status_of(service, "v2") is ModelVersionStatus.ACTIVE
    assert service.reloads.calls == 0


@pytest.mark.api
@pytest.mark.parametrize("path", ["/ui/models/v2/activate", "/ui/models/rollback", "/ui/jobs"])
def test_model_forms_without_the_session_token_change_nothing(service: Service, path: str) -> None:
    sign_in(service)
    register(service, ("v1", ModelVersionStatus.RETIRED), ("v2", ModelVersionStatus.CANDIDATE))

    response = submit(service, path)

    assert response.status_code == 403
    assert (status_of(service, "v1"), status_of(service, "v2")) == (
        ModelVersionStatus.RETIRED,
        ModelVersionStatus.CANDIDATE,
    )
    assert service.store.training_jobs == {}
    assert service.reloads.calls == 0


@pytest.mark.api
def test_retraining_is_queued_and_listed(service: Service) -> None:
    token = sign_in(service)
    confirm_loads(service, NEEDED)

    response = submit(service, "/ui/jobs", csrf=token)

    assert (response.status_code, response.headers["location"]) == (303, "/ui/jobs")
    [job] = service.store.training_jobs.values()
    assert (job.mode, job.status) == (TrainingMode.FAST, TrainingJobStatus.QUEUED)
    row = page(get(service, "/ui/jobs")).xpath("//tbody/tr")[0]
    assert row.xpath("string(td[2])") == "в очереди"


@pytest.mark.api
def test_a_second_retraining_waits_for_the_first(service: Service) -> None:
    token = sign_in(service)
    confirm_loads(service, NEEDED)
    submit(service, "/ui/jobs", csrf=token)

    response = submit(service, "/ui/jobs", csrf=token)

    spec = ERROR_CATALOG[ErrorCode.TRAINING_ALREADY_RUNNING]
    assert response.status_code == spec.http_status
    assert spec.message in text_of(response)
    assert len(service.store.training_jobs) == 1


@pytest.mark.api
def test_a_finished_job_shows_its_version_and_the_gate(service: Service) -> None:
    sign_in(service)
    running = make_job().started(at=T0, base_version=None)
    finished = running.succeeded(
        at=T0 + timedelta(minutes=5),
        run_id="run-1",
        result_version=ModelVersionId("v5"),
        gate_passed=False,
    )
    service.store.training_jobs[FIRST_JOB_ID] = finished

    [row] = page(get(service, "/ui/jobs")).xpath("//tbody/tr")

    cells = [" ".join(cell.text_content().split()) for cell in row.xpath("td")]
    assert cells == [
        "2026-09-16 12:00:00 UTC",
        "готово",
        "2026-09-16 12:05:00 UTC",
        "v5",
        "отказал",
    ]


def missing_package() -> LoadPredictor:
    raise FileNotFoundError("models/v2/manifest.json")


@pytest.mark.api
def test_a_version_that_did_not_load_is_explained_while_the_old_one_answers(
    service: Service,
) -> None:
    sign_in(service)

    service.model.load_in_background(missing_package).join()

    assert "FileNotFoundError: models/v2/manifest.json" in text_of(get(service, "/ui/models"))
    assert page(get(service, "/ui/")).xpath("//span[contains(@class, 'badge')]/text()") == [
        "fake-1"
    ]


@pytest.mark.api
def test_the_header_tells_a_failed_first_load_from_a_slow_one() -> None:
    service = make_service()
    sign_in(service)

    service.model.load_in_background(missing_package).join()

    assert "модель не загрузилась" in text_of(get(service, "/ui/"))


@pytest.mark.api
def test_with_fewer_loads_than_the_gate_needs_retraining_is_refused(service: Service) -> None:
    token = sign_in(service)
    confirm_loads(service, NEEDED - 1)

    response = submit(service, "/ui/jobs", csrf=token)

    assert response.status_code == 400
    assert "Подтверждённых загрузок пока мало." in text_of(response)
    assert service.store.training_jobs == {}
    assert page(get(service, "/ui/jobs")).xpath("//form[@action='/ui/jobs']/button/@disabled")


@pytest.mark.api
def test_without_a_running_worker_retraining_is_refused(service: Service) -> None:
    token = sign_in(service)
    confirm_loads(service, NEEDED)
    service.worker.online = False

    response = submit(service, "/ui/jobs", csrf=token)

    assert response.status_code == 409
    assert "Воркер не запущен." in text_of(response)
    assert service.store.training_jobs == {}
    jobs = page(get(service, "/ui/jobs"))
    assert jobs.xpath("//form[@action='/ui/jobs']/button/@disabled")
    assert jobs.xpath("//span[contains(@class, 'badge')]/text()")[-1] == "не запущен"


@pytest.mark.api
def test_a_job_left_by_a_dead_worker_can_be_taken_off(service: Service) -> None:
    token = sign_in(service)
    confirm_loads(service, NEEDED)
    # An id of its own: the service hands out UUID(int=1) to the job started at the end.
    running = make_job(UUID(int=99)).started(at=T0, base_version=None)
    service.store.training_jobs[running.id] = running
    service.worker.online = False

    offered = page(get(service, "/ui/jobs"))
    assert offered.xpath("//form[@action='/ui/jobs/release']/button/text()") == [
        "Снять зависшую задачу"
    ]
    assert "новые дообучения не запускаются" in text_of(get(service, "/ui/jobs"))

    submit(service, "/ui/jobs/release", csrf=token)

    ended = service.store.training_jobs[running.id]
    assert ended.status.value == "failed"
    assert ended.error_code is ErrorCode.INTERNAL_ERROR
    # The queue is free again, so a retraining can be started once the worker is back.
    service.worker.online = True
    assert submit(service, "/ui/jobs", csrf=token).status_code == 303


@pytest.mark.api
def test_a_running_job_of_a_live_worker_is_not_taken_off(service: Service) -> None:
    token = sign_in(service)
    running = make_job().started(at=T0, base_version=None)
    service.store.training_jobs[running.id] = running

    response = submit(service, "/ui/jobs/release", csrf=token)

    assert response.status_code == 409
    assert "Воркер запущен, задача идёт." in text_of(response)
    assert service.store.training_jobs[running.id] == running
    # And the page of a live worker offers no such button at all.
    assert page(get(service, "/ui/jobs")).xpath("//form[@action='/ui/jobs/release']") == []


@pytest.mark.api
def test_the_page_counts_confirmed_loads_against_what_the_gate_needs(service: Service) -> None:
    sign_in(service)
    confirm_loads(service, 3)

    text = text_of(get(service, "/ui/jobs"))

    assert "Подтвержденных оценок, которых активная модель ещё не видела: 3." in text


@pytest.mark.api
def test_a_queued_job_says_it_waits_for_the_worker(service: Service) -> None:
    token = sign_in(service)
    confirm_loads(service, NEEDED)
    submit(service, "/ui/jobs", csrf=token)

    assert "Задача ждёт запуск воркера." in text_of(get(service, "/ui/jobs"))
