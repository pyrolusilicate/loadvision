"""In-memory unit of work and repositories that follow the port contracts of the database."""

from types import TracebackType
from typing import Self
from uuid import UUID

from loadvision.application.ports import AssessmentAlreadyExistsError, ImageRecord
from loadvision.domain.assessment import Assessment
from loadvision.domain.confirmed_label import ConfirmedLabel
from loadvision.domain.model_version import ModelVersion, ModelVersionId, ModelVersionStatus
from loadvision.domain.shipment import Shipment, ShipmentId
from loadvision.domain.training_job import TrainingJob, TrainingJobStatus


class _Table[K, V]:
    """Committed rows of one table plus the rows written in the current unit of work."""

    def __init__(self, committed: dict[K, V]) -> None:
        self._committed = committed
        self._pending: dict[K, V] = {}

    def get(self, key: K) -> V | None:
        if key in self._pending:
            return self._pending[key]
        return self._committed.get(key)

    def put(self, key: K, value: V) -> None:
        self._pending[key] = value

    def rows(self) -> list[V]:
        return list({**self._committed, **self._pending}.values())

    def commit(self) -> None:
        self._committed.update(self._pending)
        self._pending.clear()


class InMemoryShipmentRepository:
    def __init__(self, table: _Table[ShipmentId, Shipment]) -> None:
        self._table = table

    def get(self, shipment_id: ShipmentId) -> Shipment | None:
        return self._table.get(shipment_id)

    def add(self, shipment: Shipment) -> None:
        if self._table.get(shipment.id) is None:
            self._table.put(shipment.id, shipment)


class InMemoryAssessmentRepository:
    def __init__(self, table: _Table[UUID, Assessment]) -> None:
        self._table = table

    def add(self, assessment: Assessment) -> None:
        # Mirrors the primary key: a repeated id fails instead of silently replacing a row.
        if self._table.get(assessment.id) is not None:
            raise ValueError(f"assessment {assessment.id} already exists")
        stored = self.find_by_shipment_and_image(assessment.shipment_id, assessment.image_sha256)
        if stored is not None:
            raise AssessmentAlreadyExistsError(
                f"shipment {assessment.shipment_id.value} already has this photo"
            )
        self._table.put(assessment.id, assessment)

    def get(self, assessment_id: UUID) -> Assessment | None:
        return self._table.get(assessment_id)

    def find_by_shipment_and_image(
        self, shipment_id: ShipmentId, image_sha256: str
    ) -> Assessment | None:
        return next(
            (
                row
                for row in self._table.rows()
                if row.shipment_id == shipment_id and row.image_sha256 == image_sha256
            ),
            None,
        )

    def find_by_image_and_model(
        self, image_sha256: str, model_version: ModelVersionId
    ) -> Assessment | None:
        return next(
            (
                row
                for row in self._table.rows()
                if row.image_sha256 == image_sha256 and row.model_version == model_version
            ),
            None,
        )

    def list_by_shipment(self, shipment_id: ShipmentId) -> list[Assessment]:
        rows = [row for row in self._table.rows() if row.shipment_id == shipment_id]
        return sorted(rows, key=lambda row: row.created_at, reverse=True)


class InMemoryImageRepository:
    def __init__(self, table: _Table[str, ImageRecord]) -> None:
        self._table = table

    def add(self, record: ImageRecord) -> None:
        if self._table.get(record.sha256) is None:
            self._table.put(record.sha256, record)

    def get(self, sha256: str) -> ImageRecord | None:
        return self._table.get(sha256)


class InMemoryLabelRepository:
    def __init__(self, table: _Table[str, ConfirmedLabel]) -> None:
        self._table = table

    def get(self, image_sha256: str) -> ConfirmedLabel | None:
        return self._table.get(image_sha256)

    def save(self, label: ConfirmedLabel) -> None:
        self._table.put(label.image_sha256, label)

    def list_for_training(self) -> list[ConfirmedLabel]:
        usable = [row for row in self._table.rows() if row.load_pct is not None]
        return sorted(usable, key=lambda row: row.image_sha256)


class InMemoryTrainingJobRepository:
    def __init__(self, table: _Table[UUID, TrainingJob]) -> None:
        self._table = table

    def add(self, job: TrainingJob) -> None:
        if self._table.get(job.id) is not None:
            raise ValueError(f"training job {job.id} already exists")
        self._table.put(job.id, job)

    def get(self, job_id: UUID) -> TrainingJob | None:
        return self._table.get(job_id)

    def find_unfinished(self) -> TrainingJob | None:
        return next(
            iter(self._by_status(TrainingJobStatus.QUEUED, TrainingJobStatus.RUNNING)), None
        )

    def claim_next(self) -> TrainingJob | None:
        # No locking here: the fake has no parallel workers, and the queue under load is
        # tested against Postgres.
        if self._by_status(TrainingJobStatus.RUNNING):
            return None
        return next(iter(self._by_status(TrainingJobStatus.QUEUED)), None)

    def list_recent(self, limit: int) -> list[TrainingJob]:
        newest = sorted(self._table.rows(), key=lambda row: (row.created_at, str(row.id)))
        return list(reversed(newest))[:limit]

    def update(self, job: TrainingJob) -> None:
        if self._table.get(job.id) is None:
            raise ValueError(f"training job {job.id} is not stored")
        self._table.put(job.id, job)

    def _by_status(self, *statuses: TrainingJobStatus) -> list[TrainingJob]:
        rows = [row for row in self._table.rows() if row.status in statuses]
        return sorted(rows, key=lambda row: (row.created_at, str(row.id)))


class InMemoryModelVersionRepository:
    def __init__(self, table: _Table[ModelVersionId, ModelVersion]) -> None:
        self._table = table

    def get(self, version: ModelVersionId) -> ModelVersion | None:
        return self._table.get(version)

    def get_active(self) -> ModelVersion | None:
        return next(
            (row for row in self._table.rows() if row.status is ModelVersionStatus.ACTIVE), None
        )

    def list_all(self) -> list[ModelVersion]:
        return sorted(self._table.rows(), key=lambda row: row.created_at, reverse=True)

    def add(self, model_version: ModelVersion) -> None:
        if self._table.get(model_version.version) is not None:
            raise ValueError(f"model version {model_version.version.value} is already registered")
        self._put(model_version)

    def update(self, model_version: ModelVersion) -> None:
        if self._table.get(model_version.version) is None:
            raise ValueError(f"model version {model_version.version.value} is not registered")
        self._put(model_version)

    def _put(self, model_version: ModelVersion) -> None:
        # Mirrors the partial unique index of the database: at most one active version, so a use
        # case that forgets to retire the previous one fails in unit tests too.
        active = self.get_active()
        if (
            model_version.status is ModelVersionStatus.ACTIVE
            and active is not None
            and active.version != model_version.version
        ):
            raise ValueError(f"model version {active.version.value} is already active")
        self._table.put(model_version.version, model_version)


class _Tables:
    def __init__(self, store: "InMemoryStore") -> None:
        self.shipments = _Table(store.shipments)
        self.assessments = _Table(store.assessments)
        self.images = _Table(store.images)
        self.model_versions = _Table(store.model_versions)
        self.labels = _Table(store.labels)
        self.training_jobs = _Table(store.training_jobs)

    def commit(self) -> None:
        self.shipments.commit()
        self.assessments.commit()
        self.images.commit()
        self.model_versions.commit()
        self.labels.commit()
        self.training_jobs.commit()


class InMemoryUnitOfWork:
    """Unit of work over an `InMemoryStore`.

    Writes stay pending until `commit`. Units of work see each other's commits but are not
    isolated from each other: races are tested against Postgres, not here.
    """

    def __init__(self, store: "InMemoryStore") -> None:
        self._store = store
        self._tables: _Tables | None = None

    @property
    def shipments(self) -> InMemoryShipmentRepository:
        return InMemoryShipmentRepository(self._open().shipments)

    @property
    def assessments(self) -> InMemoryAssessmentRepository:
        return InMemoryAssessmentRepository(self._open().assessments)

    @property
    def images(self) -> InMemoryImageRepository:
        return InMemoryImageRepository(self._open().images)

    @property
    def model_versions(self) -> InMemoryModelVersionRepository:
        return InMemoryModelVersionRepository(self._open().model_versions)

    @property
    def labels(self) -> InMemoryLabelRepository:
        return InMemoryLabelRepository(self._open().labels)

    @property
    def training_jobs(self) -> InMemoryTrainingJobRepository:
        return InMemoryTrainingJobRepository(self._open().training_jobs)

    def __enter__(self) -> Self:
        self._tables = _Tables(self._store)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._tables = None

    def commit(self) -> None:
        self._open().commit()

    def _open(self) -> _Tables:
        if self._tables is None:
            raise RuntimeError("unit of work is used outside its with block")
        return self._tables


class InMemoryStore:
    """Committed data shared by the units of work it creates; one store per test."""

    def __init__(self) -> None:
        self.shipments: dict[ShipmentId, Shipment] = {}
        self.assessments: dict[UUID, Assessment] = {}
        self.images: dict[str, ImageRecord] = {}
        self.model_versions: dict[ModelVersionId, ModelVersion] = {}
        self.labels: dict[str, ConfirmedLabel] = {}
        self.training_jobs: dict[UUID, TrainingJob] = {}

    def unit_of_work(self) -> InMemoryUnitOfWork:
        return InMemoryUnitOfWork(self)
