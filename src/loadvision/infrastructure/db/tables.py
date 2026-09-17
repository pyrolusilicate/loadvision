"""SQLAlchemy table definitions.

The tables are the target schema: migrations create the same schema, and an integration test
compares a migrated database with one created from these tables. Formats and allowed values in
CHECK constraints are built from the domain constants, so the database cannot drift from them.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Final

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from loadvision.application.ports import TrainingMode
from loadvision.domain.assessment import SHA256_HEX_PATTERN, ReviewReason
from loadvision.domain.model_version import MODEL_VERSION_PATTERN, ModelVersionStatus
from loadvision.domain.shipment import SHIPMENT_ID_PATTERN
from loadvision.imaging.prepared_image import MediaType

# Fixed constraint names: migrations refer to them, and autogenerate stays deterministic.
NAMING_CONVENTION: Final = {
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
}

# Values without a domain type yet; they move to the domain with ConfirmedLabel and
# TrainingJob (task 7.1).
LABEL_SOURCES: Final = ("api", "ui", "import")
JOB_STATUSES: Final = ("queued", "running", "succeeded", "failed")

metadata: Final = MetaData(naming_convention=NAMING_CONVENTION)


def _matches(column: str, pattern: str) -> str:
    # Postgres "~" finds a match anywhere in the string, so the pattern is anchored here.
    return f"{column} ~ '^{pattern}$'"


def _one_of(column: str, values: Iterable[str]) -> str:
    return f"{column} IN ({_sql_strings(values)})"


def _sql_strings(values: Iterable[str]) -> str:
    items = [str(value) for value in values]
    if any("'" in item for item in items):
        raise ValueError("allowed values must not contain quotes")
    return ", ".join(f"'{item}'" for item in items)


def _percent(column: str) -> str:
    # Postgres orders NaN above every number, so "<= 100" also rejects NaN and infinity.
    return f"{column} >= 0 AND {column} <= 100"


def _non_negative(column: str) -> str:
    # NaN and infinity pass ">= 0" in Postgres; "< 'Infinity'" rejects both.
    return f"{column} >= 0 AND {column} < 'Infinity'::double precision"


def _aware(name: str, *, nullable: bool = False) -> Column[datetime]:
    return Column(name, DateTime(timezone=True), nullable=nullable)


images: Final = Table(
    "images",
    metadata,
    Column("sha256", Text, primary_key=True),
    Column("media_type", Text, nullable=False),
    Column("width", Integer, nullable=False),
    Column("height", Integer, nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    _aware("created_at"),
    CheckConstraint(_matches("sha256", SHA256_HEX_PATTERN), name="sha256_format"),
    CheckConstraint(_one_of("media_type", MediaType), name="media_type_known"),
    CheckConstraint("width > 0 AND height > 0", name="size_positive"),
    CheckConstraint("size_bytes > 0", name="bytes_positive"),
)

shipments: Final = Table(
    "shipments",
    metadata,
    Column("id", Text, primary_key=True),
    _aware("created_at"),
    CheckConstraint(_matches("id", SHIPMENT_ID_PATTERN), name="id_format"),
)

model_versions: Final = Table(
    "model_versions",
    metadata,
    Column("version", Text, primary_key=True),
    Column("status", Text, nullable=False),
    _aware("created_at"),
    _aware("activated_at", nullable=True),
    # Filled by `model register` (task 4.10); the domain version does not carry them yet.
    Column("package_uri", Text),
    Column("manifest", JSONB),
    Column("cv_mae", Double),
    Column("within_10", Double),
    Column("gate_report", JSONB),
    CheckConstraint(_matches("version", MODEL_VERSION_PATTERN), name="version_format"),
    CheckConstraint(_one_of("status", ModelVersionStatus), name="status_known"),
    # The same rule as in the domain ModelVersion: only a version that has been active has an
    # activation time.
    CheckConstraint(
        "(activated_at IS NOT NULL) = "
        f"({_one_of('status', (ModelVersionStatus.ACTIVE, ModelVersionStatus.RETIRED))})",
        name="activated_at_matches_status",
    ),
    CheckConstraint(_non_negative("cv_mae"), name="cv_mae_non_negative"),
    CheckConstraint(_percent("within_10"), name="within_10_percent"),
    Index(
        "uq_model_versions_one_active",
        "status",
        unique=True,
        postgresql_where=text(f"status = '{ModelVersionStatus.ACTIVE}'"),
    ),
)

assessments: Final = Table(
    "assessments",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("shipment_id", Text, ForeignKey(shipments.c.id), nullable=False),
    Column("image_sha256", Text, ForeignKey(images.c.sha256), nullable=False),
    Column("load_pct", Double, nullable=False),
    Column("declared_load_pct", Double),
    Column("uncertainty", Double),
    Column("ood_score", Double),
    Column("brightness", Double),
    Column("sharpness", Double),
    # needs_review is not stored: it is derived from the reasons.
    Column("review_reasons", ARRAY(Text), nullable=False),
    Column("model_version", Text, ForeignKey(model_versions.c.version), nullable=False),
    Column("processing_ms", Integer, nullable=False),
    Column("request_id", Text),
    _aware("created_at"),
    UniqueConstraint("shipment_id", "image_sha256"),
    CheckConstraint(_percent("load_pct"), name="load_pct_range"),
    CheckConstraint(_percent("declared_load_pct"), name="declared_load_pct_range"),
    CheckConstraint(_non_negative("uncertainty"), name="uncertainty_non_negative"),
    CheckConstraint(_non_negative("ood_score"), name="ood_score_non_negative"),
    CheckConstraint(_non_negative("brightness"), name="brightness_non_negative"),
    CheckConstraint(_non_negative("sharpness"), name="sharpness_non_negative"),
    CheckConstraint(
        f"review_reasons <@ ARRAY[{_sql_strings(ReviewReason)}]::text[]",
        name="review_reasons_known",
    ),
    CheckConstraint("processing_ms >= 0", name="processing_ms_non_negative"),
    # Shipment history, newest first.
    Index("ix_assessments_shipment_id_created_at", "shipment_id", "created_at"),
    # Reuse of a prediction for the same photo and model version in another shipment.
    Index("ix_assessments_image_sha256_model_version", "image_sha256", "model_version"),
)

confirmed_labels: Final = Table(
    "confirmed_labels",
    metadata,
    # One label per photo: a repeated confirmation replaces it.
    Column("image_sha256", Text, ForeignKey(images.c.sha256), primary_key=True),
    Column("load_pct", Double, nullable=False),
    Column("source", Text, nullable=False),
    Column("author", Text),
    _aware("created_at"),
    _aware("updated_at"),
    CheckConstraint(_percent("load_pct"), name="load_pct_range"),
    CheckConstraint(_one_of("source", LABEL_SOURCES), name="source_known"),
    CheckConstraint("updated_at >= created_at", name="updated_after_created"),
)

training_jobs: Final = Table(
    "training_jobs",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("mode", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("run_id", Text),
    Column("base_version", Text, ForeignKey(model_versions.c.version)),
    Column("result_version", Text, ForeignKey(model_versions.c.version)),
    Column("gate_passed", Boolean),
    Column("error_code", Text),
    _aware("created_at"),
    _aware("started_at", nullable=True),
    _aware("finished_at", nullable=True),
    CheckConstraint(_one_of("mode", TrainingMode), name="mode_known"),
    CheckConstraint(_one_of("status", JOB_STATUSES), name="status_known"),
    CheckConstraint("(started_at IS NULL) = (status = 'queued')", name="started_unless_queued"),
    CheckConstraint(
        "(finished_at IS NOT NULL) = (status IN ('succeeded', 'failed'))",
        name="finished_when_done",
    ),
    CheckConstraint("(error_code IS NOT NULL) = (status = 'failed')", name="error_when_failed"),
    Index(
        "uq_training_jobs_one_running",
        "status",
        unique=True,
        postgresql_where=text("status = 'running'"),
    ),
)
