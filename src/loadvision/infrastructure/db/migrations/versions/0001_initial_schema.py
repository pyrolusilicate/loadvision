"""Initial schema: photos, shipments, assessments, labels, model versions, training jobs.

Revision ID: 0001
Revises:
Create Date: 2026-09-16

Constraint texts are copied, not imported from tables.py: a migration is a snapshot, and a later
edit of the tables must not rewrite history. Names are wrapped in op.f because Alembic applies the
naming convention of the tables to migrations too and would prefix a ready name again.
tests/integration/test_schema.py checks that a migrated database matches the tables exactly.
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _aware(name: str, *, nullable: bool = False) -> sa.Column[datetime]:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    op.create_table(
        "images",
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        _aware("created_at"),
        sa.PrimaryKeyConstraint("sha256", name=op.f("pk_images")),
        sa.CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_images_sha256_format")),
        sa.CheckConstraint(
            "media_type IN ('image/jpeg', 'image/png', 'image/webp')",
            name=op.f("ck_images_media_type_known"),
        ),
        sa.CheckConstraint("width > 0 AND height > 0", name=op.f("ck_images_size_positive")),
        sa.CheckConstraint("size_bytes > 0", name=op.f("ck_images_bytes_positive")),
    )
    op.create_table(
        "shipments",
        sa.Column("id", sa.Text(), nullable=False),
        _aware("created_at"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_shipments")),
        sa.CheckConstraint("id ~ '^[A-Za-z0-9_-]{1,64}$'", name=op.f("ck_shipments_id_format")),
    )
    op.create_table(
        "model_versions",
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        _aware("created_at"),
        _aware("activated_at", nullable=True),
        sa.Column("package_uri", sa.Text(), nullable=True),
        sa.Column("manifest", postgresql.JSONB(), nullable=True),
        sa.Column("cv_mae", sa.Double(), nullable=True),
        sa.Column("within_10", sa.Double(), nullable=True),
        sa.Column("gate_report", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("version", name=op.f("pk_model_versions")),
        sa.CheckConstraint(
            "version ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'",
            name=op.f("ck_model_versions_version_format"),
        ),
        sa.CheckConstraint(
            "status IN ('candidate', 'active', 'rejected', 'retired')",
            name=op.f("ck_model_versions_status_known"),
        ),
        sa.CheckConstraint(
            "(activated_at IS NOT NULL) = (status IN ('active', 'retired'))",
            name=op.f("ck_model_versions_activated_at_matches_status"),
        ),
        sa.CheckConstraint(
            "cv_mae >= 0 AND cv_mae < 'Infinity'::double precision",
            name=op.f("ck_model_versions_cv_mae_non_negative"),
        ),
        sa.CheckConstraint(
            "within_10 >= 0 AND within_10 <= 100", name=op.f("ck_model_versions_within_10_percent")
        ),
    )
    op.create_index(
        op.f("uq_model_versions_one_active"),
        "model_versions",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "assessments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("shipment_id", sa.Text(), nullable=False),
        sa.Column("image_sha256", sa.Text(), nullable=False),
        sa.Column("load_pct", sa.Double(), nullable=False),
        sa.Column("declared_load_pct", sa.Double(), nullable=True),
        sa.Column("uncertainty", sa.Double(), nullable=True),
        sa.Column("ood_score", sa.Double(), nullable=True),
        sa.Column("brightness", sa.Double(), nullable=True),
        sa.Column("sharpness", sa.Double(), nullable=True),
        sa.Column("review_reasons", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("processing_ms", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.Text(), nullable=True),
        _aware("created_at"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_assessments")),
        sa.ForeignKeyConstraint(
            ["shipment_id"], ["shipments.id"], name=op.f("fk_assessments_shipment_id")
        ),
        sa.ForeignKeyConstraint(
            ["image_sha256"], ["images.sha256"], name=op.f("fk_assessments_image_sha256")
        ),
        sa.ForeignKeyConstraint(
            ["model_version"], ["model_versions.version"], name=op.f("fk_assessments_model_version")
        ),
        sa.UniqueConstraint(
            "shipment_id", "image_sha256", name=op.f("uq_assessments_shipment_id_image_sha256")
        ),
        sa.CheckConstraint(
            "load_pct >= 0 AND load_pct <= 100", name=op.f("ck_assessments_load_pct_range")
        ),
        sa.CheckConstraint(
            "declared_load_pct >= 0 AND declared_load_pct <= 100",
            name=op.f("ck_assessments_declared_load_pct_range"),
        ),
        sa.CheckConstraint(
            "uncertainty >= 0 AND uncertainty < 'Infinity'::double precision",
            name=op.f("ck_assessments_uncertainty_non_negative"),
        ),
        sa.CheckConstraint(
            "ood_score >= 0 AND ood_score < 'Infinity'::double precision",
            name=op.f("ck_assessments_ood_score_non_negative"),
        ),
        sa.CheckConstraint(
            "brightness >= 0 AND brightness < 'Infinity'::double precision",
            name=op.f("ck_assessments_brightness_non_negative"),
        ),
        sa.CheckConstraint(
            "sharpness >= 0 AND sharpness < 'Infinity'::double precision",
            name=op.f("ck_assessments_sharpness_non_negative"),
        ),
        sa.CheckConstraint(
            "review_reasons <@ "
            "ARRAY['low_confidence', 'unusual_scene', 'too_dark', 'blurry']::text[]",
            name=op.f("ck_assessments_review_reasons_known"),
        ),
        sa.CheckConstraint(
            "processing_ms >= 0", name=op.f("ck_assessments_processing_ms_non_negative")
        ),
    )
    op.create_index(
        op.f("ix_assessments_shipment_id_created_at"), "assessments", ["shipment_id", "created_at"]
    )
    op.create_index(
        op.f("ix_assessments_image_sha256_model_version"),
        "assessments",
        ["image_sha256", "model_version"],
    )
    op.create_table(
        "confirmed_labels",
        sa.Column("image_sha256", sa.Text(), nullable=False),
        sa.Column("load_pct", sa.Double(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("author", sa.Text(), nullable=True),
        _aware("created_at"),
        _aware("updated_at"),
        sa.PrimaryKeyConstraint("image_sha256", name=op.f("pk_confirmed_labels")),
        sa.ForeignKeyConstraint(
            ["image_sha256"], ["images.sha256"], name=op.f("fk_confirmed_labels_image_sha256")
        ),
        sa.CheckConstraint(
            "load_pct >= 0 AND load_pct <= 100", name=op.f("ck_confirmed_labels_load_pct_range")
        ),
        sa.CheckConstraint(
            "source IN ('api', 'ui', 'import')", name=op.f("ck_confirmed_labels_source_known")
        ),
        sa.CheckConstraint(
            "updated_at >= created_at", name=op.f("ck_confirmed_labels_updated_after_created")
        ),
    )
    op.create_table(
        "training_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("base_version", sa.Text(), nullable=True),
        sa.Column("result_version", sa.Text(), nullable=True),
        sa.Column("gate_passed", sa.Boolean(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        _aware("created_at"),
        _aware("started_at", nullable=True),
        _aware("finished_at", nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_training_jobs")),
        sa.ForeignKeyConstraint(
            ["base_version"], ["model_versions.version"], name=op.f("fk_training_jobs_base_version")
        ),
        sa.ForeignKeyConstraint(
            ["result_version"],
            ["model_versions.version"],
            name=op.f("fk_training_jobs_result_version"),
        ),
        sa.CheckConstraint("mode IN ('fast', 'full')", name=op.f("ck_training_jobs_mode_known")),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name=op.f("ck_training_jobs_status_known"),
        ),
        sa.CheckConstraint(
            "error_code IN ('INVALID_CONTENT_TYPE', 'INVALID_XML', 'INVALID_SHIPMENT_ID', "
            "'INVALID_ENCODING', 'PAYLOAD_TOO_LARGE', 'UNSUPPORTED_FORMAT', 'CORRUPTED_IMAGE', "
            "'SHIPMENT_NOT_FOUND', 'ASSESSMENT_NOT_FOUND', 'TRAINING_ALREADY_RUNNING', "
            "'MODEL_VERSION_STATUS_CONFLICT', 'MODEL_NOT_READY', 'MODEL_INFERENCE_FAILED', "
            "'STAGE_CHECK_FAILED', 'MODEL_PACKAGE_INVALID', 'INTERNAL_ERROR')",
            name=op.f("ck_training_jobs_error_code_known"),
        ),
        sa.CheckConstraint(
            "(started_at IS NULL) = (status = 'queued')",
            name=op.f("ck_training_jobs_started_unless_queued"),
        ),
        sa.CheckConstraint(
            "(finished_at IS NOT NULL) = (status IN ('succeeded', 'failed'))",
            name=op.f("ck_training_jobs_finished_when_done"),
        ),
        sa.CheckConstraint(
            "(error_code IS NOT NULL) = (status = 'failed')",
            name=op.f("ck_training_jobs_error_when_failed"),
        ),
    )
    op.create_index(
        op.f("uq_training_jobs_one_running"),
        "training_jobs",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )


def downgrade() -> None:
    op.drop_table("training_jobs")
    op.drop_table("confirmed_labels")
    op.drop_table("assessments")
    op.drop_table("model_versions")
    op.drop_table("shipments")
    op.drop_table("images")
