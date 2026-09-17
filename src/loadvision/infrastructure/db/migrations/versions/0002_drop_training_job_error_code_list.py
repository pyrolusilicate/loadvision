"""Drop the list of allowed error codes from training jobs.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-17

The check repeated the whole error catalog, so every new code, even one the API alone uses,
needed a migration. Which codes exist is decided by `ErrorCode`; the database still requires a
code for a failed job (ck_training_jobs_error_when_failed).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(op.f("ck_training_jobs_error_code_known"), "training_jobs", type_="check")


def downgrade() -> None:
    # Restores the list of revision 0001; a job that failed with a newer code blocks the downgrade.
    op.create_check_constraint(
        op.f("ck_training_jobs_error_code_known"),
        "training_jobs",
        "error_code IN ('INVALID_CONTENT_TYPE', 'INVALID_XML', 'INVALID_SHIPMENT_ID', "
        "'INVALID_ENCODING', 'PAYLOAD_TOO_LARGE', 'UNSUPPORTED_FORMAT', 'CORRUPTED_IMAGE', "
        "'SHIPMENT_NOT_FOUND', 'ASSESSMENT_NOT_FOUND', 'TRAINING_ALREADY_RUNNING', "
        "'MODEL_VERSION_STATUS_CONFLICT', 'MODEL_NOT_READY', 'MODEL_INFERENCE_FAILED', "
        "'STAGE_CHECK_FAILED', 'MODEL_PACKAGE_INVALID', 'INTERNAL_ERROR')",
    )
