"""Keep what an operator weighed an assessment with, and what came out.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-22

The weight of the cargo was computed on the page and forgotten as soon as it was left. It
belongs to the assessment rather than to the photo: the same photo in another shipment may sit
in another body, and the body is what sets the volume.

Both sides are stored. The inputs, so the number can be seen for what it is and redone, and the
answer, because the table of densities is ours to change and a stored weight must not change
under it by itself.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from loadvision.domain.cargo_weight import CargoKind

revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KINDS = ", ".join(f"'{kind.value}'" for kind in CargoKind)


def upgrade() -> None:
    op.create_table(
        "weighings",
        # One weighing per assessment: weighing it again replaces the old numbers.
        sa.Column(
            "assessment_id",
            sa.Uuid(),
            sa.ForeignKey("assessments.id"),
            primary_key=True,
        ),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("length_m", sa.Double(), nullable=False),
        sa.Column("width_m", sa.Double(), nullable=False),
        sa.Column("height_m", sa.Double(), nullable=False),
        sa.Column("cargo_kind", sa.Text(), nullable=False),
        sa.Column("kilograms", sa.Double(), nullable=False),
        sa.Column("low_kg", sa.Double(), nullable=False),
        sa.Column("high_kg", sa.Double(), nullable=False),
        sa.Column("cubic_metres", sa.Double(), nullable=False),
        sa.Column("author", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(f"cargo_kind IN ({_KINDS})", name="cargo_kind_known"),
        sa.CheckConstraint("length_m > 0 AND width_m > 0 AND height_m > 0", name="sides_positive"),
        sa.CheckConstraint(
            "low_kg <= kilograms AND kilograms <= high_kg", name="weight_within_bounds"
        ),
        sa.CheckConstraint("cubic_metres >= 0", name="cubic_metres_non_negative"),
    )


def downgrade() -> None:
    op.drop_table("weighings")
