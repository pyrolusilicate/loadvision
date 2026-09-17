"""Let a confirmed label mark a photo as unusable instead of giving its load.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-17

An operator who sees a covered lens or a black frame confirms that the photo is unusable; such a
label has no load and training leaves the photo out. `unusable` repeats the NULL of `load_pct`
on purpose, so a reader of the table does not have to know what NULL means; the check keeps the
two in agreement.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Labels stored before this revision all have a load, hence the temporary default.
    op.add_column(
        "confirmed_labels",
        sa.Column("unusable", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column("confirmed_labels", "unusable", server_default=None)
    op.alter_column("confirmed_labels", "load_pct", existing_type=sa.Double(), nullable=True)
    op.create_check_constraint(
        op.f("ck_confirmed_labels_load_pct_unless_unusable"),
        "confirmed_labels",
        "(load_pct IS NULL) = unusable",
    )


def downgrade() -> None:
    # Revision 0002 has no place for an unusable photo: such labels block the downgrade.
    op.drop_constraint(
        op.f("ck_confirmed_labels_load_pct_unless_unusable"), "confirmed_labels", type_="check"
    )
    op.alter_column("confirmed_labels", "load_pct", existing_type=sa.Double(), nullable=False)
    op.drop_column("confirmed_labels", "unusable")
