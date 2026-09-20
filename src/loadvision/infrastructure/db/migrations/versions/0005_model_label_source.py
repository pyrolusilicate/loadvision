"""Let the service itself mark a photo unusable.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-20

A frame that holds nothing of a truck body is marked unusable at assessment time, so the
shipment does not count a prediction nobody should trust. The verdict carries its own source,
`model`, and a person cancels it from the pages; no label already stored changes.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONSTRAINT = "ck_confirmed_labels_source_known"
_WITH_MODEL = "source IN ('api', 'ui', 'import', 'model')"
_WITHOUT_MODEL = "source IN ('api', 'ui', 'import')"


def upgrade() -> None:
    op.drop_constraint(op.f(_CONSTRAINT), "confirmed_labels", type_="check")
    op.create_check_constraint(op.f(_CONSTRAINT), "confirmed_labels", _WITH_MODEL)


def downgrade() -> None:
    # Verdicts of the model have no place in revision 0004; a person's labels stay.
    op.execute("DELETE FROM confirmed_labels WHERE source = 'model'")
    op.drop_constraint(op.f(_CONSTRAINT), "confirmed_labels", type_="check")
    op.create_check_constraint(op.f(_CONSTRAINT), "confirmed_labels", _WITHOUT_MODEL)
