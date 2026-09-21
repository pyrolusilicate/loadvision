"""Keep why the gate let a candidate through or refused it, next to the decision itself.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-21

The decision was stored as a single boolean, so a page could say "отказал" and nothing else,
and the numbers behind it lived only in the log of the worker. The gate already writes the
reason in one line - how many photos the two versions share, how far apart they are and what
the tolerance was - and this column is where that line lands.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Jobs finished before this revision keep NULL: their reason was never stored anywhere.
    op.add_column("training_jobs", sa.Column("gate_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_jobs", "gate_reason")
