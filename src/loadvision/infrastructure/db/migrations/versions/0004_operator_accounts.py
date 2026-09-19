"""Operator accounts and their sign-in sessions for the operator pages.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-18

Accounts are made from the command line. A session row keeps the SHA-256 of the browser's
token, not the token, so a copy of the table lets nobody in. Failed sign-ins are counted on the
account row: thirty in fifteen minutes lock the login until the window ends.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "operators",
        sa.Column("login", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("failed_sign_ins", sa.Integer(), nullable=False),
        sa.Column("first_failure_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "login ~ '^[a-z0-9][a-z0-9._-]{0,63}$'", name=op.f("ck_operators_login_format")
        ),
        sa.CheckConstraint(
            "failed_sign_ins >= 0", name=op.f("ck_operators_failed_sign_ins_non_negative")
        ),
        sa.CheckConstraint(
            "(failed_sign_ins = 0) = (first_failure_at IS NULL)",
            name=op.f("ck_operators_failure_window_matches_count"),
        ),
        sa.PrimaryKeyConstraint("login", name=op.f("pk_operators")),
    )
    op.create_table(
        "operator_sessions",
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("login", sa.Text(), nullable=False),
        sa.Column("csrf", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "token_hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_operator_sessions_token_hash_format")
        ),
        sa.CheckConstraint(
            "expires_at > created_at", name=op.f("ck_operator_sessions_expires_after_created")
        ),
        sa.ForeignKeyConstraint(
            ["login"], ["operators.login"], name=op.f("fk_operator_sessions_login")
        ),
        sa.PrimaryKeyConstraint("token_hash", name=op.f("pk_operator_sessions")),
    )
    op.create_index("ix_operator_sessions_login", "operator_sessions", ["login"])


def downgrade() -> None:
    op.drop_index("ix_operator_sessions_login", table_name="operator_sessions")
    op.drop_table("operator_sessions")
    op.drop_table("operators")
