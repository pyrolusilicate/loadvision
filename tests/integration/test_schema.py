import re
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from uuid import UUID

import psycopg
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import URL, Connection, Engine, inspect, text
from sqlalchemy.exc import IntegrityError

from loadvision.domain.assessment import SHA256_HEX_PATTERN, ReviewReason
from loadvision.domain.model_version import ModelVersion, ModelVersionId, ModelVersionStatus
from loadvision.domain.shipment import ShipmentId
from loadvision.errors import LoadvisionError
from loadvision.infrastructure.db import tables
from tests.builders import T0
from tests.integration.database import alembic_config, engine_for, throwaway_database

type Row = dict[str, Any]

SHA_A = "a" * 64
SHA_B = "b" * 64
NAN = float("nan")
INF = float("inf")


def image_row(**overrides: Any) -> Row:
    row: Row = {
        "sha256": SHA_B,
        "media_type": "image/jpeg",
        "width": 1440,
        "height": 1920,
        "size_bytes": 240_000,
        "created_at": T0,
    }
    return row | overrides


def shipment_row(**overrides: Any) -> Row:
    return {"id": "RP-2", "created_at": T0} | overrides


def model_row(**overrides: Any) -> Row:
    return {"version": "v2", "status": "candidate", "created_at": T0, "activated_at": None} | (
        overrides
    )


def assessment_row(**overrides: Any) -> Row:
    row: Row = {
        "id": UUID(int=2),
        "shipment_id": "RP-1",
        "image_sha256": SHA_A,
        "load_pct": 55.0,
        "review_reasons": [],
        "model_version": "baseline-v0",
        "processing_ms": 180,
        "created_at": T0,
    }
    return row | overrides


def label_row(**overrides: Any) -> Row:
    row: Row = {
        "image_sha256": SHA_A,
        "load_pct": 60.0,
        "unusable": False,
        "source": "api",
        "created_at": T0,
        "updated_at": T0,
    }
    return row | overrides


def operator_row(**overrides: Any) -> Row:
    row: Row = {
        "login": "operator-17",
        "password_hash": "scrypt$00$11",
        "active": True,
        "created_at": T0,
        "failed_sign_ins": 0,
        "first_failure_at": None,
    }
    return row | overrides


def session_row(**overrides: Any) -> Row:
    row: Row = {
        "token_hash": SHA_A,
        "login": "operator-1",
        "csrf": "csrf-token",
        "created_at": T0,
        "expires_at": T0 + timedelta(hours=12),
    }
    return row | overrides


def job_row(**overrides: Any) -> Row:
    return {"id": UUID(int=2), "mode": "fast", "status": "queued", "created_at": T0} | overrides


ROW_BUILDERS: dict[str, Callable[..., Row]] = {
    "images": image_row,
    "shipments": shipment_row,
    "model_versions": model_row,
    "assessments": assessment_row,
    "confirmed_labels": label_row,
    "training_jobs": job_row,
    "operators": operator_row,
    "operator_sessions": session_row,
}


def insert(connection: Connection, table_name: str, row: Row) -> None:
    connection.execute(tables.metadata.tables[table_name].insert().values(row))


def violated_constraint(connection: Connection, table_name: str, row: Row) -> str | None:
    """Inserts a row inside a savepoint; returns the name of the violated constraint, if any."""
    savepoint = connection.begin_nested()
    try:
        insert(connection, table_name, row)
    except IntegrityError as error:
        savepoint.rollback()
        assert isinstance(error.orig, psycopg.Error)
        return error.orig.diag.constraint_name or "unnamed"
    savepoint.commit()
    return None


def schema_snapshot(connection: Connection) -> dict[str, list[tuple[Any, ...]]]:
    """Columns, constraints and indexes as Postgres itself prints them."""
    queries = {
        "columns": """
            SELECT table_name, column_name, data_type, udt_name, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name <> 'alembic_version'
            ORDER BY table_name, column_name
        """,
        "constraints": """
            SELECT rel.relname, con.conname, pg_get_constraintdef(con.oid)
            FROM pg_constraint AS con
            JOIN pg_class AS rel ON rel.oid = con.conrelid
            JOIN pg_namespace AS ns ON ns.oid = rel.relnamespace
            WHERE ns.nspname = 'public' AND rel.relname <> 'alembic_version'
            ORDER BY rel.relname, con.conname
        """,
        "indexes": """
            SELECT tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public' AND tablename <> 'alembic_version'
            ORDER BY tablename, indexname
        """,
    }
    return {
        name: [tuple(row) for row in connection.execute(text(sql))] for name, sql in queries.items()
    }


@pytest.fixture
def seeded(connection: Connection) -> Connection:
    """Connection with one photo, one shipment, one candidate version and one operator."""
    insert(connection, "images", image_row(sha256=SHA_A))
    insert(connection, "shipments", shipment_row(id="RP-1"))
    insert(connection, "model_versions", model_row(version="baseline-v0"))
    insert(connection, "operators", operator_row(login="operator-1"))
    return connection


@pytest.mark.integration
def test_migrated_schema_is_the_schema_of_the_tables(
    server_url: URL, migrated_engine: Engine
) -> None:
    # Autogenerate compares neither CHECK texts nor partial index conditions, so the catalogs
    # of both databases are compared as Postgres prints them.
    with throwaway_database(server_url) as url, engine_for(url) as engine:
        with engine.begin() as connection:
            tables.metadata.create_all(connection)
        with engine.connect() as connection:
            expected = schema_snapshot(connection)
    with migrated_engine.connect() as connection:
        assert schema_snapshot(connection) == expected


@pytest.mark.integration
def test_autogenerate_finds_no_difference(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), tables.metadata) == []


@pytest.mark.integration
def test_downgrade_removes_the_schema_and_upgrade_restores_it(server_url: URL) -> None:
    with throwaway_database(server_url) as url, engine_for(url) as engine:
        config = alembic_config(url)
        command.upgrade(config, "head")
        command.downgrade(config, "base")
        assert set(inspect(engine).get_table_names()) == {"alembic_version"}
        command.upgrade(config, "head")
        assert set(inspect(engine).get_table_names()) == {
            "alembic_version",
            *tables.metadata.tables,
        }


def check_names(engine: Engine, table_name: str) -> set[str]:
    return {
        str(check["name"])
        for check in inspect(engine).get_check_constraints(table_name)
        if check["name"] is not None
    }


@pytest.mark.integration
def test_each_revision_downgrades_to_the_previous_schema(server_url: URL) -> None:
    with throwaway_database(server_url) as url, engine_for(url) as engine:
        config = alembic_config(url)
        command.upgrade(config, "head")
        assert "ck_training_jobs_error_code_known" not in check_names(engine, "training_jobs")
        assert "ck_confirmed_labels_load_pct_unless_unusable" in check_names(
            engine, "confirmed_labels"
        )
        command.downgrade(config, "0003")
        assert {"operators", "operator_sessions"}.isdisjoint(inspect(engine).get_table_names())
        command.downgrade(config, "0002")
        assert "ck_confirmed_labels_load_pct_unless_unusable" not in check_names(
            engine, "confirmed_labels"
        )
        assert "unusable" not in {
            c["name"] for c in inspect(engine).get_columns("confirmed_labels")
        }
        command.downgrade(config, "0001")
        assert "ck_training_jobs_error_code_known" in check_names(engine, "training_jobs")
        command.upgrade(config, "head")
        assert "ck_training_jobs_error_code_known" not in check_names(engine, "training_jobs")
        assert "ck_confirmed_labels_load_pct_unless_unusable" in check_names(
            engine, "confirmed_labels"
        )


@pytest.mark.integration
def test_every_table_has_a_row_builder() -> None:
    assert set(ROW_BUILDERS) == set(tables.metadata.tables)


@pytest.mark.integration
@pytest.mark.parametrize("table_name", sorted(ROW_BUILDERS))
def test_accepts_a_valid_row(seeded: Connection, table_name: str) -> None:
    assert violated_constraint(seeded, table_name, ROW_BUILDERS[table_name]()) is None


@pytest.mark.integration
@pytest.mark.parametrize(
    ("table_name", "overrides"),
    [
        ("assessments", {"load_pct": 0.0, "declared_load_pct": 100.0}),
        ("assessments", {"load_pct": 100.0, "declared_load_pct": 0.0}),
        ("assessments", {"review_reasons": [reason.value for reason in ReviewReason]}),
        (
            "assessments",
            {"uncertainty": 0.0, "ood_score": 0.12, "brightness": 0.31, "sharpness": 0.78},
        ),
        ("model_versions", {"status": "active", "activated_at": T0}),
        ("confirmed_labels", {"load_pct": None, "unusable": True}),
        ("training_jobs", {"status": "running", "started_at": T0}),
        ("training_jobs", {"status": "succeeded", "started_at": T0, "finished_at": T0}),
        (
            "training_jobs",
            {
                "status": "failed",
                "started_at": T0,
                "finished_at": T0,
                "error_code": "STAGE_CHECK_FAILED",
            },
        ),
    ],
)
def test_accepts_boundary_row(seeded: Connection, table_name: str, overrides: Row) -> None:
    assert violated_constraint(seeded, table_name, ROW_BUILDERS[table_name](**overrides)) is None


@pytest.mark.integration
@pytest.mark.parametrize(
    ("table_name", "overrides", "constraint"),
    [
        ("images", {"sha256": "B" * 64}, "ck_images_sha256_format"),
        ("images", {"media_type": "image/heic"}, "ck_images_media_type_known"),
        ("images", {"width": 0}, "ck_images_size_positive"),
        ("images", {"size_bytes": 0}, "ck_images_bytes_positive"),
        ("shipments", {"id": "RP 2"}, "ck_shipments_id_format"),
        ("model_versions", {"version": ".hidden"}, "ck_model_versions_version_format"),
        ("model_versions", {"status": "deleted"}, "ck_model_versions_status_known"),
        ("model_versions", {"status": "active"}, "ck_model_versions_activated_at_matches_status"),
        ("model_versions", {"cv_mae": NAN}, "ck_model_versions_cv_mae_non_negative"),
        ("model_versions", {"cv_mae": INF}, "ck_model_versions_cv_mae_non_negative"),
        ("model_versions", {"within_10": 100.5}, "ck_model_versions_within_10_percent"),
        ("model_versions", {"within_10": NAN}, "ck_model_versions_within_10_percent"),
        ("assessments", {"load_pct": NAN}, "ck_assessments_load_pct_range"),
        ("assessments", {"load_pct": INF}, "ck_assessments_load_pct_range"),
        ("assessments", {"load_pct": -0.1}, "ck_assessments_load_pct_range"),
        ("assessments", {"load_pct": 100.1}, "ck_assessments_load_pct_range"),
        ("assessments", {"declared_load_pct": 100.1}, "ck_assessments_declared_load_pct_range"),
        ("assessments", {"uncertainty": -1.0}, "ck_assessments_uncertainty_non_negative"),
        ("assessments", {"ood_score": NAN}, "ck_assessments_ood_score_non_negative"),
        ("assessments", {"brightness": INF}, "ck_assessments_brightness_non_negative"),
        ("assessments", {"sharpness": NAN}, "ck_assessments_sharpness_non_negative"),
        ("assessments", {"review_reasons": ["looks_odd"]}, "ck_assessments_review_reasons_known"),
        ("assessments", {"processing_ms": -1}, "ck_assessments_processing_ms_non_negative"),
        ("assessments", {"shipment_id": "RP-404"}, "fk_assessments_shipment_id"),
        ("assessments", {"image_sha256": SHA_B}, "fk_assessments_image_sha256"),
        ("assessments", {"model_version": "v404"}, "fk_assessments_model_version"),
        ("confirmed_labels", {"load_pct": 100.5}, "ck_confirmed_labels_load_pct_range"),
        ("confirmed_labels", {"source": "email"}, "ck_confirmed_labels_source_known"),
        (
            "confirmed_labels",
            {"load_pct": None},
            "ck_confirmed_labels_load_pct_unless_unusable",
        ),
        (
            "confirmed_labels",
            {"unusable": True},
            "ck_confirmed_labels_load_pct_unless_unusable",
        ),
        (
            "confirmed_labels",
            {"updated_at": T0 - timedelta(seconds=1)},
            "ck_confirmed_labels_updated_after_created",
        ),
        ("confirmed_labels", {"image_sha256": SHA_B}, "fk_confirmed_labels_image_sha256"),
        ("training_jobs", {"mode": "slow"}, "ck_training_jobs_mode_known"),
        ("training_jobs", {"status": "paused", "started_at": T0}, "ck_training_jobs_status_known"),
        ("training_jobs", {"status": "running"}, "ck_training_jobs_started_unless_queued"),
        ("training_jobs", {"started_at": T0}, "ck_training_jobs_started_unless_queued"),
        (
            "training_jobs",
            {"status": "succeeded", "started_at": T0},
            "ck_training_jobs_finished_when_done",
        ),
        (
            "training_jobs",
            {"status": "failed", "started_at": T0, "finished_at": T0},
            "ck_training_jobs_error_when_failed",
        ),
        (
            "training_jobs",
            {
                "status": "succeeded",
                "started_at": T0,
                "finished_at": T0,
                "error_code": "INTERNAL_ERROR",
            },
            "ck_training_jobs_error_when_failed",
        ),
        ("training_jobs", {"base_version": "v404"}, "fk_training_jobs_base_version"),
        ("training_jobs", {"result_version": "v404"}, "fk_training_jobs_result_version"),
        ("operators", {"login": "Ivan"}, "ck_operators_login_format"),
        ("operators", {"login": ".hidden"}, "ck_operators_login_format"),
        ("operators", {"failed_sign_ins": -1}, "ck_operators_failed_sign_ins_non_negative"),
        ("operators", {"failed_sign_ins": 3}, "ck_operators_failure_window_matches_count"),
        ("operators", {"first_failure_at": T0}, "ck_operators_failure_window_matches_count"),
        ("operator_sessions", {"token_hash": "A" * 64}, "ck_operator_sessions_token_hash_format"),
        ("operator_sessions", {"expires_at": T0}, "ck_operator_sessions_expires_after_created"),
        ("operator_sessions", {"login": "nobody"}, "fk_operator_sessions_login"),
    ],
)
def test_rejects_invalid_row(
    seeded: Connection, table_name: str, overrides: Row, constraint: str
) -> None:
    row = ROW_BUILDERS[table_name](**overrides)
    assert violated_constraint(seeded, table_name, row) == constraint


@pytest.mark.integration
@pytest.mark.parametrize(
    ("table_name", "first", "second", "constraint"),
    [
        (
            "model_versions",
            {"version": "v2", "status": "active", "activated_at": T0},
            {"version": "v3", "status": "active", "activated_at": T0},
            "uq_model_versions_one_active",
        ),
        (
            "assessments",
            {"id": UUID(int=2)},
            {"id": UUID(int=3)},
            "uq_assessments_shipment_id_image_sha256",
        ),
        ("confirmed_labels", {}, {"load_pct": 70.0}, "pk_confirmed_labels"),
        (
            "training_jobs",
            {"id": UUID(int=2), "status": "running", "started_at": T0},
            {"id": UUID(int=3), "status": "running", "started_at": T0},
            "uq_training_jobs_one_running",
        ),
    ],
)
def test_rejects_a_second_row_where_one_is_allowed(
    seeded: Connection, table_name: str, first: Row, second: Row, constraint: str
) -> None:
    builder = ROW_BUILDERS[table_name]
    assert violated_constraint(seeded, table_name, builder(**first)) is None
    assert violated_constraint(seeded, table_name, builder(**second)) == constraint


@pytest.mark.integration
@pytest.mark.parametrize(
    ("table_name", "first", "second"),
    [
        (
            "model_versions",
            {"version": "v2", "status": "active", "activated_at": T0},
            {"version": "v3", "status": "retired", "activated_at": T0},
        ),
        ("training_jobs", {"id": UUID(int=2)}, {"id": UUID(int=3)}),
    ],
)
def test_accepts_rows_that_only_look_alike(
    seeded: Connection, table_name: str, first: Row, second: Row
) -> None:
    builder = ROW_BUILDERS[table_name]
    assert violated_constraint(seeded, table_name, builder(**first)) is None
    assert violated_constraint(seeded, table_name, builder(**second)) is None


@pytest.mark.integration
def test_same_photo_may_be_assessed_in_another_shipment(seeded: Connection) -> None:
    insert(seeded, "shipments", shipment_row(id="RP-2"))
    assert violated_constraint(seeded, "assessments", assessment_row(id=UUID(int=2))) is None
    second = assessment_row(id=UUID(int=3), shipment_id="RP-2")
    assert violated_constraint(seeded, "assessments", second) is None


@pytest.mark.integration
@pytest.mark.parametrize(
    "value", ["RP-2026-000123", "R" * 64, "a_b-C9", "R" * 65, "RP 1", "RP-1\n", "", "рп-1", "RP/1"]
)
def test_shipment_id_rule_matches_the_domain(seeded: Connection, value: str) -> None:
    try:
        ShipmentId(value)
    except LoadvisionError:
        domain_accepts = False
    else:
        domain_accepts = True
    row = shipment_row(id=value)
    assert (violated_constraint(seeded, "shipments", row) is None) == domain_accepts


@pytest.mark.integration
@pytest.mark.parametrize(
    "value", ["2026-09-20.1", "a", "A" * 64, "A" * 65, ".hidden", "-x", "v 1", "v1\n", ""]
)
def test_model_version_rule_matches_the_domain(seeded: Connection, value: str) -> None:
    try:
        ModelVersionId(value)
    except ValueError:
        domain_accepts = False
    else:
        domain_accepts = True
    row = model_row(version=value)
    assert (violated_constraint(seeded, "model_versions", row) is None) == domain_accepts


@pytest.mark.integration
@pytest.mark.parametrize("value", ["b" * 64, "B" * 64, "b" * 63, "b" * 65, "g" * 64])
def test_photo_hash_rule_matches_the_domain(seeded: Connection, value: str) -> None:
    domain_accepts = re.fullmatch(SHA256_HEX_PATTERN, value) is not None
    row = image_row(sha256=value)
    assert (violated_constraint(seeded, "images", row) is None) == domain_accepts


@pytest.mark.integration
@pytest.mark.parametrize("status", list(ModelVersionStatus))
@pytest.mark.parametrize("activated_at", [None, T0])
def test_activation_time_rule_matches_the_domain(
    seeded: Connection, status: ModelVersionStatus, activated_at: Any
) -> None:
    try:
        ModelVersion(
            version=ModelVersionId("v2"), status=status, created_at=T0, activated_at=activated_at
        )
    except ValueError:
        domain_accepts = False
    else:
        domain_accepts = True
    row = model_row(status=status.value, activated_at=activated_at)
    assert (violated_constraint(seeded, "model_versions", row) is None) == domain_accepts
