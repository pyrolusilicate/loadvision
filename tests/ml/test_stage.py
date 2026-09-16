import pickle
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from loadvision.errors import ERROR_CATALOG, ErrorCode, LoadvisionError
from loadvision.ml.pipeline.stage import (
    RUN_ID_PATTERN,
    RunContext,
    StageCheckError,
    StageResult,
    new_run_id,
)
from tests.builders import T0


def make_ctx(tmp_path: Path, run_id: str = "run_20260916_120000_a3f") -> RunContext:
    return RunContext(
        run_id=run_id,
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        config={},
        seed=42,
    )


@pytest.mark.ml
@pytest.mark.parametrize("run_id", ["run_20260916_120000_a3f", "a", "exp.v2-1", "a" * 100])
def test_run_context_accepts_folder_safe_run_id(tmp_path: Path, run_id: str) -> None:
    assert make_ctx(tmp_path, run_id).run_dir == tmp_path / "runs" / run_id


@pytest.mark.ml
@pytest.mark.parametrize(
    "run_id", ["", ".hidden", "..", "-flag", "a/b", "a\\b", "a b", "запуск", "a" * 101, "run\n"]
)
def test_run_context_rejects_unsafe_run_id(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run id"):
        make_ctx(tmp_path, run_id)


@pytest.mark.ml
def test_new_run_id_is_utc_start_time_and_random_tail() -> None:
    moscow = T0.astimezone(timezone(timedelta(hours=3)))
    run_id = new_run_id(moscow)
    assert re.fullmatch(r"run_20260916_120000_[0-9a-f]{6}", run_id)
    assert re.fullmatch(RUN_ID_PATTERN, run_id)


@pytest.mark.ml
def test_new_run_ids_differ_within_one_second() -> None:
    assert len({new_run_id(T0) for _ in range(20)}) == 20


@pytest.mark.ml
def test_new_run_id_rejects_naive_time() -> None:
    with pytest.raises(ValueError, match="now"):
        new_run_id(datetime(2026, 9, 16, 12, 0))


@pytest.mark.ml
def test_stage_result_is_empty_by_default() -> None:
    result = StageResult()
    assert (result.artifacts, result.metrics, result.counts, result.cached) == ({}, {}, {}, False)


@pytest.mark.ml
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_stage_result_rejects_non_finite_metric(value: float) -> None:
    with pytest.raises(ValueError, match="mae"):
        StageResult(metrics={"within_10": 0.8, "mae": value})


@pytest.mark.ml
def test_stage_check_error_carries_catalog_code() -> None:
    error = StageCheckError("oof covers 1195 of 1200 rows")
    assert isinstance(error, LoadvisionError)
    assert error.code is ErrorCode.STAGE_CHECK_FAILED
    assert error.spec is ERROR_CATALOG[ErrorCode.STAGE_CHECK_FAILED]
    assert str(error) == "STAGE_CHECK_FAILED: oof covers 1195 of 1200 rows"


@pytest.mark.ml
def test_stage_check_error_survives_pickling_between_processes() -> None:
    restored = pickle.loads(pickle.dumps(StageCheckError("oof covers 1195 of 1200 rows")))
    assert type(restored) is StageCheckError
    assert restored.code is ErrorCode.STAGE_CHECK_FAILED
    assert restored.detail == "oof covers 1195 of 1200 rows"
