import csv
from pathlib import Path

import pytest

from loadvision.ml.packaging.model_package import PACKAGE_OOF, ModelPackage, ModelPackageError
from loadvision.ml.training.gate import GateRules, decide, read_package_oof
from tests.model_packages import build_package

# (image_id, fold, label, predicted)
type Row = tuple[str, int, float, float]


def oof_bytes(rows: list[Row]) -> bytes:
    lines = ["image_id,fold,label,raw,predicted"]
    lines += [f"{name},{fold},{label:g},{value:g},{value:g}" for name, fold, label, value in rows]
    return ("\n".join(lines) + "\n").encode()


def package(root: Path, rows: list[Row], version: str = "v1") -> ModelPackage:
    return build_package(
        root,
        files={"head.npz": b"ridge head", PACKAGE_OOF: oof_bytes(rows)},
        version=version,
    )


def rows_off_by(error: float, count: int = 40, fold_shift: int = 0) -> list[Row]:
    """Rows whose prediction misses the label by exactly `error`, alternating in sign."""
    return [
        (f"img_{i:03d}", (i % 5) + fold_shift, 50.0, 50.0 + (error if i % 2 else -error))
        for i in range(count)
    ]


@pytest.mark.ml
def test_the_first_version_passes_without_anything_to_compare(tmp_path: Path) -> None:
    candidate = package(tmp_path / "c", rows_off_by(5.0))

    report = decide(candidate, None, new_labels=0)

    # Nothing to be worse than, and refusing here would leave the service with no model at all.
    assert report.passed
    assert report.active_mae is None


@pytest.mark.ml
def test_a_candidate_that_improves_is_let_through(tmp_path: Path) -> None:
    active = package(tmp_path / "a", rows_off_by(8.0))
    candidate = package(tmp_path / "c", rows_off_by(5.0), version="v2")

    report = decide(candidate, active, new_labels=30)

    assert report.passed
    assert (report.active_mae, report.candidate_mae) == (8.0, 5.0)
    assert report.difference == pytest.approx(-3.0)
    assert report.common_rows == 40


@pytest.mark.ml
def test_a_candidate_that_is_clearly_worse_is_refused(tmp_path: Path) -> None:
    active = package(tmp_path / "a", rows_off_by(5.0))
    candidate = package(tmp_path / "c", rows_off_by(9.0), version="v2")

    report = decide(candidate, active, new_labels=30)

    assert not report.passed
    assert report.difference == pytest.approx(4.0)
    assert "хуже" in report.reason


@pytest.mark.ml
def test_a_candidate_worse_only_within_the_split_noise_is_let_through(tmp_path: Path) -> None:
    active = package(tmp_path / "a", rows_off_by(5.0))
    candidate = package(tmp_path / "c", rows_off_by(5.2), version="v2")

    report = decide(candidate, active, new_labels=30)

    # 0.2 is below the 0.3 tolerance: refusing noise would block every retrain forever.
    assert report.passed
    assert report.difference == pytest.approx(0.2)


@pytest.mark.ml
def test_too_few_new_labels_stops_the_candidate_before_any_comparison(tmp_path: Path) -> None:
    active = package(tmp_path / "a", rows_off_by(9.0))
    candidate = package(tmp_path / "c", rows_off_by(1.0), version="v2")

    report = decide(candidate, active, new_labels=3)

    # Even a candidate that looks much better: three labels cannot move 716 rows honestly.
    assert not report.passed
    assert report.candidate_mae is None
    assert "новых меток 3" in report.reason


@pytest.mark.ml
def test_the_label_threshold_is_configurable(tmp_path: Path) -> None:
    active = package(tmp_path / "a", rows_off_by(8.0))
    candidate = package(tmp_path / "c", rows_off_by(5.0), version="v2")

    report = decide(candidate, active, new_labels=5, rules=GateRules(min_new_labels=5))

    assert report.passed


@pytest.mark.ml
def test_only_the_shared_rows_are_compared(tmp_path: Path) -> None:
    active = package(tmp_path / "a", rows_off_by(6.0, count=20))
    fresh = rows_off_by(6.0, count=20) + [(f"img_new_{i}", i % 5, 50.0, 50.0) for i in range(10)]
    candidate = package(tmp_path / "c", fresh, version="v2")

    report = decide(candidate, active, new_labels=30)

    # The candidate saw ten photos the active version never did; they must not flatter it.
    assert report.common_rows == 20
    assert report.candidate_mae == pytest.approx(6.0)


@pytest.mark.ml
def test_a_row_that_changed_fold_makes_the_comparison_a_lie(tmp_path: Path) -> None:
    active = package(tmp_path / "a", rows_off_by(6.0))
    candidate = package(tmp_path / "c", rows_off_by(6.0, fold_shift=1), version="v2")

    with pytest.raises(ModelPackageError, match="changed fold"):
        decide(candidate, active, new_labels=30)


@pytest.mark.ml
def test_two_versions_with_no_photo_in_common_are_refused(tmp_path: Path) -> None:
    active = package(tmp_path / "a", [("img_old", 0, 50.0, 50.0), ("img_old2", 1, 50.0, 50.0)])
    candidate = package(
        tmp_path / "c", [("img_new", 0, 50.0, 50.0), ("img_new2", 1, 50.0, 50.0)], version="v2"
    )

    with pytest.raises(ModelPackageError, match="share no measured photo"):
        decide(candidate, active, new_labels=30)


@pytest.mark.ml
def test_a_package_without_predictions_is_refused(tmp_path: Path) -> None:
    broken = build_package(tmp_path / "b", files={"head.npz": b"ridge head"})

    with pytest.raises(ModelPackageError, match="not listed"):
        read_package_oof(broken)


@pytest.mark.ml
def test_predictions_missing_a_column_are_refused(tmp_path: Path) -> None:
    broken = build_package(
        tmp_path / "b",
        files={"head.npz": b"ridge head", PACKAGE_OOF: b"image_id,predicted\nimg_0,50\n"},
    )

    with pytest.raises(ModelPackageError, match="missing fold, label"):
        read_package_oof(broken)


@pytest.mark.ml
def test_the_report_survives_a_trip_through_json(tmp_path: Path) -> None:
    active = package(tmp_path / "a", rows_off_by(8.0))
    candidate = package(tmp_path / "c", rows_off_by(5.0), version="v2")

    report = decide(candidate, active, new_labels=30).as_json()

    # The worker stores this in the job and the API hands it to a person.
    assert set(report) == {
        "passed",
        "reason",
        "new_labels",
        "common_rows",
        "candidate_mae",
        "active_mae",
        "difference",
        "difference_2se",
    }
    assert report["passed"] is True


@pytest.mark.ml
def test_the_package_built_by_the_pipeline_carries_readable_predictions(tmp_path: Path) -> None:
    rows = rows_off_by(4.0, count=10)
    built = package(tmp_path / "p", rows)

    stored = read_package_oof(built)

    with (built.root / PACKAGE_OOF).open(newline="", encoding="utf-8") as file:
        written = list(csv.DictReader(file))
    assert len(stored) == len(written) == 10
    assert stored["img_000"] == (0, 50.0, 46.0)
