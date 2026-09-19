"""Deciding whether a freshly trained candidate may replace the active model version."""

import csv
from dataclasses import dataclass
from typing import Final

import numpy as np

from loadvision.ml.evaluation.metrics import paired_difference
from loadvision.ml.packaging.model_package import PACKAGE_OOF, ModelPackage, ModelPackageError

# Two versions are compared on the rows they share. A row keeps its fold forever (rule 2 of
# docs/plan/folds-json.md), so both were predicted by a model that never saw them, and the
# difference says something about the models rather than about the split.
_COLUMNS: Final = ("image_id", "fold", "label", "predicted")


@dataclass(frozen=True, slots=True, kw_only=True)
class GateRules:
    """When a candidate is allowed to replace the active version.

    Attributes:
        min_new_labels: Fewest confirmed labels worth retraining for. Below this the metric
            cannot move further than the noise, so activating would be a coin toss.
        tolerance: How much worse the candidate may be on the shared rows, in percentage
            points. The default is the measured noise of a split, 0.3: a difference smaller
            than that is not evidence of anything, and refusing it would block every retrain.
    """

    min_new_labels: int = 20
    tolerance: float = 0.3


DEFAULT_RULES: Final = GateRules()


@dataclass(frozen=True, slots=True, kw_only=True)
class GateReport:
    """Numbers behind the decision, ready for JSON, for the API and for a person.

    Attributes:
        passed: Whether the candidate may be activated.
        reason: Why, in one line.
        new_labels: Confirmed labels the candidate was trained with.
        common_rows: Photos both versions were measured on.
        candidate_mae: MAE of the candidate on those rows.
        active_mae: MAE of the active version on those rows, or None when there is none.
        difference: `candidate_mae` minus `active_mae`; above zero means worse.
        difference_2se: Two standard errors of that paired difference.
    """

    passed: bool
    reason: str
    new_labels: int
    common_rows: int = 0
    candidate_mae: float | None = None
    active_mae: float | None = None
    difference: float | None = None
    difference_2se: float | None = None

    def as_json(self) -> dict[str, object]:
        """Returns the report as plain JSON types, for `TrainingOutcome.gate_report`."""
        return {
            "passed": self.passed,
            "reason": self.reason,
            "new_labels": self.new_labels,
            "common_rows": self.common_rows,
            "candidate_mae": self.candidate_mae,
            "active_mae": self.active_mae,
            "difference": self.difference,
            "difference_2se": self.difference_2se,
        }


def read_package_oof(package: ModelPackage) -> dict[str, tuple[int, float, float]]:
    """Reads the out-of-fold predictions a package carries.

    Args:
        package: Verified model package.

    Returns:
        Photo id to its fold, its label and what the package predicted for it.

    Raises:
        ModelPackageError: If the package carries no predictions or they are malformed.
    """
    path = package.path(PACKAGE_OOF)
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            missing = sorted(set(_COLUMNS) - set(reader.fieldnames or ()))
            if missing:
                raise ModelPackageError(f"{PACKAGE_OOF}: missing {', '.join(missing)}")
            return {
                row["image_id"]: (int(row["fold"]), float(row["label"]), float(row["predicted"]))
                for row in reader
            }
    except (OSError, ValueError) as error:
        raise ModelPackageError(f"{PACKAGE_OOF}: {error}") from error


def decide(
    candidate: ModelPackage,
    active: ModelPackage | None,
    *,
    new_labels: int,
    rules: GateRules = DEFAULT_RULES,
) -> GateReport:
    """Compares a candidate with the active version on the photos they share.

    Only shared rows count. The candidate was trained on more photos, so its own metric is not
    comparable with the active one's; the rows both versions were measured on are, because a
    row never changes fold.

    Args:
        candidate: Freshly built package.
        active: Package the service answers with now, or None before the first activation.
        new_labels: How many confirmed labels went into the candidate.
        rules: Thresholds of the gate.

    Returns:
        The decision with the numbers behind it.

    Raises:
        ModelPackageError: If a package carries no usable out-of-fold predictions, or the two
            disagree about the fold of a shared photo, which would make the comparison a lie.
    """
    if active is None:
        return GateReport(
            passed=True,
            reason="активной версии ещё нет, кандидат становится первой",
            new_labels=new_labels,
        )
    if new_labels < rules.min_new_labels:
        return GateReport(
            passed=False,
            reason=(
                f"новых меток {new_labels}, нужно хотя бы {rules.min_new_labels}: "
                f"меньше не сдвинет метрику дальше шума"
            ),
            new_labels=new_labels,
        )

    theirs = read_package_oof(active)
    ours = read_package_oof(candidate)
    shared = sorted(set(theirs) & set(ours))
    if not shared:
        raise ModelPackageError("the two versions share no measured photo")
    moved = [image_id for image_id in shared if theirs[image_id][0] != ours[image_id][0]]
    if moved:
        raise ModelPackageError(
            f"{len(moved)} shared photos changed fold between the versions, "
            f"such as {moved[0]}; the comparison would not be out-of-fold"
        )

    labels = np.array([theirs[image_id][1] for image_id in shared])
    active_error = np.abs(labels - np.array([theirs[image_id][2] for image_id in shared]))
    candidate_error = np.abs(labels - np.array([ours[image_id][2] for image_id in shared]))
    mean, spread = paired_difference(candidate_error, active_error)
    passed = mean <= rules.tolerance
    return GateReport(
        passed=passed,
        reason=(
            f"кандидат {'не хуже' if passed else 'хуже'} активной на {len(shared)} общих фото: "
            f"{mean:+.3f} при допуске {rules.tolerance:+.2f}"
        ),
        new_labels=new_labels,
        common_rows=len(shared),
        candidate_mae=float(candidate_error.mean()),
        active_mae=float(active_error.mean()),
        difference=mean,
        difference_2se=spread,
    )
