"""`TrainingRunner` that retrains by running the very pipeline research runs."""

import hashlib
import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from loadvision.application.ports import (
    ImageStorage,
    TrainingMode,
    TrainingOutcome,
    TrainingRequest,
    TrainingStatus,
)
from loadvision.errors import ErrorCode
from loadvision.ml.data.confirmed import ImportReport, build_dataset
from loadvision.ml.data.manifest import load_split
from loadvision.ml.data.splits import check_unchanged, extend_folds, read_folds, write_folds
from loadvision.ml.packaging.model_package import ModelPackage, ModelPackageError
from loadvision.ml.pipeline.runner import PipelineRunner, RunSummary
from loadvision.ml.pipeline.stage import RunContext, new_run_id
from loadvision.ml.pipeline.stages import build_pipeline
from loadvision.ml.pipeline.stages.build_package import PACKAGE_DIR
from loadvision.ml.training.gate import DEFAULT_RULES, GateReport, GateRules, decide

logger = logging.getLogger(__name__)

DATASETS_DIR: Final = "datasets"

_LABEL_SET_CHARS: Final = 16


class PipelineTrainingRunner:
    """Retrains on confirmed labels and compares the result with the active version.

    The worker and the CLI call the same object, and it runs the same stages a research run
    does. That is the point: whatever is measured in research is what the service retrains
    with, rather than a second implementation that drifts.

    One run does four things. It builds a dataset that is the base one plus the confirmed
    labels; it gives the new photos a fold without moving a single stored row; it runs the
    pipeline, which ends in a model package; and it compares that package with the active one
    on the photos both were measured on.

    Args:
        config: Experiment config with `members` and, optionally, `postprocess`.
        data_dir: Base dataset root with `train/` and `test/`.
        runs_dir: Folder that holds run folders.
        cache_dir: Cache kept between runs: embeddings and the built datasets.
        storage: Where the bytes of confirmed photos live.
        folds_file: Canonical folds file. New rows are appended to it, stored rows never move.
        active_package: Folder of the package the service answers with, or None before the
            first activation.
        clock: Returns the current timezone-aware time.
        device: Torch device the encoders run on.
        rules: Thresholds of the promotion gate.
        seed: Seed recorded in the run context.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        data_dir: Path,
        runs_dir: Path,
        cache_dir: Path,
        storage: ImageStorage,
        clock: Callable[[], datetime],
        folds_file: Path = Path("configs/splits/folds.json"),
        active_package: Path | None = None,
        device: str = "cpu",
        rules: GateRules = DEFAULT_RULES,
        seed: int = 0,
    ) -> None:
        self._config = dict(config)
        self._data_dir = data_dir
        self._runs_dir = runs_dir
        self._cache_dir = cache_dir
        self._storage = storage
        self._clock = clock
        self._folds_file = folds_file
        self._active_package = active_package
        self._device = device
        self._rules = rules
        self._seed = seed

    def run(self, request: TrainingRequest) -> TrainingOutcome:
        """Trains a candidate and reports whether it may replace the active version.

        Expected failures come back inside the outcome, never as an exception: the worker has
        to record them in the job and go on serving with the model it already has.

        Args:
            request: What to train and what to compare against.

        Returns:
            Outcome with the run id, the package and the gate decision.
        """
        run_id = new_run_id(self._clock())
        if request.mode is not TrainingMode.FAST:
            return _failed(
                run_id,
                ErrorCode.INTERNAL_ERROR,
                f"mode {request.mode} fine-tunes the encoder; only {TrainingMode.FAST} is built",
            )
        try:
            dataset, report = self._dataset(request)
            self._extend_folds(dataset)
        except (OSError, ValueError) as error:
            return _failed(run_id, ErrorCode.INTERNAL_ERROR, f"{type(error).__name__}: {error}")

        summary = self._pipeline(run_id, dataset)
        failed = summary.failed_stage
        if failed is not None:
            return _failed(
                run_id,
                failed.error_code or ErrorCode.INTERNAL_ERROR,
                f"stage {failed.name}: {failed.error_detail or failed.status.value}",
            )

        package_path = self._runs_dir / run_id / PACKAGE_DIR
        try:
            gate = self._gate(package_path, report)
        except ModelPackageError as error:
            return _failed(run_id, error.code, error.detail or "model package is invalid")

        logger.info("run %s: %s", run_id, gate.reason)
        return TrainingOutcome(
            run_id=run_id,
            status=TrainingStatus.SUCCEEDED,
            package_path=package_path,
            gate_passed=gate.passed,
            gate_report={**gate.as_json(), **report.as_counts()},
        )

    def _dataset(self, request: TrainingRequest) -> tuple[Path, ImportReport]:
        """Builds the dataset of this label set, reusing it when the labels repeat."""
        digest = hashlib.sha256(
            "\n".join(
                sorted(f"{item.image_sha256}:{item.load_pct.value:g}" for item in request.labels)
            ).encode()
        ).hexdigest()[:_LABEL_SET_CHARS]
        target = self._cache_dir / DATASETS_DIR / digest
        report = build_dataset(self._data_dir, target, request.labels, self._storage)
        logger.info("dataset %s: %s", digest, report.as_counts())
        return target, report

    def _extend_folds(self, dataset: Path) -> None:
        """Appends the new photos to the canonical folds file, leaving stored rows alone."""
        folds = read_folds(self._folds_file)
        extended = extend_folds(folds, load_split(dataset, "train"))
        check_unchanged(folds, extended)
        if extended.as_dict() != folds.as_dict():
            write_folds(self._folds_file, extended)

    def _pipeline(self, run_id: str, dataset: Path) -> RunSummary:
        config = {**self._config, "folds": str(self._folds_file)}
        ctx = RunContext(
            run_id=run_id,
            runs_dir=self._runs_dir,
            cache_dir=self._cache_dir,
            data_dir=dataset,
            config=config,
            seed=self._seed,
        )
        stages = build_pipeline(config, device=self._device)
        summary = PipelineRunner(clock=self._clock).run(stages, ctx)
        logger.info("run %s finished: %s", run_id, summary.status.value)
        return summary

    def _gate(self, package_path: Path, report: ImportReport) -> GateReport:
        candidate = ModelPackage.load(package_path)
        active = None if self._active_package is None else ModelPackage.load(self._active_package)
        return decide(
            candidate,
            active,
            new_labels=report.added + report.relabelled,
            rules=self._rules,
        )


def _failed(run_id: str, code: ErrorCode, detail: str) -> TrainingOutcome:
    logger.warning("run %s failed: %s", run_id, detail)
    return TrainingOutcome(
        run_id=run_id,
        status=TrainingStatus.FAILED,
        error_code=code,
        error_detail=detail,
    )
