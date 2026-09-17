"""Stage that fits one head per fold and writes out-of-fold predictions."""

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.data.manifest import Sample, load_split
from loadvision.ml.data.splits import check_covers, read_folds
from loadvision.ml.evaluation.cross_validation import out_of_fold
from loadvision.ml.models.backbones import Backbone, load_backbone
from loadvision.ml.packaging.model_package import file_sha256
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.ml.pipeline.stages.extract_embeddings import EMBEDDINGS_DIR
from loadvision.ml.training.embedding_cache import embed_samples
from loadvision.ml.training.head_trainer import FittedHead
from loadvision.ml.training.postprocess_fit import fit_chain

OOF_FILE: Final = "oof.csv"

_COLUMNS: Final = ("image_id", "fold", "label", "raw", "predicted")
_MIN_LOAD: Final = 0.0
_MAX_LOAD: Final = 100.0

type Member = tuple[str, float]
type Chain = Sequence[tuple[str, Mapping[str, float]]]


@dataclass(frozen=True, slots=True)
class TrainFolds:
    """Fits every member on four folds, predicts the fifth and postprocesses the result.

    This is the only honest measurement of the whole chain: a photo is scored by weights, by a
    calibration curve and by thresholds that were all fitted without it, and without any photo
    of its truck, because folds keep whole scenes together.

    Attributes:
        members: Encoder and ridge penalty of every ensemble member.
        chain: Postprocessing steps applied after averaging the members.
        folds_file: Canonical folds file.
        device: Torch device the encoders run on.
        load: How an encoder is built from its registry key; replaced in tests by a
            stand-in that needs no download.
    """

    members: Sequence[Member]
    chain: Chain = ()
    folds_file: Path = Path("configs/splits/folds.json")
    device: str = "cpu"
    load: Callable[[str, str], Backbone] = load_backbone

    @property
    def name(self) -> str:
        """Stage name."""
        return "train_folds"

    @property
    def version(self) -> str:
        """Stage version."""
        return "1"

    def inputs(self, ctx: RunContext) -> Mapping[str, str]:
        """What the result depends on."""
        return {
            "members": ";".join(f"{name}:{alpha:g}" for name, alpha in self.members),
            "chain": ";".join(name for name, _ in self.chain),
            "folds": file_sha256(self.folds_file) if self.folds_file.is_file() else "missing",
            "seed": str(ctx.seed),
        }

    def run(self, ctx: RunContext) -> StageResult:
        """Computes out-of-fold predictions and writes `oof.csv`."""
        samples = load_split(ctx.data_dir, "train")
        folds = read_folds(self.folds_file)
        check_covers(folds, samples)
        numbers = folds.fold_indices(samples)
        labels = np.array([sample.label for sample in samples], dtype=np.float64)

        parts: list[NDArray[np.float64]] = []
        metrics: dict[str, float] = {}
        for encoder, alpha in self.members:
            backbone = self.load(encoder, self.device)
            features = embed_samples(
                samples, backbone, cache_dir=ctx.cache_dir / EMBEDDINGS_DIR, device=self.device
            ).astype(np.float64)
            result = out_of_fold(
                features,
                labels,
                numbers,
                lambda a, b, penalty=alpha: FittedHead.fit(a, b, alpha=penalty).predict,
            )
            metrics[f"mae_{encoder}"] = result.metrics["mae"]
            parts.append(result.predictions)

        raw = np.clip(np.mean(parts, axis=0), _MIN_LOAD, _MAX_LOAD)
        predicted = self._apply_chain(raw, labels, numbers)
        path = ctx.run_dir / OOF_FILE
        _write(path, samples, numbers, labels, raw, predicted)
        return StageResult(
            artifacts={"oof": path},
            counts={"photos": len(samples), "members": len(self.members), "folds": folds.n_folds},
            metrics=metrics,
        )

    def check(self, ctx: RunContext, result: StageResult) -> None:
        """Verifies every training photo is predicted exactly once, inside the allowed range.

        Raises:
            StageCheckError: If the file is missing, a photo is missing or repeated, or a
                prediction is not a finite load.
        """
        path = result.artifacts.get("oof")
        if path is None or not path.is_file():
            raise StageCheckError(f"{self.name}: out-of-fold predictions were not written")
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        expected = {sample.image_id for sample in load_split(ctx.data_dir, "train")}
        seen = [row["image_id"] for row in rows]
        if sorted(seen) != sorted(expected):
            raise StageCheckError(
                f"{self.name}: {len(seen)} predictions for {len(expected)} training photos"
            )
        values = np.array([float(row["predicted"]) for row in rows])
        if not np.isfinite(values).all() or values.min() < _MIN_LOAD or values.max() > _MAX_LOAD:
            raise StageCheckError(f"{self.name}: predictions outside [0, 100] or not finite")

    def _apply_chain(
        self, raw: NDArray[np.float64], labels: NDArray[np.float64], numbers: Sequence[int]
    ) -> NDArray[np.float64]:
        if not self.chain:
            return raw
        # The chain of every fold is fitted on the other four, so a threshold or a curve never
        # sees the photos it is judged on.
        corrected = np.empty_like(raw)
        for fold in sorted(set(numbers)):
            validation = np.array([value == fold for value in numbers])
            apply = fit_chain(self.chain, raw[~validation], labels[~validation])
            corrected[validation] = apply(raw[validation])
        return corrected


def _write(
    path: Path,
    samples: Sequence[Sample],
    numbers: Sequence[int],
    labels: NDArray[np.float64],
    raw: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(_COLUMNS)
        for sample, fold, label, before, after in zip(
            samples, numbers, labels, raw, predicted, strict=True
        ):
            writer.writerow(
                [
                    sample.image_id,
                    fold,
                    f"{label:g}",
                    f"{before:.4f}",
                    f"{after:.4f}",
                ]
            )
