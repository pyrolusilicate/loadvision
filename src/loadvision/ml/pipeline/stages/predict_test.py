"""Stage that fits the shipped heads on every labelled photo and writes `submission.csv`."""

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.data.manifest import load_split
from loadvision.ml.models.backbones import Backbone, load_backbone
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.ml.pipeline.stages.extract_embeddings import EMBEDDINGS_DIR
from loadvision.ml.pipeline.stages.train_folds import OOF_FILE, Chain, Member
from loadvision.ml.training.embedding_cache import embed_samples
from loadvision.ml.training.head_trainer import FittedHead
from loadvision.ml.training.postprocess_fit import fit_chain

SUBMISSION_FILE: Final = "submission.csv"

_MIN_LOAD: Final = 0.0
_MAX_LOAD: Final = 100.0
_DECIMALS: Final = 1


@dataclass(frozen=True, slots=True)
class PredictTest:
    """Predicts the test split with heads fitted on all labelled photos.

    The quality of this file is the number `evaluate` reported, not anything measured here:
    the test labels are hidden. The heads differ from the fold heads only in seeing more
    photos, and the postprocessing chain is fitted on the out-of-fold predictions, which is
    the only place the hedging of a head is visible without cheating.

    Attributes:
        members: Encoder and ridge penalty of every ensemble member.
        chain: Postprocessing steps, fitted on `oof.csv`.
        device: Torch device the encoders run on.
        load: How an encoder is built from its registry key; replaced in tests by a
            stand-in that needs no download.
    """

    members: Sequence[Member]
    chain: Chain = ()
    device: str = "cpu"
    load: Callable[[str, str], Backbone] = load_backbone

    @property
    def name(self) -> str:
        """Stage name."""
        return "predict_test"

    @property
    def version(self) -> str:
        """Stage version."""
        return "1"

    def inputs(self, ctx: RunContext) -> Mapping[str, str]:
        """What the result depends on."""
        return {
            "members": ";".join(f"{name}:{alpha:g}" for name, alpha in self.members),
            "chain": ";".join(name for name, _ in self.chain),
            "oof": str(ctx.run_dir / OOF_FILE),
        }

    def run(self, ctx: RunContext) -> StageResult:
        """Fits the shipped heads and writes the submission.

        Raises:
            FileNotFoundError: If the training stage left no out-of-fold predictions.
        """
        train = load_split(ctx.data_dir, "train")
        test = load_split(ctx.data_dir, "test")
        labels = np.array([sample.label for sample in train], dtype=np.float64)
        cache_dir = ctx.cache_dir / EMBEDDINGS_DIR

        parts: list[NDArray[np.float64]] = []
        for encoder, alpha in self.members:
            backbone = self.load(encoder, self.device)
            train_features = embed_samples(
                train, backbone, cache_dir=cache_dir, device=self.device
            ).astype(np.float64)
            test_features = embed_samples(
                test, backbone, cache_dir=cache_dir, device=self.device
            ).astype(np.float64)
            head = FittedHead.fit(train_features, labels, alpha=alpha)
            parts.append(head.predict(test_features))

        predictions = np.clip(np.mean(parts, axis=0), _MIN_LOAD, _MAX_LOAD)
        if self.chain:
            raw, reference = _read_oof(ctx.run_dir / OOF_FILE)
            predictions = fit_chain(self.chain, raw, reference)(predictions)

        path = ctx.run_dir / SUBMISSION_FILE
        _write(path, [sample.image_id for sample in test], predictions)
        return StageResult(
            artifacts={"submission": path},
            counts={"test": len(test), "members": len(self.members)},
            metrics={
                "pred_min": float(predictions.min()),
                "pred_max": float(predictions.max()),
                "pred_mean": float(predictions.mean()),
            },
        )

    def check(self, ctx: RunContext, result: StageResult) -> None:
        """Verifies the submission matches what the platform accepts.

        Raises:
            StageCheckError: If the file is missing, the ids differ from the test split, or a
                value is not a finite load.
        """
        path = result.artifacts.get("submission")
        if path is None or not path.is_file():
            raise StageCheckError(f"{self.name}: submission was not written")
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["image_id", "load_pct"]:
                raise StageCheckError(f"{self.name}: header is {reader.fieldnames}")
            rows = list(reader)
        expected = [sample.image_id for sample in load_split(ctx.data_dir, "test")]
        if [row["image_id"] for row in rows] != expected:
            raise StageCheckError(
                f"{self.name}: {len(rows)} rows do not match the {len(expected)} test photos"
            )
        values = np.array([float(row["load_pct"]) for row in rows])
        if not np.isfinite(values).all() or values.min() < _MIN_LOAD or values.max() > _MAX_LOAD:
            raise StageCheckError(f"{self.name}: values outside [0, 100] or not finite")


def _read_oof(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; train_folds did not run")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    raw = np.array([float(row["raw"]) for row in rows])
    labels = np.array([float(row["label"]) for row in rows])
    return raw, labels


def _write(path: Path, image_ids: Sequence[str], predictions: NDArray[np.float64]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", "load_pct"))
        for image_id, value in zip(image_ids, predictions, strict=True):
            writer.writerow([image_id, f"{float(value):.{_DECIMALS}f}"])
