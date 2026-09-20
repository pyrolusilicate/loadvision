"""Stage that packs the trained ensemble into the model package the service loads."""

import csv
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
from numpy.typing import NDArray

from loadvision.domain.timestamps import format_utc
from loadvision.ml.data.manifest import load_split
from loadvision.ml.evaluation.metrics import mae_by_range, summary
from loadvision.ml.inference.ood import OodReference
from loadvision.ml.models.backbones import Backbone, load_backbone
from loadvision.ml.packaging.checkpoints import place_checkpoint
from loadvision.ml.packaging.manifest import (
    CvMetrics,
    FoldsRef,
    Manifest,
    Member,
    PostprocessStep,
    Preprocessing,
    Runtime,
)
from loadvision.ml.packaging.model_package import (
    PACKAGE_OOF,
    ModelPackage,
    file_sha256,
    hash_package_files,
    write_package,
)
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.ml.pipeline.stages.build_manifest import MANIFEST_FILE as DATASET_TABLE
from loadvision.ml.pipeline.stages.extract_embeddings import EMBEDDINGS_DIR
from loadvision.ml.pipeline.stages.train_folds import OOF_FILE, Chain, read_facts
from loadvision.ml.pipeline.stages.train_folds import Member as MemberSpec
from loadvision.ml.training.embedding_cache import embed_samples
from loadvision.ml.training.head_trainer import fit_head
from loadvision.ml.training.postprocess_fit import FittedChain, fit_chain
from loadvision.settings import GIT_COMMIT_ENV

PACKAGE_DIR: Final = "package"

LICENSES_DIR: Final = Path("licenses")

_NOTICE_FILE: Final = "NOTICE.md"

# Averaging ten neighbours: one nearest photo is noisy, a hundred washes the score out.
# Where the unusual-scene threshold sits among the training scores. Measured on v2 and 1023
# dataset photos: at 99 the flag fired on 13 ordinary photos of trucks, at 99.9 on one, while
# the frames a person calls broken score three times the threshold and are flagged either way.
OOD_PERCENTILE: Final = 99.9
_OOD_NEIGHBOURS: Final = 10

_PERCENT: Final = 100.0


@dataclass(frozen=True, slots=True)
class BuildPackage:
    """Fits the shipped heads and the chain on every labelled photo and writes the package.

    The package is what the service loads, so building it inside the run is what makes the
    submission and the service answer with the same model rather than with two models that
    happen to share a config.

    Attributes:
        members: Encoder and head of every ensemble member.
        chain: Postprocessing steps, fitted on the out-of-fold predictions.
        ood_percentile: Where the unusual-scene threshold sits in the distribution of training
            scores; 99.9 means one training photo in a thousand would be flagged.
        package_version: Package version in the `ModelVersionId` format.
        now: Build time; the current time when omitted.
        folds_file: Canonical folds file, recorded next to the CV metrics.
        licenses_dir: Folder holding one text per SPDX identifier the registry names.
        device: Torch device the encoders run on while fitting.
    """

    members: Sequence[MemberSpec]
    chain: Chain = ()
    package_version: str = "v1"
    folds_file: Path = Path("configs/splits/folds.json")
    licenses_dir: Path = LICENSES_DIR
    device: str = "cpu"
    now: datetime | None = None
    ood_percentile: float = OOD_PERCENTILE

    @property
    def name(self) -> str:
        """Stage name."""
        return "build_package"

    @property
    def version(self) -> str:
        """Version of the stage logic; the package has its own `package_version`."""
        return "1"

    def inputs(self, ctx: RunContext) -> Mapping[str, str]:
        """What the result depends on."""
        return {
            "members": ";".join(member.describe() for member in self.members),
            "chain": ";".join(name for name, _ in self.chain),
            "oof": str(ctx.run_dir / OOF_FILE),
            "package_version": self.package_version,
        }

    def run(self, ctx: RunContext) -> StageResult:
        """Writes `package/` with the heads, the curves, the licences and `manifest.json`.

        Raises:
            FileNotFoundError: If the training stage left no out-of-fold predictions, or a
                checkpoint names a licence with no text in `licenses_dir`.
        """
        # Read first: a package without the out-of-fold predictions would carry no honest
        # metrics, and finding that out after loading four encoders wastes a minute.
        raw = _read_raw(ctx.run_dir / OOF_FILE)
        train = load_split(ctx.data_dir, "train")
        labels = np.array([sample.label for sample in train], dtype=np.float64)
        root = ctx.run_dir / PACKAGE_DIR
        root.mkdir(parents=True, exist_ok=True)

        members: list[Member] = []
        loaded: list[Backbone] = []
        weight = 1.0 / len(self.members)
        weights: dict[str, Path] = {}
        pooled: tuple[int, str, NDArray[np.float64]] | None = None
        for index, spec in enumerate(self.members):
            backbone = load_backbone(spec.encoder, self.device)
            loaded.append(backbone)
            features = embed_samples(
                train, backbone, cache_dir=ctx.cache_dir / EMBEDDINGS_DIR, device=self.device
            ).astype(np.float64)
            head_path = root / f"m{index}" / "head.npz"
            fit_head(spec.head, features, labels, spec.params).save(head_path)
            # Members that name the same checkpoint share its file, so an ensemble of four
            # carries two sets of weights rather than four copies of the same bytes.
            checkpoint = weights.get(backbone.spec.timm_name) or place_checkpoint(
                root, backbone.spec.timm_name
            )
            weights[backbone.spec.timm_name] = checkpoint
            if pooled is None and backbone.spec.pooling == "pooled":
                # The reference holds the embeddings of one ordinary member: a grid member
                # would store twelve times the numbers for the same answer.
                pooled = (index, spec.encoder, features)
            members.append(
                _member(
                    backbone,
                    head_path.relative_to(root),
                    checkpoint.relative_to(root),
                    weight,
                )
            )

        # The package carries the predictions it was measured by, so the gate can compare it
        # with another version without refitting anything.
        oof = root / PACKAGE_OOF
        oof.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ctx.run_dir / OOF_FILE, oof)

        steps = self._fitted_steps(ctx, root, train, labels, raw)
        ood = self._ood(root, pooled)
        licenses = self._licenses(root, loaded)
        manifest = Manifest.model_validate(
            {
                "schema_version": 1,
                "version": self.package_version,
                "created_at": format_utc(self.now or datetime.now(UTC)),
                "git_commit": _git_commit(),
                "data_hash": _data_hash(ctx),
                "members": [member.model_dump(mode="json") for member in members],
                "postprocess": [step.model_dump(mode="json") for step in steps],
                "licenses": licenses,
                **ood,
                **self._cv(ctx),
                "files": hash_package_files(root),
            }
        )
        write_package(root, manifest)
        return StageResult(
            artifacts={"package": root, "manifest": root / "manifest.json"},
            counts={
                "members": len(members),
                "steps": len(steps),
                "files": len(manifest.files),
                "ood_reference": 0 if not ood else len(train),
                "licenses": len(licenses),
            },
            metrics={"weight_sum": sum(member.weight for member in members)},
        )

    def check(self, ctx: RunContext, result: StageResult) -> None:
        """Loads the package the way the service does and predicts with it.

        Raises:
            StageCheckError: If the package cannot be loaded or answers outside [0, 100].
        """
        root = result.artifacts.get("package")
        if root is None or not (root / "manifest.json").is_file():
            raise StageCheckError(f"{self.name}: package was not written")
        # Imported here so the stage does not drag the inference module into every run.
        from loadvision.ml.inference.predictor import PackagePredictor
        from loadvision.ml.training.embedding_cache import read_image

        try:
            predictor = PackagePredictor(ModelPackage.load(root), device=self.device)
        except Exception as error:
            raise StageCheckError(f"{self.name}: package does not load: {error}") from error
        photos = [read_image(sample.path) for sample in load_split(ctx.data_dir, "test")[:2]]
        values = [prediction.load_pct.value for prediction in predictor.predict(photos)]
        if not values or not all(0.0 <= value <= _PERCENT for value in values):
            raise StageCheckError(f"{self.name}: package predicted {values}")

    def _ood(
        self, root: Path, pooled: tuple[int, str, NDArray[np.float64]] | None
    ) -> dict[str, Any]:
        if pooled is None:
            return {}
        index, backbone, embeddings = pooled
        return write_ood(root, index, backbone, embeddings, self.ood_percentile)

    def _licenses(self, root: Path, backbones: Sequence[Backbone]) -> list[str]:
        return write_licenses(root, backbones, self.licenses_dir)

    def _fitted_steps(
        self,
        ctx: RunContext,
        root: Path,
        train: Sequence[Any],
        labels: NDArray[np.float64],
        raw: NDArray[np.float64],
    ) -> list[PostprocessStep]:
        if not self.chain:
            return []
        return write_steps(root, fit_chain(self.chain, raw, labels, read_facts(ctx, train)))

    def _cv(self, ctx: RunContext) -> dict[str, Any]:
        return cv_block(ctx.run_dir / OOF_FILE, self.folds_file)


def _member(backbone: Backbone, head: Path, weights: Path, weight: float) -> Member:
    return Member.model_validate(
        {
            "backbone": backbone.name,
            "runtime": Runtime.TORCH.value,
            "head": str(head),
            "weights": str(weights),
            "weight": weight,
            "preprocessing": Preprocessing.model_validate(
                {"name": backbone.preprocessing, "params": dict(backbone.params)}
            ).model_dump(mode="json"),
        }
    )


def _read_raw(path: Path) -> NDArray[np.float64]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; train_folds did not run")
    with path.open(newline="", encoding="utf-8") as file:
        return np.array([float(row["raw"]) for row in csv.DictReader(file)])


def _git_commit() -> str:
    # An image has no checkout and no git; its build records the commit in the environment.
    built_from = os.environ.get(GIT_COMMIT_ENV)
    if built_from:
        return built_from
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return "0" * 40
    return result.stdout.strip() or "0" * 40


def _data_hash(ctx: RunContext) -> str:
    return file_sha256(ctx.run_dir / DATASET_TABLE)


def write_ood(
    root: Path, member: int, backbone: str, embeddings: NDArray[np.float64], percentile: float
) -> dict[str, Any]:
    """Writes the training embeddings a photo is compared against, with their threshold.

    The threshold is a percentile of the training scores themselves, so the service flags the
    scenes that are unusual for this very model rather than for a number someone chose. The
    percentile is high on purpose: a flag on an ordinary photo of a truck costs the operator a
    pointless check, and the frames worth flagging are far above any of them.

    Args:
        root: Package folder.
        member: Index of the member whose embeddings these are.
        backbone: Registry key of that member's encoder.
        embeddings: One row per training photo, from that very encoder.
        percentile: Where the unusual-scene threshold sits among the training scores.

    Returns:
        The `ood` block of the manifest.
    """
    reference = OodReference.build(backbone, embeddings)
    path = root / "ood" / "reference.npz"
    reference.save(path)
    scores = np.array([reference.score(row, _OOD_NEIGHBOURS) for row in embeddings])
    return {
        "ood": {
            "member": member,
            "k": _OOD_NEIGHBOURS,
            "threshold": float(np.percentile(scores, percentile)),
            "reference": str(path.relative_to(root)),
        }
    }


class Credit(NamedTuple):
    """One more set of weights a package carries besides the encoders of its members.

    Attributes:
        name: Checkpoint name, as its source calls it.
        license: SPDX identifier, naming a text in the licences folder.
        origin: Who published it, for the NOTICE.
        source: Where it comes from.
    """

    name: str
    license: str
    origin: str
    source: str


def write_licenses(
    root: Path,
    backbones: Sequence[Backbone],
    licenses_dir: Path,
    *,
    fine_tuned: bool = False,
    extra: Sequence[Credit] = (),
) -> list[str]:
    """Copies the licence of every checkpoint into the package and writes the NOTICE.

    The package carries the weights, so it has to carry their terms with them: Apache-2.0
    asks for the licence text and the attribution to travel with any copy, and for a note
    when the weights were changed. A checkpoint whose licence has no text stops the build
    instead of shipping unlicensed weights.

    Args:
        root: Package folder.
        backbones: Loaded encoders of the members, in member order.
        licenses_dir: Folder holding one text per SPDX identifier the registry names.
        fine_tuned: Whether the package carries weights fine-tuned by the team rather than
            the untouched checkpoints.
        extra: Weights besides the members' encoders, such as a depth model; they are never
            fine-tuned by us.

    Returns:
        Package paths of the licence texts and the NOTICE, sorted.

    Raises:
        FileNotFoundError: If a licence named by the registry has no text in `licenses_dir`.
    """
    checkpoints = {
        backbone.spec.timm_name: backbone.spec
        for backbone in sorted(backbones, key=lambda item: item.spec.timm_name)
    }
    credits = [
        Credit(timm_name, spec.license, spec.origin, f"https://huggingface.co/timm/{timm_name}")
        for timm_name, spec in checkpoints.items()
    ]
    folder = root / "licenses"
    folder.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name in sorted({item.license for item in [*credits, *extra]}):
        source = licenses_dir / f"{name}.txt"
        if not source.is_file():
            raise FileNotFoundError(
                f"{source} is missing; a package may not ship weights under {name} "
                f"without their licence text"
            )
        shutil.copyfile(source, folder / f"{name}.txt")
        written.append(f"licenses/{name}.txt")

    change = (
        "- веса дообучены командой Loadvision на размеченных фото кузовов: изменены все слои"
        if fine_tuned
        else "- веса не изменялись"
    )
    lines = [
        "# NOTICE",
        "",
        "Пакет несёт веса предобученных моделей. Каждый чекпоинт остаётся под своей лицензией;",
        "их тексты лежат рядом в этой же папке.",
        "",
    ]
    for item in credits:
        lines += [
            f"## {item.name}",
            "",
            f"- происхождение: {item.origin}",
            f"- лицензия: {item.license}, текст в `licenses/{item.license}.txt`",
            f"- источник: {item.source}",
            change,
            "",
        ]
    for item in extra:
        lines += [
            f"## {item.name}",
            "",
            f"- происхождение: {item.origin}",
            f"- лицензия: {item.license}, текст в `licenses/{item.license}.txt`",
            f"- источник: {item.source}",
            "- веса не изменялись",
            "",
        ]
    (folder / _NOTICE_FILE).write_text("\n".join(lines), encoding="utf-8")
    written.append(f"licenses/{_NOTICE_FILE}")
    return sorted(written)


def write_steps(root: Path, fitted: FittedChain) -> list[PostprocessStep]:
    """Writes the curves of a fitted chain into the package and describes its steps.

    Args:
        root: Package folder.
        fitted: Chain fitted on the out-of-fold predictions.

    Returns:
        The `postprocess` entries of the manifest, in chain order.
    """
    steps: list[PostprocessStep] = []
    for index, step in enumerate(fitted.steps):
        path = step.save_curves_to(root / "postprocess" / f"step_{index}.npz")
        steps.append(
            PostprocessStep.model_validate(
                {
                    "name": step.name,
                    "params": step.params,
                    "file": None if path is None else str(path.relative_to(root)),
                }
            )
        )
    return steps


def cv_block(oof: Path, folds_file: Path) -> dict[str, Any]:
    """Out-of-fold metrics of a package and the folds they were measured on.

    Args:
        oof: `oof.csv` the package was measured by.
        folds_file: Folds file of that measurement.

    Returns:
        The `cv` and `folds` blocks of the manifest.
    """
    with oof.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    labels = np.array([float(row["label"]) for row in rows])
    predicted = np.array([float(row["predicted"]) for row in rows])
    metrics = summary(labels, predicted, [int(row["fold"]) for row in rows])
    return {
        "cv": CvMetrics(
            mae=metrics["mae"],
            within_10=metrics["within_10"] * _PERCENT,
            mae_by_range=mae_by_range(labels, predicted),
        ).model_dump(mode="json"),
        "folds": FoldsRef(file=str(folds_file), sha256=file_sha256(folds_file)).model_dump(
            mode="json"
        ),
    }
