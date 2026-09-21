"""Model package manifest schema."""

import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_serializer,
    model_validator,
)

from loadvision.domain.model_version import ModelVersionId
from loadvision.domain.timestamps import format_utc
from loadvision.imaging.crops import WHOLE

MANIFEST_FILE: Final = "manifest.json"

# Segments start with a letter, digit, "_" or "-": no ".", "..", hidden files, empty segments,
# backslashes or absolute paths, so a path from a manifest cannot leave the package folder.
RELATIVE_PATH_PATTERN: Final = r"[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*"
# Keys of the backbone and preprocessing registries.
REGISTRY_KEY_PATTERN: Final = "[a-z][a-z0-9_]{0,63}"

# Member weights are floats, so their sum is 1 only up to rounding.
_WEIGHT_SUM_TOLERANCE: Final = 1e-6

_RELATIVE_PATH_RE: Final = re.compile(RELATIVE_PATH_PATTERN)


def _check_relative_path(value: str) -> str:
    if _RELATIVE_PATH_RE.fullmatch(value) is None:
        raise ValueError(f"path must match {RELATIVE_PATH_PATTERN}")
    return value


def _check_package_path(value: str) -> str:
    if value == MANIFEST_FILE:
        raise ValueError(f"{MANIFEST_FILE} cannot list itself")
    return value


def _check_model_version(value: str) -> str:
    ModelVersionId(value)
    return value


type RelativePath = Annotated[str, AfterValidator(_check_relative_path)]
type PackagePath = Annotated[RelativePath, AfterValidator(_check_package_path)]
type Sha256Hex = Annotated[str, Field(pattern="^[0-9a-f]{64}$")]
type RegistryKey = Annotated[str, Field(pattern=f"^{REGISTRY_KEY_PATTERN}$")]


class _Schema(BaseModel):
    # Unknown fields are errors, so a typo in a field name cannot silently turn a setting off.
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Runtime(StrEnum):
    """How a member's backbone is executed."""

    NUMPY = "numpy"
    ONNX = "onnx"
    TORCH = "torch"


class Preprocessing(_Schema):
    """Preprocessing of one member.

    Attributes:
        name: Key in the preprocessing registry of `imaging/preprocessing.py`.
        params: Numeric parameters of that preprocessing, such as input height and width.
    """

    name: RegistryKey
    params: dict[str, StrictInt | StrictFloat] = Field(default_factory=dict)


class HeadKind(StrEnum):
    """How the head file of a member is read."""

    LINEAR = "linear"
    BINS = "bins"


class Member(_Schema):
    """One model of the ensemble.

    Attributes:
        backbone: Key in the backbone registry.
        runtime: How the backbone is executed.
        weights: Backbone weights; None only for the numpy runtime, which computes fixed
            features.
        head: Head file that turns backbone output into a load percent.
        weight: Share of the member in the ensemble average.
        preprocessing: Input preparation for this backbone.
        inputs: What the encoder is shown, by key of the input registry of `imaging/crops.py`.
            The head reads the vectors in this order, so it must match the training.
        head_kind: How to read the head file. `linear` is the ridge head every package before
            21.09 carries, in the `.npz` form; `bins` is a distribution over the grid the
            expert marked on, saved as safetensors. The default keeps older packages loading
            unchanged.
    """

    backbone: RegistryKey
    runtime: Runtime
    weights: PackagePath | None = None
    head: PackagePath
    weight: float = Field(ge=0, le=1)
    preprocessing: Preprocessing
    inputs: tuple[RegistryKey, ...] = Field(default=(WHOLE,), min_length=1)
    head_kind: HeadKind = HeadKind.LINEAR

    @model_validator(mode="after")
    def _require_weights_for_neural_runtimes(self) -> Self:
        if self.runtime is not Runtime.NUMPY and self.weights is None:
            raise ValueError(f"runtime {self.runtime} needs a weights file")
        return self


class CvMetrics(_Schema):
    """Cross-validation metrics of the package on the folds from `Manifest.folds`.

    Attributes:
        mae: Mean absolute error in percentage points.
        within_10: Percent of photos with an absolute error of at most 10 points.
        mae_by_range: MAE by label range, keyed by range name such as ``90-99``.
    """

    mae: float = Field(ge=0)
    within_10: float = Field(ge=0, le=100)
    mae_by_range: dict[str, Annotated[float, Field(ge=0)]] = Field(default_factory=dict)


class OodConfig(_Schema):
    """Out-of-distribution score: distance to the nearest train embeddings.

    Attributes:
        member: Index of the ensemble member whose embeddings the reference holds.
        k: Number of nearest train embeddings averaged into the score.
        threshold: Score above which a photo is flagged as an unusual scene.
        reference: File with the train embeddings.
    """

    member: int = Field(ge=0)
    k: int = Field(ge=1)
    threshold: float = Field(ge=0)
    reference: PackagePath


class DepthRef(_Schema):
    """Depth model whose grid the members read next to the photo.

    A member of such a package is fed the embedding of its encoder and this grid, in that
    order, so the head sees exactly what it was fitted on.

    Attributes:
        model: Folder inside the package with the depth checkpoint.
        rows: Rows of the grid the depth map is pooled into.
        cols: Columns of that grid.
    """

    model: PackagePath
    rows: int = Field(ge=1)
    cols: int = Field(ge=1)


class PostprocessStep(_Schema):
    """One step of the chain applied to the ensemble output.

    Attributes:
        name: Key in the postprocessing registry of `ml/postprocess.py`.
        params: Numeric parameters, including the ones learned on out-of-fold predictions,
            such as a snap threshold.
        file: Learned curve of the step, such as a calibration; None for steps without one.
    """

    name: RegistryKey
    params: dict[str, StrictInt | StrictFloat] = Field(default_factory=dict)
    file: PackagePath | None = None


class FoldsRef(_Schema):
    """Folds the CV metrics were computed on.

    Attributes:
        file: Repository path of the folds file, such as ``configs/splits/folds.json``.
        sha256: Hash of that file.
    """

    file: RelativePath
    sha256: Sha256Hex


class Manifest(_Schema):
    """`manifest.json` of a model package: what is inside and how to run it.

    Attributes:
        schema_version: Format version; changes only on an incompatible edit.
        version: Package version, the `ModelVersionId` format.
        created_at: Build time, timezone-aware; written in UTC.
        git_commit: Commit of the code that built the package.
        data_hash: SHA-256 of the dataset manifest the models were trained on.
        members: Models of the ensemble; their weights sum to 1.
        cv: CV metrics; None when the models were not evaluated on the shared folds.
        ood: Out-of-distribution score settings; None disables the score.
        depth: Depth model the members read alongside the photo; None when they read the photo
            alone.
        postprocess: Steps applied to the ensemble output in list order; every prediction is
            clipped to [0, 100] after them, so clipping is not a step.
        folds: Folds of the CV metrics; None when `cv` is None.
        licenses: License texts of the weights.
        files: SHA-256 of every package file except the manifest; the loader verifies them.
    """

    schema_version: Literal[1]
    version: Annotated[str, AfterValidator(_check_model_version)]
    created_at: AwareDatetime
    git_commit: str = Field(pattern="^[0-9a-f]{40}$")
    data_hash: Sha256Hex
    members: tuple[Member, ...] = Field(min_length=1)
    cv: CvMetrics | None = None
    ood: OodConfig | None = None
    depth: DepthRef | None = None
    postprocess: tuple[PostprocessStep, ...] = ()
    folds: FoldsRef | None = None
    licenses: tuple[PackagePath, ...] = ()
    files: dict[PackagePath, Sha256Hex]

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        total = math.fsum(member.weight for member in self.members)
        if abs(total - 1) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"member weights must sum to 1, got {total}")
        if (self.cv is None) != (self.folds is None):
            raise ValueError("cv and folds must be set together")
        if self.ood is not None and self.ood.member >= len(self.members):
            raise ValueError(f"ood.member {self.ood.member} is not a member index")
        unlisted = sorted(set(self.referenced_files()) - set(self.files))
        if unlisted:
            raise ValueError(f"files not listed in files: {', '.join(unlisted)}")
        return self

    @field_serializer("created_at")
    def _write_created_at(self, value: datetime) -> str:
        return format_utc(value)

    @property
    def model_version(self) -> ModelVersionId:
        """Package version as a domain value."""
        return ModelVersionId(self.version)

    def referenced_files(self) -> list[str]:
        """Returns every package file the manifest points to, in manifest order."""
        paths: list[str] = []
        for member in self.members:
            if member.weights is not None:
                paths.append(member.weights)
            paths.append(member.head)
        if self.ood is not None:
            paths.append(self.ood.reference)
        paths.extend(step.file for step in self.postprocess if step.file is not None)
        paths.extend(self.licenses)
        return paths
