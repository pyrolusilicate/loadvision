"""LoadPredictor implementation backed by a model package."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

from loadvision.application.ports import Prediction
from loadvision.domain.assessment import LoadPercent
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage
from loadvision.imaging.preprocessing import preprocess
from loadvision.ml.packaging.manifest import Member, Postprocess, Runtime
from loadvision.ml.packaging.model_package import ModelPackage, ModelPackageError

_HEAD_KEYS: Final = frozenset({"mean", "scale", "weights", "intercept"})

_MIN_LOAD_PCT = 0.0
_MAX_LOAD_PCT = 100.0


@dataclass(frozen=True, slots=True)
class RidgeHead:
    """Linear head with the standardization it was fitted with.

    Attributes:
        mean: Mean of every feature in the training set.
        scale: Standard deviation of every feature; never zero.
        weights: Weight of every standardized feature.
        intercept: Value predicted for an average photo.
    """

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    weights: NDArray[np.float64]
    intercept: float

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reads a head from an `.npz` file with `mean`, `scale`, `weights` and `intercept`.

        Args:
            path: Head file inside a model package.

        Returns:
            The head.

        Raises:
            ModelPackageError: If the file cannot be read, a key is missing, the shapes
                disagree or a scale is not positive.
        """
        try:
            with np.load(path, allow_pickle=False) as saved:
                stored = set(saved.files)
                missing = sorted(_HEAD_KEYS - stored)
                if missing:
                    raise ModelPackageError(f"{path.name}: missing {', '.join(missing)}")
                # Only the four keys are read: a head file may carry extra arrays, such as the
                # baseline's own method and median, and they are none of our business.
                arrays = {key: np.asarray(saved[key], dtype=np.float64) for key in _HEAD_KEYS}
        except (OSError, ValueError) as error:
            raise ModelPackageError(f"{path.name}: {error}") from error
        mean, scale, weights = arrays["mean"], arrays["scale"], arrays["weights"]
        if not (mean.shape == scale.shape == weights.shape) or mean.ndim != 1:
            raise ModelPackageError(f"{path.name}: mean, scale and weights must share one shape")
        if not bool((scale > 0).all()):
            raise ModelPackageError(f"{path.name}: scale must be positive")
        return cls(
            mean=mean, scale=scale, weights=weights, intercept=float(arrays["intercept"].item())
        )

    def apply(self, features: NDArray[np.floating[Any]]) -> float:
        """Returns the load percent this head predicts for one feature vector.

        Args:
            features: Features of one photo, in the order the head was fitted on.

        Returns:
            Prediction before the ensemble average and postprocessing.

        Raises:
            ModelPackageError: If the number of features does not match the head.
        """
        flat = np.asarray(features, dtype=np.float64).ravel()
        if flat.shape != self.mean.shape:
            raise ModelPackageError(f"head expects {self.mean.size} features, got {flat.size}")
        return float((flat - self.mean) / self.scale @ self.weights + self.intercept)


@dataclass(frozen=True, slots=True)
class _LoadedMember:
    member: Member
    head: RidgeHead


def _apply_postprocess(value: float, postprocess: Postprocess) -> float:
    clipped = min(max(value, _MIN_LOAD_PCT), _MAX_LOAD_PCT)
    if postprocess.round_to is not None:
        clipped = round(clipped / postprocess.round_to) * postprocess.round_to
    if postprocess.snap_100 is not None and clipped >= postprocess.snap_100:
        clipped = _MAX_LOAD_PCT
    return min(max(clipped, _MIN_LOAD_PCT), _MAX_LOAD_PCT)


class PackagePredictor:
    """Predicts load with the ensemble described by a model package.

    Every member prepares the photo with its own preprocessing, so members with different
    input sizes live in one package. The ensemble is their weighted average, and
    `uncertainty` is how far the members are apart.

    Args:
        package: Verified model package.

    Raises:
        ModelPackageError: If a member needs a runtime that is not supported yet, or its head
            cannot be read. Both fail here, at load time, not on the first request.
    """

    def __init__(self, package: ModelPackage) -> None:
        self._package = package
        self._members = tuple(
            _LoadedMember(member=member, head=self._load_head(package, member))
            for member in package.manifest.members
        )

    @staticmethod
    def _load_head(package: ModelPackage, member: Member) -> RidgeHead:
        if member.runtime is not Runtime.NUMPY:
            raise ModelPackageError(f"runtime {member.runtime} is not supported yet")
        return RidgeHead.load(package.path(member.head))

    @property
    def version(self) -> ModelVersionId:
        """Version of the model package that makes the predictions."""
        return self._package.version

    def predict(self, images: Sequence[PreparedImage]) -> list[Prediction]:
        """Predicts load for a batch of photos.

        Args:
            images: Decoded photos.

        Returns:
            One prediction per photo, in the same order.

        Raises:
            LoadvisionError: If a photo cannot be prepared or a head does not match its
                features.
        """
        return [self._predict_one(image) for image in images]

    def _predict_one(self, image: PreparedImage) -> Prediction:
        values = [
            loaded.head.apply(
                preprocess(
                    loaded.member.preprocessing.name, image, loaded.member.preprocessing.params
                )
            )
            for loaded in self._members
        ]
        average = sum(
            value * loaded.member.weight
            for value, loaded in zip(values, self._members, strict=True)
        )
        spread = max(values) - min(values) if len(values) > 1 else None
        return Prediction(
            load_pct=LoadPercent(_apply_postprocess(average, self._package.manifest.postprocess)),
            uncertainty=spread,
        )
