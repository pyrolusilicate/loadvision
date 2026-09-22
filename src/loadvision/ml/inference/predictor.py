"""LoadPredictor implementation backed by a model package."""

import json
import logging
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

from loadvision.application.ports import Prediction
from loadvision.domain.assessment import LoadPercent, ReviewReason
from loadvision.domain.model_version import ModelVersionId
from loadvision.errors import LoadvisionError
from loadvision.imaging import crops
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.preprocessing import preprocess
from loadvision.ml.inference.ood import OodReference
from loadvision.ml.inference.unusable_frames import frames_fingerprint, unusable_frames
from loadvision.ml.packaging.baseline_package import BASELINE_BACKBONE
from loadvision.ml.packaging.manifest import HeadKind, Member, Runtime
from loadvision.ml.packaging.model_package import ModelPackage, ModelPackageError
from loadvision.ml.postprocess import ASPECT, Applier, load_curves, make_chain

_logger = logging.getLogger(__name__)

_HEAD_KEYS: Final = frozenset({"mean", "scale", "weights", "intercept"})

# The numpy runtime has no network: the preprocessing already returns the features the head
# reads, and the baseline features are the only ones it knows.
_NUMPY_BACKBONES: Final = frozenset({BASELINE_BACKBONE})

# Size of the blank photo every member predicts on at load time.
_PROBE_HEIGHT: Final = 8
_PROBE_WIDTH: Final = 6

_MIN_LOAD_PCT = 0.0
_MAX_LOAD_PCT = 100.0

# What a head of bins calls the grid it was fitted on, when it carries one at all.
_CENTRES_KEY: Final = "centres"
# How far the stored grid may stand from ours: the bins of a real head are whole percent points
# apart, and this leaves room only for the noise of keeping them as float32.
_CENTRES_TOLERANCE: Final = 1e-3


def _clip(value: float) -> float:
    return min(max(value, _MIN_LOAD_PCT), _MAX_LOAD_PCT)


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

        The value is clipped, exactly as `ml.training.head_trainer.FittedHead.predict` clips it:
        a member that answers 111 must weigh the same here as it did when the ensemble was
        measured, otherwise the service and the submission drift apart on every photo where
        one member leaves the range.

        Args:
            features: Features of one photo, in the order the head was fitted on.

        Returns:
            Load from 0 to 100, before the ensemble average and postprocessing.

        Raises:
            ModelPackageError: If the number of features does not match the head.
        """
        flat = np.asarray(features, dtype=np.float64).ravel()
        if flat.shape != self.mean.shape:
            raise ModelPackageError(f"head expects {self.mean.size} features, got {flat.size}")
        return _clip(float((flat - self.mean) / self.scale @ self.weights + self.intercept))


@dataclass(frozen=True, slots=True)
class BinsHead:
    """Head that answers with a distribution over the grid the expert marked on.

    The member predicts a weight for every bin, softmax turns the weights into shares, and the
    answer is the mean of the grid under them. It cannot be written as a ridge head, so it
    travels in its own file and the manifest says so with `head_kind`.

    Attributes:
        weights: One row of feature weights per bin.
        bias: One number per bin.
        centres: Load percent at the centre of every bin.
    """

    weights: NDArray[np.float64]
    bias: NDArray[np.float64]
    centres: NDArray[np.float64]

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reads the head from the safetensors file the training wrote.

        Args:
            path: Head file inside a model package.

        Returns:
            The head.

        Raises:
            ModelPackageError: If the file cannot be read or its shapes disagree.
        """
        from safetensors.numpy import load_file

        try:
            stored = load_file(str(path))
        except (OSError, ValueError) as error:
            raise ModelPackageError(f"{path.name}: {error}") from error
        missing = sorted({"layer.weight", "layer.bias"} - set(stored))
        if missing:
            raise ModelPackageError(f"{path.name}: missing {', '.join(missing)}")
        weights = np.asarray(stored["layer.weight"], dtype=np.float64)
        bias = np.asarray(stored["layer.bias"], dtype=np.float64)
        if weights.ndim != 2 or bias.shape != (weights.shape[0],):
            raise ModelPackageError(f"{path.name}: weights and bias must agree on the bin count")
        centres = np.linspace(_MIN_LOAD_PCT, _MAX_LOAD_PCT, weights.shape[0])
        if _CENTRES_KEY in stored:
            _check_centres(path, np.asarray(stored[_CENTRES_KEY], dtype=np.float64), centres)
        return cls(weights=weights, bias=bias, centres=centres)

    def apply(self, features: NDArray[np.floating[Any]]) -> float:
        """Returns the load percent this head predicts for one feature vector.

        Raises:
            ModelPackageError: If the number of features does not match the head.
        """
        flat = np.asarray(features, dtype=np.float64).ravel()
        if flat.shape[0] != self.weights.shape[1]:
            raise ModelPackageError(
                f"head expects {self.weights.shape[1]} features, got {flat.size}"
            )
        scores = self.weights @ flat + self.bias
        shares = np.exp(scores - scores.max())
        return _clip(float(shares @ self.centres / shares.sum()))


def _check_centres(path: Path, stored: NDArray[np.float64], grid: NDArray[np.float64]) -> None:
    """Fails when a head was fitted on a grid of bins other than the one this service assumes.

    The centres are not read from the file, they are computed: an even grid from 0 to 100 over
    as many bins as the weights carry. A head fitted on any other grid - bins packed closer
    near the empty end, or a grid that stops short of a full body - would load and answer all
    the same, only wrong, and nothing would say so. So when the file carries its own centres,
    they have to agree with ours.

    Training writes them as shares of a full body and this service speaks in percent, so a grid
    that ends at 1 is read as shares and scaled by a hundred.

    Args:
        path: Head file, named in the message.
        stored: Centres as the file carries them.
        grid: Centres this service computed.

    Raises:
        ModelPackageError: If the grids disagree by more than the noise of float32 storage.
    """
    if stored.shape != grid.shape:
        raise ModelPackageError(
            f"{path.name}: the head carries {stored.size} bin centres for {grid.size} bins"
        )
    percent = stored * _MAX_LOAD_PCT if float(stored.max()) <= 1.0 else stored
    if not bool(np.allclose(percent, grid, atol=_CENTRES_TOLERANCE)):
        raise ModelPackageError(
            f"{path.name}: the head was fitted on bins from {percent.min():.2f} to "
            f"{percent.max():.2f}, and this service reads them as {grid.min():.2f} to "
            f"{grid.max():.2f} in equal steps"
        )


type Head = RidgeHead | BinsHead


def load_head(kind: HeadKind, path: Path) -> Head:
    """Reads the head of a member the way its `head_kind` asks for."""
    return BinsHead.load(path) if kind is HeadKind.BINS else RidgeHead.load(path)


@dataclass(frozen=True, slots=True)
class _LoadedMember:
    member: Member
    head: Head
    # None for the numpy runtime, where the preprocessing already returns the features.
    encoder: object | None = None


class PackagePredictor:
    """Predicts load with the ensemble described by a model package.

    Every member prepares the photo with its own preprocessing, so members with different
    input sizes live in one package. The ensemble is their weighted average, and
    `uncertainty` is how far the members are apart. A photo whose OOD score is above the
    threshold of the package is marked as an unusual scene.

    Members that name the same checkpoint at the same input size run it once per input and
    pool its tokens each in their own way, so an ensemble of four costs two runs of the
    encoder. A member may be shown more than the whole frame: `Member.inputs` names what it
    sees, and its head reads the vectors in that order.

    A package may name a depth model; then every torch member is fed its encoder embedding
    together with the depth grid of the photo, in that order, the way its head was fitted.

    Args:
        package: Verified model package.
        device: Torch device for members with the torch runtime.
        scale_cache: File the measured scale of this package is kept in, so a restart does
            not pay for it again; None measures it every time.

    Raises:
        ModelPackageError: If a member needs a runtime or backbone that is not supported, its
            head cannot be read, its preprocessing and head do not fit together, or a
            postprocessing step is unknown or misses its file. Every member predicts on a blank
            photo here, so a wrong parameter or a head for other features fails at load time,
            not on the first request.
    """

    def __init__(
        self, package: ModelPackage, device: str = "cpu", scale_cache: Path | None = None
    ) -> None:
        self._package = package
        self._device = device
        self._members = tuple(
            _LoadedMember(
                member=member,
                head=load_head(member.head_kind, package.path(member.head)),
                encoder=self._load_encoder(member),
            )
            for member in package.manifest.members
        )
        self._postprocess = self._load_postprocess(package)
        self._depth = self._load_depth(package)
        self._ood = self._load_ood(package)
        self._check_on_blank_photo()
        self._unusable = self._scale(scale_cache)

    def _load_encoder(self, member: Member) -> object | None:
        if member.runtime is Runtime.NUMPY:
            if member.backbone not in _NUMPY_BACKBONES:
                raise ModelPackageError(
                    f"backbone {member.backbone!r} is not supported by runtime {member.runtime}"
                )
            return None
        if member.runtime is not Runtime.TORCH:
            raise ModelPackageError(f"runtime {member.runtime} is not supported yet")
        # Imported here and not at the top: a package of numpy members must load in a service
        # image that carries no torch at all.
        from loadvision.ml.models.backbones import load_backbone

        # The validator of `Member` gives every torch member a weights file; the package has
        # already checked its hash.
        weights = None if member.weights is None else self._package.path(member.weights)
        try:
            return load_backbone(member.backbone, self._device, weights)
        except KeyError as error:
            raise ModelPackageError(f"unknown backbone {member.backbone!r}") from error

    def _load_depth(self, package: ModelPackage) -> object | None:
        config = package.manifest.depth
        if config is None:
            return None
        # Imported here with torch: a package of numpy members needs neither.
        from loadvision.ml.models.depth import DepthEncoder

        # A folder, not a file, so it is taken from the root; every file inside it is listed
        # in the manifest and was verified when the package was loaded.
        try:
            return DepthEncoder.load(package.root / config.model, self._device)
        except (OSError, ValueError) as error:
            raise ModelPackageError(f"depth model: {error}") from error

    def _load_ood(self, package: ModelPackage) -> OodReference | None:
        config = package.manifest.ood
        if config is None:
            return None
        backbone = self._members[config.member].member.backbone
        try:
            return OodReference.load(package.path(config.reference), backbone)
        except (OSError, ValueError) as error:
            raise ModelPackageError(f"ood reference: {error}") from error

    @staticmethod
    def _load_postprocess(package: ModelPackage) -> Applier | None:
        steps = package.manifest.postprocess
        if not steps:
            return None
        try:
            return make_chain(
                [
                    (
                        step.name,
                        dict(step.params),
                        load_curves(package.path(step.file)) if step.file else None,
                    )
                    for step in steps
                ]
            )
        except (KeyError, OSError, ValueError) as error:
            raise ModelPackageError(f"postprocessing chain: {error}") from error

    def _check_on_blank_photo(self) -> None:
        pixels: NDArray[np.uint8] = np.zeros((_PROBE_HEIGHT, _PROBE_WIDTH, 3), dtype=np.uint8)
        probe = PreparedImage(b"blank", MediaType.PNG, pixels)
        for index, loaded in enumerate(self._members):
            try:
                self._member_value(loaded, probe)
            except LoadvisionError as error:
                raise ModelPackageError(f"member {index}: {error.detail}") from error

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

    def _member_value(self, loaded: _LoadedMember, image: PreparedImage) -> float:
        return self._features({id(loaded): loaded}, image)[id(loaded)][0]

    def _features(
        self, members: dict[int, _LoadedMember], image: PreparedImage
    ) -> dict[int, tuple[float, NDArray[np.float64] | None]]:
        """Predicts every member for one photo, running each encoder at most once.

        Returns the value of every member and, for members with an encoder, its embedding: the
        out-of-distribution score is the distance from that very embedding to the training ones.
        """
        values: dict[int, tuple[float, NDArray[np.float64] | None]] = {}
        groups: dict[tuple[str, str, str, str], list[tuple[int, _LoadedMember]]] = {}
        depth = None if self._depth is None else _depth_grid(self._depth, image)
        for key, loaded in members.items():
            if loaded.encoder is None:
                preprocessing = loaded.member.preprocessing
                prepared = preprocess(preprocessing.name, image, preprocessing.params)
                values[key] = (loaded.head.apply(prepared), None)
                continue
            groups.setdefault(_group_key(loaded), []).append((key, loaded))

        for sharing in groups.values():
            values.update(self._torch_group(sharing, image, depth))
        return values

    def _torch_group(
        self,
        sharing: list[tuple[int, _LoadedMember]],
        image: PreparedImage,
        depth: NDArray[np.float64] | None = None,
    ) -> dict[int, tuple[float, NDArray[np.float64] | None]]:
        import torch

        first = _backbone(sharing[0][1])
        wanted = {name for _key, loaded in sharing for name in loaded.member.inputs}
        # One run of the encoder per input and view for the whole group; its members differ
        # only in how they pool the tokens, and that is cheap next to the encoder itself.
        totals: dict[tuple[int, str], NDArray[np.float64]] = {}
        with torch.inference_mode():
            for name in wanted:
                shown = crops.get(name)(image)
                for view in first.views:
                    tokens = first.tokens(first.prepare([shown], view).to(self._device))
                    for key, loaded in sharing:
                        if name not in loaded.member.inputs:
                            continue
                        backbone = _backbone(loaded)
                        pooled = backbone.pool(tokens)
                        if view.mirrored:
                            pooled = backbone.unmirror(pooled)
                        vector = pooled.float().cpu().numpy()[0].astype(np.float64)
                        cell = (key, name)
                        totals[cell] = vector if cell not in totals else totals[cell] + vector
        views = len(first.views)
        computed: dict[int, tuple[float, NDArray[np.float64] | None]] = {}
        for key, loaded in sharing:
            vectors = [totals[(key, name)] / views for name in loaded.member.inputs]
            # The head was fitted on the vectors of its inputs in this order, followed by the
            # depth grid. The first vector, the photo as a whole, is what the
            # out-of-distribution score compares.
            parts = vectors if depth is None else [*vectors, depth]
            computed[key] = (loaded.head.apply(np.concatenate(parts)), vectors[0])
        return computed

    @property
    def band(self) -> tuple[float, float] | None:
        """Returns the two ends of this package's own scale of distances.

        Returns:
            The far edge of the training photos and the distance at which frames holding
            nothing land, or None when the package measures no distances.
        """
        config = self._package.manifest.ood
        if config is None or self._unusable is None:
            return None
        return config.threshold, self._unusable

    def ood_score(self, image: PreparedImage) -> float | None:
        """Returns how far a photo stands from the training ones, and nothing else.

        The package answers with five members, a depth map, five heads and a calibration curve,
        and none of that reaches this number: the score compares one embedding of one member
        with the training ones. Asking for a whole prediction to read it off costs three times
        the work, and a tool that walks a folder of photos pays that on every one of them.

        Args:
            image: Photo to place on the scale.

        Returns:
            Distance to the training photos, or None when the package measures none.
        """
        config = self._package.manifest.ood
        if config is None or self._ood is None:
            return None
        embedding = self._embedding(self._members[config.member], image)
        return None if embedding is None else self._ood.score(embedding, config.k)

    def _embedding(self, loaded: _LoadedMember, image: PreparedImage) -> NDArray[np.float64] | None:
        """Returns the vector the distance is measured from, None for a member with no encoder.

        It is the first input of the member, the photo as a whole - the very vector
        `_torch_group` hands to the score; the other inputs and the depth grid only feed the
        head. The averaging over views is not repeated here: `Backbone.embed` is the one place
        the way from a photo to a vector lives, and a second copy of it would drift.
        """
        import torch

        backbone = _backbone(loaded)
        if backbone is None:
            return None
        shown = crops.get(loaded.member.inputs[0])(image)
        with torch.inference_mode():
            embedded = backbone.embed([shown], self._device)
        return np.asarray(embedded.float().cpu().numpy()[0], dtype=np.float64)

    def _predict_one(self, image: PreparedImage) -> Prediction:
        by_key = dict(enumerate(self._members))
        computed = self._features(by_key, image)
        values = [computed[index][0] for index in range(len(self._members))]
        average = sum(
            value * loaded.member.weight
            for value, loaded in zip(values, self._members, strict=True)
        )
        corrected = _clip(average)
        if self._postprocess is not None:
            aspect = np.array([image.width / image.height], dtype=np.float64)
            corrected = float(
                self._postprocess(np.array([corrected], dtype=np.float64), {ASPECT: aspect})[0]
            )
        spread = max(values) - min(values) if len(values) > 1 else None
        ood_score = self._ood_score(computed)
        ood = self._package.manifest.ood
        return Prediction(
            load_pct=LoadPercent(_clip(corrected)),
            uncertainty=spread,
            ood_score=ood_score,
            # The score alone says nothing: the caller compares it with the scale of this
            # package, not with a number of its own.
            ood_threshold=None if ood is None or ood_score is None else ood.threshold,
            ood_unusable=None if ood_score is None else self._unusable,
            review_reasons=self._review_reasons(ood_score),
        )

    def _scale(self, cache: Path | None) -> float | None:
        """Returns the measured scale of this package, from the cache when it is still valid.

        The measurement repeats exactly only where it was made: the same frames through the
        same weights on the same build. A different package, a different set of frames or a
        different build makes a stored number stale, and all three are named in the file.

        Args:
            cache: File to read and write, or None to measure without keeping the result.

        Returns:
            The distance at which empty frames land, or None when the package measures none.
        """
        frames = frames_fingerprint()
        build = _build_id(self._device)
        stored = _read_scale(cache) if cache is not None else None
        if stored is not None and stored.get("frames") == frames and stored.get("build") == build:
            measured = stored.get("unusable")
            if isinstance(measured, int | float):
                _logger.info("scale of %s read from %s", self._package.manifest.version, cache)
                return float(measured)
        measured = self._measure_unusable()
        if cache is not None and measured is not None:
            _write_scale(cache, self._package.manifest.version, frames, build, measured)
        return measured

    def _measure_unusable(self) -> float | None:
        """Returns where frames holding nothing land on the scale of this package.

        Costs one prediction per frame, once, when the package loads. Without it the caller
        would have only the threshold, which says where the training photos end but nothing
        about where a grey wall or noise begins.

        Returns:
            The smallest distance the empty frames reach, or None when the package measures
            no distances at all.
        """
        if self._ood is None:
            return None
        # The distance only, not a whole prediction: the other members, the depth map, the
        # heads and the postprocessing say nothing about where a frame sits, and this runs at
        # load time, before the service answers anything.
        scores = [self.ood_score(frame) for frame in unusable_frames()]
        measured = [score for score in scores if score is not None]
        if not measured:
            return None
        floor = min(measured)
        _logger.info("frames with nothing in them start at %.4f on this package", floor)
        return floor

    def _ood_score(
        self, computed: dict[int, tuple[float, NDArray[np.float64] | None]]
    ) -> float | None:
        config = self._package.manifest.ood
        if config is None or self._ood is None:
            return None
        embedding = computed[config.member][1]
        if embedding is None:
            return None
        return self._ood.score(embedding, config.k)

    def _review_reasons(self, ood_score: float | None) -> frozenset[ReviewReason]:
        # The threshold comes with the package: every encoder has its own scale of distances.
        config = self._package.manifest.ood
        if config is None or ood_score is None or ood_score <= config.threshold:
            return frozenset()
        return frozenset({ReviewReason.UNUSUAL_SCENE})


def _read_scale(path: Path) -> dict[str, Any] | None:
    """Reads a stored measurement; anything unreadable is treated as no measurement at all."""
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return stored if isinstance(stored, dict) else None


def _build_id(device: str) -> str:
    """Names what a measurement was made with, so another build does not reuse it.

    The two CPU builds we run agree to a rounding error - the macOS wheel and the linux one in
    the container land 1.2e-7 apart on the same frames - but a card runs other kernels, and a
    measurement made on the processor says nothing about the same package on the card. The
    version of torch is read from the installed distribution and not from the module: a service
    image of numpy-only packages carries no torch at all and must still load.

    Args:
        device: Torch device the predictor runs on.

    Returns:
        One line naming the device, the installed torch and the platform.
    """
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_version = "none"
    return f"{device} torch-{torch_version} {platform.system()}-{platform.machine()}"


def _write_scale(path: Path, version: str, frames: str, build: str, unusable: float) -> None:
    """Keeps a measurement beside the packages; losing it costs only the next measurement."""
    record = {"version": version, "frames": frames, "build": build, "unusable": unusable}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        _logger.warning("the scale of %s was measured but not kept: %s", version, error)


def _depth_grid(encoder: Any, image: PreparedImage) -> NDArray[np.float64]:
    return np.asarray(encoder.grid(image), dtype=np.float64)


def _backbone(loaded: _LoadedMember) -> Any:
    return loaded.encoder


def _group_key(loaded: _LoadedMember) -> tuple[str, str, str, str]:
    """What members must share to run the encoder once: the very weights, the input, the views.

    The weights file is part of the key, not only the checkpoint name: an ensemble of models
    fine-tuned from one checkpoint, such as the five models of a cross-validation, names the
    same architecture five times and must still run each of them.
    """
    backbone = _backbone(loaded)
    preprocessing = loaded.member.preprocessing
    numbers = ",".join(f"{key}={value:g}" for key, value in sorted(preprocessing.params.items()))
    return (
        f"{backbone.spec.timm_name}:{backbone.spec.height}x{backbone.spec.width}",
        loaded.member.weights or "",
        f"{preprocessing.name}({numbers})",
        ",".join(backbone.spec.views),
    )
