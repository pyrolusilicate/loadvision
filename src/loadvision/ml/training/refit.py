"""Retraining that keeps the encoders of a package and fits its heads again.

The worker used to train another family: four frozen encoders under ridge heads, against a
service answering with five fine-tuned ones. The gate then refused every candidate, and its
verdict said "frozen is worse than fine-tuned" rather than anything about the new labels.

Here the candidate is built in the family that is served. The encoders of the active package
are taken as they are and never trained; only the heads are fitted again, on the dataset that
now carries the confirmed labels. A run therefore costs one pass of the encoders over the
photos that are new since the last one - the embedding cache is keyed by the hash of a photo
and keeps everything it has seen before.

Two heads are fitted per member, and the difference between them matters. The head that ships
sees every photo, because more photos make a better head and the encoder is frozen anyway. The
head behind `metrics/oof.csv` sees only the folds its own encoder was fine-tuned on, so the
photo it is measured on was never seen by either of them. Mixing the two would turn the number
the gate reads into a claim about photos the model had already met.
"""

import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from loadvision.domain.timestamps import format_utc
from loadvision.ml.data.manifest import Sample, load_split
from loadvision.ml.data.splits import check_covers, read_folds
from loadvision.ml.models.backbones import load_backbone
from loadvision.ml.models.depth import DEPTH_MODEL, GRID_COLS, GRID_ROWS, depth_grids
from loadvision.ml.packaging.checkpoints import WEIGHTS_DIR
from loadvision.ml.packaging.manifest import Manifest, Member, PostprocessStep
from loadvision.ml.packaging.model_package import (
    ModelPackage,
    ModelPackageError,
    hash_package_files,
    write_package,
)
from loadvision.ml.pipeline.stages.build_package import cv_block, write_steps
from loadvision.ml.pipeline.stages.extract_embeddings import EMBEDDINGS_DIR
from loadvision.ml.pipeline.stages.train_folds import write_oof
from loadvision.ml.postprocess import ASPECT
from loadvision.ml.training.embedding_cache import embed_samples, read_image
from loadvision.ml.training.head_trainer import fit_head
from loadvision.ml.training.postprocess_fit import fit_chain

logger = logging.getLogger(__name__)

# Where a package keeps the predictions it was measured by; the gate reads this file.
PACKAGE_OOF: Final = "metrics/oof.csv"
# Ridge penalty of the refitted heads. The heads of a fine-tuned package were trained with the
# encoder rather than fitted on cached vectors, so there is no value to inherit; this is the
# middle of the range that proved best for a vector of this width.
DEFAULT_ALPHA: Final = 300.0
HEAD_KIND: Final = "ridge"
# Depth grids live beside the embeddings, under this name.
DEPTH_CACHE: Final = "depth.npz"

_MIN_LOAD: Final = 0.0
_MAX_LOAD: Final = 100.0


def can_refit(active: ModelPackage, n_folds: int) -> bool:
    """Whether the heads of this package can be fitted again on its own encoders.

    A photo is measured with the one member whose encoder never saw it, so the package has to
    carry exactly one member per fold, and every member has to bring weights of its own. A
    package built from an experiment config carries one member per encoder instead, and there
    the heads cannot be refitted this way - which is not a failure, only a different shape.

    Args:
        active: Package the service answers with.
        n_folds: How many folds the canonical file holds.

    Returns:
        True when `refit_heads` can work on this package.
    """
    members = active.manifest.members
    return len(members) == n_folds and all(member.weights is not None for member in members)


def refit_heads(
    *,
    active: ModelPackage,
    dataset: Path,
    folds_file: Path,
    cache_dir: Path,
    out_dir: Path,
    version: str,
    device: str = "cpu",
    alpha: float = DEFAULT_ALPHA,
    now: datetime | None = None,
) -> Path:
    """Builds a candidate package from the encoders of `active` and freshly fitted heads.

    Args:
        active: Package the service answers with; its encoders and everything but the heads are
            carried over untouched.
        dataset: Dataset root with `train/`, the base one plus the confirmed labels.
        folds_file: Canonical folds file; new photos already have a fold in it.
        cache_dir: Cache kept between runs; embeddings live under `EMBEDDINGS_DIR` in it.
        out_dir: Folder to write the candidate package into. It must not exist yet.
        version: Version of the candidate, normally the id of the run.
        device: Torch device the encoders run on.
        alpha: Ridge penalty of the refitted heads.
        now: Time written into the manifest; defaults to now.

    Returns:
        Path of the written package.

    Raises:
        ModelPackageError: If the active package has no member with an encoder of its own.
        FileNotFoundError: If the dataset or the folds file is missing.
        ValueError: If a fold holds no photo of its own.
    """
    samples = load_split(dataset, "train")
    folds = read_folds(folds_file)
    check_covers(folds, samples)
    numbers = np.asarray(folds.fold_indices(samples), dtype=np.int64)
    labels = np.array([sample.label for sample in samples], dtype=np.float64)
    members = active.manifest.members
    if not members:
        raise ModelPackageError(f"{active.root}: package has no members to refit")

    if len(members) != folds.n_folds:
        raise ModelPackageError(
            f"{active.root}: {len(members)} members against {folds.n_folds} folds; this refit "
            "measures a photo with the one member whose encoder never saw it, so it needs "
            "exactly one member per fold"
        )

    depth = _depth(active, samples, cache_dir, device)
    root = _copy_package(active.root, out_dir)
    computed = {}
    for index, member in enumerate(members):
        features = _features(active, member, index, samples, cache_dir, device, depth)
        # The head that ships sees every photo; the encoder is frozen, so nothing leaks.
        fit_head(HEAD_KIND, features, labels, {"alpha": alpha}).save(root / member.head)
        computed[index] = _by_fold(features, labels, numbers, folds.n_folds, alpha)

    raw = _out_of_fold(computed, labels, numbers, folds.n_folds)
    raw = np.clip(raw, _MIN_LOAD, _MAX_LOAD)
    # The calibration is worth about 1.9 points of MAE, so where it is fitted decides whether
    # the number the gate reads means anything. The one behind `oof.csv` is fitted per fold, on
    # the four the photo is not in; the one that ships is fitted on everything, like any other
    # package. Carrying over the curves of the active package instead would calibrate new heads
    # by the old ones, and would hand the candidate a metric fitted on its own answers.
    facts = {ASPECT: _aspects(samples)}
    steps = [(step.name, dict(step.params)) for step in active.manifest.postprocess]
    predicted = _calibrated(steps, raw, labels, numbers, facts)
    oof = root / PACKAGE_OOF
    write_oof(oof, samples, numbers.tolist(), labels, raw, predicted)
    shipped = write_steps(root, fit_chain(steps, raw, labels, facts)) if steps else []
    _write_manifest(root, active, version, folds_file, oof, shipped, now)
    logger.info(
        "refit %s from %s: %d photos, %d members, out-of-fold MAE %.3f",
        version,
        active.version.value,
        len(samples),
        len(members),
        float(np.abs(predicted - labels).mean()),
    )
    return root


def _features(
    active: ModelPackage,
    member: Member,
    index: int,
    samples: Sequence[Sample],
    cache_dir: Path,
    device: str,
    depth: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Returns the vectors this member's head is fitted on: its embedding, then the depth grid.

    The cache file is named after the package and the member, not after the registry key. Two
    packages name the same architecture, and the fingerprint of the cache covers preprocessing
    but not the weights, so one name for both would quietly fit a head on another model's
    vectors.
    """
    if member.weights is None:
        raise ModelPackageError(f"member {index} has no weights of its own to reuse")
    backbone = load_backbone(member.backbone, device, active.path(member.weights))
    named = replace(backbone, name=f"{active.version.value}-m{index}")
    vectors = embed_samples(
        samples, named, cache_dir=cache_dir / EMBEDDINGS_DIR, device=device
    ).astype(np.float64)
    return vectors if depth is None else np.hstack([vectors, depth])


def _by_fold(
    features: NDArray[np.float64],
    labels: NDArray[np.float64],
    numbers: NDArray[np.int64],
    n_folds: int,
    alpha: float,
) -> dict[int, NDArray[np.float64]]:
    """Predicts every fold with a head fitted on the other folds, for one member."""
    predictions: dict[int, NDArray[np.float64]] = {}
    for fold in range(n_folds):
        seen = numbers != fold
        held_out = numbers == fold
        if not bool(seen.any()) or not bool(held_out.any()):
            raise ValueError(f"fold {fold} has no photos on one of its sides")
        head = fit_head(HEAD_KIND, features[seen], labels[seen], {"alpha": alpha})
        predictions[fold] = np.asarray(head.predict(features[held_out]), dtype=np.float64)
    return predictions


def _out_of_fold(
    computed: dict[int, dict[int, NDArray[np.float64]]],
    labels: NDArray[np.float64],
    numbers: NDArray[np.int64],
    n_folds: int,
) -> NDArray[np.float64]:
    """Scores every photo with the one member whose encoder never saw it.

    Which member that is, is not written anywhere: the manifest names no fold, and assuming
    that member `i` was held out from fold `i` would be a guess that quietly turns the measured
    number into a claim about photos the model had already met. So it is measured instead. An
    encoder is worst on the fold it was not fine-tuned on, so the largest error in a member's
    row marks its own fold, and the marks have to form a permutation of the folds.

    Args:
        computed: Per member, per fold, the predictions of a head fitted without that fold.
        labels: Reference loads of every photo.
        numbers: Fold of every photo.
        n_folds: How many folds there are.

    Returns:
        One prediction per photo, from the member that was held out from its fold.

    Raises:
        ModelPackageError: If the members do not mark one fold each.
    """
    errors = np.zeros((len(computed), n_folds), dtype=np.float64)
    for member, by_fold in computed.items():
        for fold, predicted in by_fold.items():
            errors[member, fold] = float(np.abs(predicted - labels[numbers == fold]).mean())
    marked = errors.argmax(axis=1)
    if sorted(marked.tolist()) != list(range(n_folds)):
        raise ModelPackageError(
            "cannot tell which fold each member was held out from: the worst folds are "
            f"{marked.tolist()}, which is not one fold per member. Errors per member and fold:\n"
            + "\n".join("  " + " ".join(f"{value:6.2f}" for value in row) for row in errors)
        )
    logger.info("member to held-out fold: %s", marked.tolist())
    raw = np.zeros(labels.size, dtype=np.float64)
    for member, fold in enumerate(marked.tolist()):
        raw[numbers == fold] = computed[member][fold]
    return raw


def _depth(
    active: ModelPackage, samples: Sequence[Sample], cache_dir: Path, device: str
) -> NDArray[np.float64] | None:
    """Returns the depth grid of every photo, or None for a package that uses no depth.

    The checkpoint comes from the package, not from the hub: a package carries its own copy of
    the depth network precisely so that nothing has to be downloaded, and the worker on the
    machine of the organizers has no way out to the hub at all.

    The grids are kept on disk beside the embeddings and keyed by the hash of a photo, for the
    same reason: a retraining then pays only for the photos that are new since the last one.
    The cache of `ml/models/depth.py` lives inside one process, so every run of the worker would
    otherwise start by measuring the whole dataset again.

    Raises:
        ModelPackageError: If the package asks for a grid of another shape than the one the
            depth network pools into; the heads would be fitted on features of one width and
            the service would feed them another.
    """
    config = active.manifest.depth
    if config is None:
        return None
    if (config.rows, config.cols) != (GRID_ROWS, GRID_COLS):
        raise ModelPackageError(
            f"{active.root}: the package asks for a depth grid of {config.rows}x{config.cols} "
            f"and this build pools into {GRID_ROWS}x{GRID_COLS}"
        )
    source = active.root / config.model
    path = cache_dir / EMBEDDINGS_DIR / DEPTH_CACHE
    # The checkpoint itself identifies the cache: two packages may ship different depth weights,
    # and their grids must not be mixed under one name.
    weights = f"{config.model}/model.safetensors"
    fingerprint = f"{active.manifest.files.get(weights, DEPTH_MODEL)}:{GRID_ROWS}x{GRID_COLS}"
    cached = _read_grids(path, fingerprint)
    missing = [sample for sample in samples if sample.sha256 not in cached]
    if missing:
        logger.info(
            "depth grids for %d of %d photos on %s, from %s",
            len(missing),
            len(samples),
            device,
            source,
        )
        computed = depth_grids(missing, device, source)
        cached.update({sample.sha256: grid for sample, grid in zip(missing, computed, strict=True)})
        _write_grids(path, cached, fingerprint)
    return np.stack([cached[sample.sha256] for sample in samples]).astype(np.float64)


def _read_grids(path: Path, fingerprint: str) -> dict[str, NDArray[np.float32]]:
    """Reads the cached grids; a cache of another depth network is thrown away, not mixed in."""
    if not path.is_file():
        return {}
    with np.load(path) as stored:
        if str(stored["fingerprint"]) != fingerprint:
            logger.info("%s was built with another depth network and is dropped", path)
            return {}
        hashes, grids = stored["hashes"], stored["grids"]
    return dict(zip((str(value) for value in hashes), grids, strict=True))


def _write_grids(path: Path, cached: dict[str, NDArray[np.float32]], fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = sorted(cached)
    np.savez(
        path,
        hashes=np.array(hashes),
        grids=np.stack([cached[value] for value in hashes]),
        fingerprint=np.array(fingerprint),
    )


def _copy_package(source: Path, target: Path) -> Path:
    """Copies the whole package; the heads, the curves and the metrics are written over after.

    Copying beats assembling: the weights, the licences, the depth network, the reference of
    the out-of-distribution score and everything a package may gain later travel to the
    candidate without this module knowing about them.

    The weights are the two gigabytes of it and this refit never writes into them, so they are
    linked rather than copied and a run costs megabytes instead. A filesystem that refuses the
    link, such as two different mounts under Docker, falls back to copying.
    """
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Hidden names are left behind on purpose: a package path may not begin with a dot, and an
    # upload to a model hub drops its own `.cache` into the folder it is sending. Copying that
    # along would put paths into the manifest that the manifest itself refuses.
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(WEIGHTS_DIR, ".*"))
    weights = source / WEIGHTS_DIR
    if weights.is_dir():
        (target / WEIGHTS_DIR).mkdir()
        for file in sorted(weights.iterdir()):
            _link_or_copy(file, target / WEIGHTS_DIR / file.name)
    return target


def _link_or_copy(source: Path, target: Path) -> None:
    """Links a file that is only ever read, and copies it where linking is not allowed."""
    try:
        os.link(source, target)
    except OSError as error:
        logger.info("%s is copied rather than linked: %s", source.name, error)
        shutil.copyfile(source, target)


def _write_manifest(
    root: Path,
    active: ModelPackage,
    version: str,
    folds_file: Path,
    oof: Path,
    steps: Sequence[PostprocessStep],
    now: datetime | None,
) -> None:
    """Writes the manifest of the candidate: the active one with new heads, curves and metrics."""
    data = active.manifest.model_dump(mode="json")
    data["version"] = version
    data["created_at"] = format_utc(now or datetime.now(UTC))
    data["postprocess"] = [step.model_dump(mode="json") for step in steps]
    data.update(cv_block(oof, folds_file))
    data["files"] = hash_package_files(root)
    write_package(root, Manifest.model_validate(data))


def _calibrated(
    steps: Sequence[tuple[str, dict[str, float]]],
    raw: NDArray[np.float64],
    labels: NDArray[np.float64],
    numbers: NDArray[np.int64],
    facts: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Calibrates every fold with a chain fitted on the other folds.

    The same rule the pipeline follows: a curve never sees the photos it is judged on. Fitting
    one chain on everything would lift the measured number by more than the whole difference the
    gate is asked to rule on.
    """
    if not steps:
        return raw
    corrected = np.empty_like(raw)
    for fold in sorted(set(numbers.tolist())):
        held_out = numbers == fold
        apply = fit_chain(steps, raw[~held_out], labels[~held_out], _pick(facts, ~held_out))
        corrected[held_out] = apply(raw[held_out], _pick(facts, held_out))
    return corrected


def _pick(
    facts: dict[str, NDArray[np.float64]], mask: NDArray[np.bool_]
) -> dict[str, NDArray[np.float64]]:
    return {name: values[mask] for name, values in facts.items()}


def _aspects(samples: Sequence[Sample]) -> NDArray[np.float64]:
    """Returns the width over height of every photo, as the service measures it."""
    return np.array([_shape(sample) for sample in samples], dtype=np.float64)


def _shape(sample: Sample) -> float:
    image = read_image(sample.path)
    return image.width / image.height
