"""Building a training dataset that carries the labels people confirmed in the service."""

import csv
import hashlib
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from loadvision.application.ports import ImageStorage, LabeledImage
from loadvision.imaging.prepared_image import MediaType
from loadvision.imaging.validator import prepare_image
from loadvision.ml.data.manifest import IMAGE_SUFFIX, Sample, Split, load_split
from loadvision.ml.training.embedding_cache import TRAINING_LIMITS

logger = logging.getLogger(__name__)

# A confirmed photo gets its id and its scene from its own hash, so importing the same label
# twice, on two machines or after a restart, always names the same row.
ID_PREFIX: Final = "img_"
SCENE_PREFIX: Final = "grp_"
_HASH_CHARS: Final = 20

_SUFFIXES: Final = {
    MediaType.JPEG: ".jpg",
    MediaType.PNG: ".png",
    MediaType.WEBP: ".webp",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ImportReport:
    """What an import did, for the run record and for the training outcome.

    Attributes:
        added: Confirmed photos that were not in the base dataset.
        relabelled: Photos already in the base dataset whose label the confirmation replaced.
        unchanged: Confirmed labels that repeat the label the dataset already carries.
        missing: Labels whose bytes the storage does not hold; they are left out.
    """

    added: int
    relabelled: int
    unchanged: int
    missing: tuple[str, ...] = ()

    @property
    def used(self) -> int:
        """How many confirmed labels made it into the dataset."""
        return self.added + self.relabelled + self.unchanged

    def as_counts(self) -> dict[str, int]:
        """Counts for `StageResult` and `TrainingOutcome`."""
        return {
            "added": self.added,
            "relabelled": self.relabelled,
            "unchanged": self.unchanged,
            "missing": len(self.missing),
        }


def build_dataset(
    base_dir: Path,
    target_dir: Path,
    labels: Sequence[LabeledImage],
    storage: ImageStorage,
) -> ImportReport:
    """Writes a dataset that is the base one plus the confirmed labels.

    The base photos are linked, not copied: a retraining run must not duplicate a gigabyte of
    JPEG, and it must not touch `data/` either, because a run has to be repeatable from the
    same inputs. The result is an ordinary dataset folder, so `load_split` reads it with the
    checks it always runs and nothing downstream knows the difference.

    A confirmed photo that is already in the base dataset keeps its id, its file and its scene
    and only changes its label. A new one becomes a scene of its own: two photos confirmed
    through the service are not known to show the same truck, and guessing that they do would
    put the same scene on both sides of a fold.

    Args:
        base_dir: Dataset root with `train/` and `test/`.
        target_dir: Folder to build; created, and emptied first if it exists.
        labels: Confirmed labels, each with the hash of its photo.
        storage: Where the confirmed photos' bytes live.

    Returns:
        What the import did.

    Raises:
        FileNotFoundError: If the base dataset is missing a split.
        ValueError: If the base dataset is inconsistent, if a label repeats a photo, or if the
            bytes the storage returns do not hash to the id they were asked for.
    """
    if len({label.image_sha256 for label in labels}) != len(labels):
        raise ValueError("labels must not repeat a photo")

    base_train = load_split(base_dir, "train")
    by_hash = {sample.sha256: sample for sample in base_train}
    _link_split(base_dir, target_dir, "test")

    images_dir = target_dir / "train" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = {sample.image_id: sample for sample in base_train}
    for sample in base_train:
        _place(sample.path, images_dir / sample.path.name)

    added, relabelled, unchanged, missing = 0, 0, 0, []
    for label in sorted(labels, key=lambda item: item.image_sha256):
        known = by_hash.get(label.image_sha256)
        if known is not None:
            if known.label == label.load_pct.value:
                unchanged += 1
                continue
            rows[known.image_id] = _relabelled(known, label.load_pct.value)
            relabelled += 1
            continue
        raw = storage.load(label.image_sha256)
        if raw is None:
            missing.append(label.image_sha256)
            continue
        rows[_image_id(label.image_sha256)] = _store(images_dir, label, raw)
        added += 1

    _write_train(target_dir / "train", rows)
    if missing:
        logger.warning("%d confirmed labels have no photo in the storage", len(missing))
    return ImportReport(
        added=added, relabelled=relabelled, unchanged=unchanged, missing=tuple(missing)
    )


def _image_id(sha256: str) -> str:
    return f"{ID_PREFIX}{sha256[:_HASH_CHARS]}"


def _scene(sha256: str) -> str:
    return f"{SCENE_PREFIX}{sha256[:_HASH_CHARS]}"


def _relabelled(sample: Sample, label: float) -> Sample:
    return Sample(
        image_id=sample.image_id,
        path=sample.path,
        sha256=sample.sha256,
        label=label,
        group_id=sample.group_id,
    )


def _store(images_dir: Path, label: LabeledImage, raw: bytes) -> Sample:
    actual = hashlib.sha256(raw).hexdigest()
    if actual != label.image_sha256:
        raise ValueError(
            f"storage returned bytes with sha256 {actual} for photo {label.image_sha256}"
        )
    # Decoded through the service validator, so the format on disk is the real one and a photo
    # the model could never read fails here, naming itself, instead of inside training.
    image = prepare_image(raw, TRAINING_LIMITS)
    image_id = _image_id(label.image_sha256)
    path = images_dir / f"{image_id}{_SUFFIXES.get(image.media_type, IMAGE_SUFFIX)}"
    path.write_bytes(raw)
    return Sample(
        image_id=image_id,
        path=path,
        sha256=label.image_sha256,
        label=label.load_pct.value,
        group_id=_scene(label.image_sha256),
    )


def _link_split(base_dir: Path, target_dir: Path, split: Split) -> None:
    source = base_dir / split
    target = target_dir / split
    (target / "images").mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.csv"):
        shutil.copyfile(path, target / path.name)
    for path in (source / "images").iterdir():
        if path.is_file():
            _place(path, target / "images" / path.name)


def _place(source: Path, target: Path) -> None:
    """Links a base photo into the new dataset, copying only if the link cannot be made."""
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _write_train(train_dir: Path, rows: dict[str, Sample]) -> None:
    labels = [(image_id, rows[image_id]) for image_id in sorted(rows)]
    with (train_dir / "train.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", "load_pct"))
        writer.writerows(
            (image_id, f"{sample.label:g}")
            for image_id, sample in labels
            if sample.label is not None
        )
    with (train_dir / "train_groups.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", "group_id"))
        writer.writerows((image_id, sample.group_id or "") for image_id, sample in labels)
