"""Dataset manifest: image ids, hashes, labels and groups."""

import csv
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from loadvision.domain.assessment import LoadPercent
from loadvision.ml.packaging.model_package import file_sha256

Split = Literal["train", "test"]

IMAGE_SUFFIX: Final = ".jpg"

_LABEL_COLUMN: Final = "load_pct"
_ID_COLUMN: Final = "image_id"
_GROUP_COLUMN: Final = "group_id"
_REPORTED_IDS: Final = 5


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    """One dataset photo with everything the pipeline needs to know about it.

    Attributes:
        image_id: Id from the csv, also the file name without the suffix.
        path: Photo on disk.
        sha256: Hash of the file bytes; the service identifies an uploaded photo the same way,
            so a photo can be matched across the dataset and the database.
        label: Expert load percent, or None for the test split.
        group_id: Shooting the photo belongs to, or None for the test split. Photos of one
            group must never be split between training and validation.
    """

    image_id: str
    path: Path
    sha256: str
    label: float | None
    group_id: str | None

    def __post_init__(self) -> None:
        """Validates the label against the domain rule.

        Raises:
            ValueError: If the label is not a finite number within [0, 100].
        """
        if self.label is not None:
            LoadPercent(self.label)


def load_split(data_dir: Path, split: Split) -> list[Sample]:
    """Reads one dataset split into a checked list of samples, sorted by image id.

    Every photo listed in the csv must exist on disk and every photo on disk must be listed,
    so a half-copied dataset fails here and not in the middle of training.

    Args:
        data_dir: Dataset root that holds `train/` and `test/`.
        split: Which split to read.

    Returns:
        One `Sample` per photo of the split.

    Raises:
        FileNotFoundError: If a csv or the images folder is missing.
        ValueError: If a csv has unexpected columns or duplicate ids, if the csv and the images
            folder disagree, if a training photo has no group, or if a label is out of range.
    """
    split_dir = data_dir / split
    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images folder not found: {images_dir}")

    labels = _read_labels(split_dir / f"{split}.csv", split)
    groups = _read_groups(split_dir / "train_groups.csv") if split == "train" else {}
    files = {path.stem: path for path in images_dir.glob(f"*{IMAGE_SUFFIX}")}

    _check_ids_match(labels.keys(), files.keys(), images_dir)
    if split == "train":
        _check_no_missing(labels.keys(), groups.keys(), "photos without a group")
        _check_no_missing(groups.keys(), labels.keys(), "grouped photos missing from the csv")

    return [
        Sample(
            image_id=image_id,
            path=files[image_id],
            sha256=file_sha256(files[image_id]),
            label=labels[image_id],
            group_id=groups.get(image_id),
        )
        for image_id in sorted(labels)
    ]


def _read_table(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(columns):
            raise ValueError(f"{path}: expected columns {list(columns)}, got {reader.fieldnames}")
        return list(reader)


def _read_labels(path: Path, split: Split) -> dict[str, float | None]:
    columns = (_ID_COLUMN, _LABEL_COLUMN) if split == "train" else (_ID_COLUMN,)
    labels: dict[str, float | None] = {}
    for row in _read_table(path, columns):
        image_id = row[_ID_COLUMN]
        if image_id in labels:
            raise ValueError(f"{path}: duplicate image id {image_id}")
        if split == "train":
            labels[image_id] = _parse_label(row[_LABEL_COLUMN], path, image_id)
        else:
            labels[image_id] = None
    return labels


def _parse_label(raw: str, path: Path, image_id: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{path}: label of {image_id} is not a number: {raw!r}") from None


def _read_groups(path: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    for row in _read_table(path, (_ID_COLUMN, _GROUP_COLUMN)):
        image_id = row[_ID_COLUMN]
        if image_id in groups:
            raise ValueError(f"{path}: duplicate image id {image_id}")
        groups[image_id] = row[_GROUP_COLUMN]
    return groups


def _check_ids_match(listed: Collection[str], on_disk: Collection[str], images_dir: Path) -> None:
    _check_no_missing(listed, on_disk, f"photos listed in the csv but missing from {images_dir}")
    _check_no_missing(on_disk, listed, f"photos in {images_dir} but missing from the csv")


def _check_no_missing(expected: Collection[str], present: Collection[str], what: str) -> None:
    missing = sorted(set(expected) - set(present))
    if not missing:
        return
    shown = ", ".join(missing[:_REPORTED_IDS])
    tail = f" and {len(missing) - _REPORTED_IDS} more" if len(missing) > _REPORTED_IDS else ""
    raise ValueError(f"{len(missing)} {what}: {shown}{tail}")
