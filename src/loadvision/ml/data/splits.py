"""Group-aware stratified cross-validation splits."""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from loadvision.domain.timestamps import format_utc
from loadvision.ml.data.manifest import Sample

SCHEMA_VERSION: Final = 1
DEFAULT_N_FOLDS: Final = 5
DEFAULT_SEED: Final = 0

# Deciles plus 100 on its own: a fifth of the dataset is labelled exactly 100, so it only stays
# evenly spread across folds when it is a bin of its own.
DEFAULT_BIN_EDGES: Final = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

# Invariants 4 and 5 of docs/plan/folds-json.md, checked on a primary split.
MAX_FOLD_SIZE_GAP: Final = 0.05
MAX_BIN_SHARE_GAP: Final = 0.05

# How many ids an error message names before it stops listing them.
_REPORTED: Final = 5


@dataclass(frozen=True, slots=True, kw_only=True)
class Folds:
    """Which fold every training photo belongs to.

    Attributes:
        schema_version: Version of the file format.
        created_at: When the primary split was made, UTC.
        seed: Seed of the primary split; the canonical file uses 0.
        n_folds: Number of folds.
        bin_edges: Label bin edges, as used by `numpy.digitize`.
        grouping: How scenes were formed.
        dedup: Encoder, metric and threshold the near-duplicate search ran with.
        scene_groups: Scene to fold number.
        images: Photo id to scene.
        exceptions: Near-duplicate pairs deliberately left in different folds.
    """

    schema_version: int
    created_at: str
    seed: int
    n_folds: int
    bin_edges: tuple[int, ...]
    grouping: str
    dedup: Mapping[str, Any]
    scene_groups: Mapping[str, int]
    images: Mapping[str, str]
    exceptions: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Checks the invariants that hold for every version of the file.

        Raises:
            ValueError: If a photo names an unknown scene, or a fold number is out of range,
                or some fold ended up empty.
        """
        unknown = sorted(set(self.images.values()) - set(self.scene_groups))
        if unknown:
            raise ValueError(f"scenes used by photos but missing from scene_groups: {unknown}")
        used = set(self.scene_groups.values())
        if not used <= set(range(self.n_folds)):
            raise ValueError(f"fold numbers outside [0, {self.n_folds}): {sorted(used)}")
        if len(used) != self.n_folds:
            raise ValueError(f"only {len(used)} of {self.n_folds} folds are used")

    def fold_of(self, image_id: str) -> int:
        """Fold number of one photo.

        Args:
            image_id: Photo id.

        Returns:
            Fold number.

        Raises:
            KeyError: If the photo is not in the file.
        """
        return self.scene_groups[self.images[image_id]]

    def fold_indices(self, samples: Sequence[Sample]) -> list[int]:
        """Fold number of every sample, in the order of `samples`."""
        return [self.fold_of(sample.image_id) for sample in samples]

    def as_dict(self) -> dict[str, Any]:
        """Returns the file content as plain JSON types."""
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "bin_edges": list(self.bin_edges),
            "grouping": self.grouping,
            "dedup": dict(self.dedup),
            "scene_groups": dict(self.scene_groups),
            "images": dict(self.images),
            "exceptions": [dict(item) for item in self.exceptions],
        }


def label_bins(
    samples: Sequence[Sample], bin_edges: Sequence[int] = DEFAULT_BIN_EDGES
) -> list[int]:
    """Bin number of every label, used to keep rare labels spread across folds.

    Args:
        samples: Labelled photos.
        bin_edges: Edges passed to `numpy.digitize`.

    Returns:
        One bin number per sample.

    Raises:
        ValueError: If a sample has no label.
    """
    if any(sample.label is None for sample in samples):
        raise ValueError("folds are built from labelled photos only")
    labels = np.array([sample.label for sample in samples], dtype=np.float64)
    return [int(value) for value in np.digitize(labels, np.asarray(bin_edges))]


def build_folds(
    samples: Sequence[Sample],
    *,
    now: datetime,
    dedup: Mapping[str, Any],
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
    bin_edges: Sequence[int] = DEFAULT_BIN_EDGES,
) -> Folds:
    """Splits labelled photos into folds that keep whole scenes together.

    Photos of one scene always land in the same fold, because a model that saw a truck during
    training would recognize it on validation instead of measuring its load. Within that
    constraint the label bins are spread as evenly as the grouping allows.

    Args:
        samples: Labelled photos, each with a group.
        now: Time of the split, timezone-aware; stored as `created_at`.
        dedup: What the near-duplicate search ran with; stored as is.
        n_folds: Number of folds.
        seed: Seed of the split.
        bin_edges: Label bin edges.

    Returns:
        Fold assignment for every photo.

    Raises:
        ValueError: If a photo has no group or no label, or if the result is too unbalanced.
    """
    if any(sample.group_id is None for sample in samples):
        raise ValueError("folds are built from grouped photos only")
    scenes = [str(sample.group_id) for sample in samples]
    bins = label_bins(samples, bin_edges)

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scene_folds: dict[str, int] = {}
    for fold, (_, validation) in enumerate(splitter.split(samples, bins, scenes)):
        for index in validation:
            scene_folds[scenes[index]] = fold

    folds = Folds(
        schema_version=SCHEMA_VERSION,
        created_at=format_utc(now),
        seed=seed,
        n_folds=n_folds,
        bin_edges=tuple(bin_edges),
        grouping="scene_group",
        dedup=dict(dedup),
        scene_groups=dict(sorted(scene_folds.items())),
        images={sample.image_id: str(sample.group_id) for sample in sorted(samples, key=_by_id)},
        exceptions=(),
    )
    _check_balance(folds, samples, bins)
    return folds


def extend_folds(folds: Folds, samples: Sequence[Sample]) -> Folds:
    """Gives a fold to photos the file does not know yet, leaving every stored row alone.

    This is rule 3 of `docs/plan/folds-json.md`, and it is what makes retraining comparable:
    a candidate and the active model must be scored on the same split, so a new label may be
    added but an old one may never move. The assignment is deterministic and uses no seed, so
    two machines importing the same labels get the same file.

    A photo of a scene the file already knows joins that scene's fold. A photo of a new scene
    goes to the fold holding the fewest rows of its label bin, ties to the lower fold number;
    the bin of a scene comes from the median label of its photos, so a scene stays whole.

    The third case of rule 3, a photo gluing two known scenes, cannot arise while a scene is
    exactly an organizers' group: a photo names one group. It becomes reachable only if the
    near-duplicate search is turned back on, and `exceptions` is carried through untouched for
    that day.

    Args:
        folds: Fold assignment read from the file.
        samples: Every labelled photo, the stored ones included; labels come from here because
            the file holds no labels of its own.

    Returns:
        Fold assignment covering all of `samples`; the same object's rows keep their folds.

    Raises:
        ValueError: If a sample has no label or no group, if a stored photo is missing from
            `samples`, or if a stored photo changed its scene.
    """
    if any(sample.label is None for sample in samples):
        raise ValueError("folds are built from labelled photos only")
    if any(sample.group_id is None for sample in samples):
        raise ValueError("folds are built from grouped photos only")

    by_id = {sample.image_id: sample for sample in samples}
    lost = sorted(set(folds.images) - set(by_id))
    if lost:
        raise ValueError(
            f"{len(lost)} stored photos are missing from the manifest: {lost[:_REPORTED]}"
        )
    moved = sorted(
        image_id for image_id, scene in folds.images.items() if by_id[image_id].group_id != scene
    )
    if moved:
        raise ValueError(f"{len(moved)} stored photos changed scene: {moved[:_REPORTED]}")

    edges = np.asarray(folds.bin_edges)
    bin_of = {
        sample.image_id: int(np.digitize(sample.label, edges))
        for sample in samples
        if sample.label is not None
    }
    rows: dict[int, Counter[int]] = {fold: Counter() for fold in range(folds.n_folds)}
    for image_id in folds.images:
        rows[folds.fold_of(image_id)][bin_of[image_id]] += 1

    fresh = [sample for sample in samples if sample.image_id not in folds.images]
    scenes = dict(folds.scene_groups)
    images = dict(folds.images)
    for scene in sorted({str(sample.group_id) for sample in fresh}):
        members = [sample for sample in fresh if sample.group_id == scene]
        if scene in scenes:
            fold = scenes[scene]
        else:
            labels = [sample.label for sample in members if sample.label is not None]
            scene_bin = int(np.digitize(float(np.median(labels)), edges))
            fold = min(range(folds.n_folds), key=lambda number: (rows[number][scene_bin], number))
            scenes[scene] = fold
        for sample in members:
            images[sample.image_id] = scene
            rows[fold][bin_of[sample.image_id]] += 1

    return replace(
        folds,
        scene_groups=dict(sorted(scenes.items())),
        images=dict(sorted(images.items())),
    )


def check_unchanged(before: Folds, after: Folds) -> None:
    """Checks invariant 6: no stored photo and no stored scene changed its fold.

    Args:
        before: Fold assignment as it was read.
        after: Fold assignment about to be written.

    Raises:
        ValueError: If a stored photo lost its row or landed in another fold.
    """
    problems: list[str] = []
    for image_id in before.images:
        if image_id not in after.images:
            problems.append(f"{image_id} disappeared")
        elif after.fold_of(image_id) != before.fold_of(image_id):
            problems.append(
                f"{image_id} moved from fold {before.fold_of(image_id)} "
                f"to {after.fold_of(image_id)}"
            )
    if problems:
        raise ValueError(f"{len(problems)} stored rows moved: {problems[:_REPORTED]}")


def balance_report(
    folds: Folds, samples: Sequence[Sample], bins: Sequence[int] | None = None
) -> dict[str, float]:
    """Measures how even the split is, as the metrics of invariants 4 and 5.

    Args:
        folds: Fold assignment.
        samples: The same photos the folds were built from.
        bins: Label bins of the samples; computed from `folds.bin_edges` when omitted.

    Returns:
        `fold_size_gap`: largest relative deviation of a fold size from the mean size.
        `bin_share_gap`: largest deviation of a bin share inside a fold from its overall share.
    """
    numbers = folds.fold_indices(samples)
    sizes = Counter(numbers)
    mean_size = len(samples) / folds.n_folds
    size_gap = max(abs(sizes[fold] - mean_size) / mean_size for fold in range(folds.n_folds))

    bin_numbers = list(bins) if bins is not None else label_bins(samples, folds.bin_edges)
    overall = Counter(bin_numbers)
    share_gap = 0.0
    for fold in range(folds.n_folds):
        inside = Counter(b for b, f in zip(bin_numbers, numbers, strict=True) if f == fold)
        for value, count in overall.items():
            share_gap = max(share_gap, abs(inside[value] / sizes[fold] - count / len(samples)))
    return {"fold_size_gap": size_gap, "bin_share_gap": share_gap}


def write_folds(path: Path, folds: Folds) -> None:
    """Writes the file with sorted keys, so a rebuilt split gives a comparable diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(folds.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_folds(path: Path) -> Folds:
    """Reads a folds file.

    Args:
        path: File to read.

    Returns:
        Fold assignment.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the schema version is unknown or an invariant fails.
    """
    content: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    version = content.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {version!r}")
    return Folds(
        schema_version=version,
        created_at=content["created_at"],
        seed=content["seed"],
        n_folds=content["n_folds"],
        bin_edges=tuple(content["bin_edges"]),
        grouping=content["grouping"],
        dedup=content["dedup"],
        scene_groups=content["scene_groups"],
        images=content["images"],
        exceptions=tuple(content.get("exceptions", ())),
    )


def check_covers(folds: Folds, samples: Sequence[Sample]) -> None:
    """Checks invariant 1: the file and the manifest describe the same photos.

    Args:
        folds: Fold assignment.
        samples: Photos of the training manifest.

    Raises:
        ValueError: If either side has photos the other does not.
    """
    listed = set(folds.images)
    present = {sample.image_id for sample in samples}
    if listed != present:
        missing = sorted(present - listed)[:5]
        extra = sorted(listed - present)[:5]
        raise ValueError(
            f"folds and manifest disagree: {len(present - listed)} photos missing {missing}, "
            f"{len(listed - present)} unknown {extra}"
        )


def _by_id(sample: Sample) -> str:
    return sample.image_id


def _check_balance(folds: Folds, samples: Sequence[Sample], bins: Sequence[int]) -> None:
    report = balance_report(folds, samples, bins)
    if report["fold_size_gap"] > MAX_FOLD_SIZE_GAP:
        raise ValueError(
            f"fold sizes differ from the mean by {report['fold_size_gap']:.1%}, "
            f"more than {MAX_FOLD_SIZE_GAP:.0%}"
        )
    if report["bin_share_gap"] > MAX_BIN_SHARE_GAP:
        raise ValueError(
            f"a label bin share differs by {report['bin_share_gap']:.1%} between a fold and the "
            f"whole set, more than {MAX_BIN_SHARE_GAP:.0%}"
        )
