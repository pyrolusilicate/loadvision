from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from loadvision.ml.data.manifest import Sample, load_split
from loadvision.ml.data.splits import (
    DEFAULT_BIN_EDGES,
    SCHEMA_VERSION,
    Folds,
    balance_report,
    build_folds,
    check_covers,
    label_bins,
    read_folds,
    write_folds,
)

DATA_DIR = Path("data")
CANONICAL = Path("configs/splits/folds.json")
NOW = datetime(2026, 9, 17, 9, 12, 3, tzinfo=UTC)
DEDUP = {"encoder": "siglip2_vitb16", "metric": "cosine", "threshold": 0.98}


def sample(image_id: str, label: float | None, group_id: str | None) -> Sample:
    return Sample(
        image_id=image_id,
        path=Path(f"{image_id}.jpg"),
        sha256="0" * 64,
        label=label,
        group_id=group_id,
    )


def dataset(groups: int = 220, per_group: int = 2) -> list[Sample]:
    """Builds photos whose labels walk over the whole range, two photos per group."""
    samples: list[Sample] = []
    for index in range(groups):
        label = float((index * 10) % 110)
        for shot in range(per_group):
            samples.append(sample(f"img_{index:03d}_{shot}", label, f"grp_{index:03d}"))
    return samples


def folds_of(folds: Folds, samples: list[Sample]) -> dict[str, set[int]]:
    by_group: dict[str, set[int]] = defaultdict(set)
    for item in samples:
        by_group[str(item.group_id)].add(folds.fold_of(item.image_id))
    return by_group


@pytest.mark.ml
def test_every_photo_gets_a_fold_and_groups_stay_together() -> None:
    samples = dataset()

    folds = build_folds(samples, now=NOW, dedup=DEDUP)

    assert set(folds.images) == {s.image_id for s in samples}
    assert all(len(seen) == 1 for seen in folds_of(folds, samples).values())
    assert set(folds.scene_groups.values()) == set(range(folds.n_folds))


@pytest.mark.ml
def test_same_seed_gives_the_same_split() -> None:
    samples = dataset()

    first = build_folds(samples, now=NOW, dedup=DEDUP, seed=0)
    second = build_folds(samples, now=NOW, dedup=DEDUP, seed=0)

    assert first.scene_groups == second.scene_groups


@pytest.mark.ml
def test_folds_are_evenly_sized() -> None:
    samples = dataset()

    folds = build_folds(samples, now=NOW, dedup=DEDUP)

    report = balance_report(folds, samples)
    assert report["fold_size_gap"] <= 0.05
    assert report["bin_share_gap"] <= 0.05


@pytest.mark.ml
def test_label_100_goes_to_its_own_bin() -> None:
    samples = [sample("img_a", 99.0, "g"), sample("img_b", 100.0, "g")]

    assert label_bins(samples, DEFAULT_BIN_EDGES) == [9, 10]


@pytest.mark.ml
def test_unlabelled_photos_are_rejected() -> None:
    samples = [*dataset(), sample("img_x", None, "grp_x")]

    with pytest.raises(ValueError, match="labelled photos only"):
        build_folds(samples, now=NOW, dedup=DEDUP)


@pytest.mark.ml
def test_ungrouped_photos_are_rejected() -> None:
    samples = [*dataset(), sample("img_x", 50.0, None)]

    with pytest.raises(ValueError, match="grouped photos only"):
        build_folds(samples, now=NOW, dedup=DEDUP)


@pytest.mark.ml
def test_photo_pointing_at_an_unknown_scene_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing from scene_groups"):
        Folds(
            schema_version=SCHEMA_VERSION,
            created_at="2026-09-17T09:12:03Z",
            seed=0,
            n_folds=2,
            bin_edges=DEFAULT_BIN_EDGES,
            grouping="scene_group",
            dedup=DEDUP,
            scene_groups={"grp_a": 0, "grp_b": 1},
            images={"img_a": "grp_ghost"},
        )


@pytest.mark.ml
def test_empty_fold_is_rejected() -> None:
    with pytest.raises(ValueError, match="of 5 folds are used"):
        Folds(
            schema_version=SCHEMA_VERSION,
            created_at="2026-09-17T09:12:03Z",
            seed=0,
            n_folds=5,
            bin_edges=DEFAULT_BIN_EDGES,
            grouping="scene_group",
            dedup=DEDUP,
            scene_groups={"grp_a": 0, "grp_b": 1},
            images={"img_a": "grp_a", "img_b": "grp_b"},
        )


@pytest.mark.ml
def test_write_and_read_round_trip(tmp_path: Path) -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP)
    path = tmp_path / "folds.json"

    write_folds(path, folds)
    restored = read_folds(path)

    assert restored.as_dict() == folds.as_dict()
    assert restored.created_at == "2026-09-17T09:12:03Z"


@pytest.mark.ml
def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    folds = build_folds(dataset(), now=NOW, dedup=DEDUP)
    path = tmp_path / "folds.json"
    write_folds(path, folds)
    broken = path.read_text(encoding="utf-8").replace('"schema_version": 1', '"schema_version": 99')
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported schema_version"):
        read_folds(path)


@pytest.mark.ml
def test_folds_that_do_not_cover_the_manifest_are_rejected() -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP)

    with pytest.raises(ValueError, match="folds and manifest disagree"):
        check_covers(folds, [*samples, sample("img_new", 50.0, "grp_new")])


@pytest.mark.data
@pytest.mark.skipif(not CANONICAL.is_file(), reason="canonical folds file is not built")
def test_canonical_folds_match_the_real_dataset() -> None:
    folds = read_folds(CANONICAL)
    samples = load_split(DATA_DIR, "train")

    check_covers(folds, samples)
    report = balance_report(folds, samples)
    assert report["fold_size_gap"] <= 0.05
    assert report["bin_share_gap"] <= 0.05
    assert all(len(seen) == 1 for seen in folds_of(folds, samples).values())
    sizes = Counter(folds.fold_indices(samples))
    assert min(sizes.values()) >= len(samples) // folds.n_folds - 1
