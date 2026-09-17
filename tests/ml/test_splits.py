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
    check_unchanged,
    extend_folds,
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


@pytest.mark.ml
def test_a_photo_of_a_known_scene_joins_that_scene_fold() -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP)
    extra = sample("img_005_new", 50.0, "grp_005")

    extended = extend_folds(folds, [*samples, extra])

    assert extended.fold_of("img_005_new") == folds.fold_of("img_005_0")


@pytest.mark.ml
def test_extending_never_moves_a_stored_row() -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP)
    fresh = [sample(f"img_new_{i:02d}", float(i % 11 * 10), f"grp_new_{i:02d}") for i in range(30)]

    extended = extend_folds(folds, [*samples, *fresh])

    check_unchanged(folds, extended)
    assert {i: extended.fold_of(i) for i in folds.images} == {
        i: folds.fold_of(i) for i in folds.images
    }
    assert len(extended.images) == len(folds.images) + 30


@pytest.mark.ml
def test_extending_keeps_the_creation_time_and_the_seed() -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP, seed=3)
    extra = sample("img_new", 40.0, "grp_new")

    extended = extend_folds(folds, [*samples, extra])

    # `created_at` records the primary split, so a later import must not touch it.
    assert (extended.created_at, extended.seed) == (folds.created_at, folds.seed)


@pytest.mark.ml
def test_a_new_scene_goes_where_its_bin_is_thinnest() -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP)
    bins = dict(zip([s.image_id for s in samples], label_bins(samples), strict=True))
    counts = Counter(folds.fold_of(image_id) for image_id, value in bins.items() if value == 5)
    thinnest = min(range(folds.n_folds), key=lambda fold: (counts[fold], fold))

    extended = extend_folds(folds, [*samples, sample("img_new", 55.0, "grp_new")])

    assert extended.fold_of("img_new") == thinnest


@pytest.mark.ml
def test_a_new_scene_stays_whole_and_is_placed_by_its_median_label() -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP)
    scene = [
        sample("img_a", 0.0, "grp_new"),
        sample("img_b", 50.0, "grp_new"),
        sample("img_c", 100.0, "grp_new"),
    ]

    extended = extend_folds(folds, [*samples, *scene])

    placed = {extended.fold_of(item.image_id) for item in scene}
    assert len(placed) == 1
    # The median of 0, 50 and 100 is 50, so the scene is placed by bin 5, not by 0 or 100.
    assert extended.fold_of("img_a") == extend_folds(
        folds, [*samples, sample("img_probe", 50.0, "grp_new")]
    ).fold_of("img_probe")


@pytest.mark.ml
def test_extending_is_deterministic() -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP)
    fresh = [sample(f"img_new_{i}", float(i * 7 % 101), f"grp_new_{i}") for i in range(12)]

    first = extend_folds(folds, [*samples, *fresh])
    again = extend_folds(folds, [*reversed(fresh), *reversed(samples)])

    assert first.as_dict() == again.as_dict()


@pytest.mark.ml
def test_a_stored_photo_missing_from_the_manifest_is_rejected() -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP)

    with pytest.raises(ValueError, match="missing from the manifest"):
        extend_folds(folds, samples[1:])


@pytest.mark.ml
def test_a_stored_photo_that_changed_scene_is_rejected() -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP)
    renamed = [sample(samples[0].image_id, samples[0].label, "grp_999"), *samples[1:]]

    with pytest.raises(ValueError, match="changed scene"):
        extend_folds(folds, renamed)


@pytest.mark.ml
def test_check_unchanged_catches_a_moved_row() -> None:
    samples = dataset()
    folds = build_folds(samples, now=NOW, dedup=DEDUP)
    moved = {image_id: scene for image_id, scene in folds.images.items()}
    other = {name: (fold + 1) % folds.n_folds for name, fold in folds.scene_groups.items()}

    with pytest.raises(ValueError, match="moved from fold"):
        check_unchanged(folds, Folds(**{**folds.as_dict(), "scene_groups": other, "images": moved}))


@pytest.mark.data
@pytest.mark.skipif(not CANONICAL.is_file(), reason="canonical folds file is not built")
def test_thirty_new_labels_leave_the_real_folds_alone() -> None:
    samples = load_split(DATA_DIR, "train")
    folds = read_folds(CANONICAL)
    fresh = [
        sample(f"img_confirmed_{i:02d}", float(i % 11 * 10), f"grp_confirmed_{i:02d}")
        for i in range(30)
    ]

    extended = extend_folds(folds, [*samples, *fresh])

    check_unchanged(folds, extended)
    check_covers(extended, [*samples, *fresh])
    assert len(extended.images) == len(samples) + 30
    report = balance_report(extended, [*samples, *fresh])
    # Invariants 4 and 5 are metrics when extending, not errors: the numbers go to the run.
    assert report["fold_size_gap"] < 0.1
