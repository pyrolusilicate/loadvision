import hashlib
from pathlib import Path

import pytest

from loadvision.ml.data.manifest import Sample, load_split
from tests import image_files

DATA_DIR = Path("data")


def write_split(
    root: Path,
    split: str,
    rows: list[tuple[str, str]],
    *,
    groups: list[tuple[str, str]] | None = None,
    images: list[str] | None = None,
) -> Path:
    """Builds a tiny dataset on disk with real, decodable photos.

    The bytes used to be arbitrary, because reading the split only hashed them. The manifest
    stage now decodes every photo to record its size, proportions and brightness, so a fake
    file fails there instead; each photo gets its own size, which keeps their hashes distinct.
    """
    split_dir = root / split
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)

    header = "image_id,load_pct" if split == "train" else "image_id"
    lines = [header] + [",".join(row) if split == "train" else row[0] for row in rows]
    (split_dir / f"{split}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if groups is not None:
        group_lines = ["image_id,group_id"] + [f"{a},{b}" for a, b in groups]
        (split_dir / "train_groups.csv").write_text("\n".join(group_lines) + "\n", encoding="utf-8")

    for index, image_id in enumerate(images if images is not None else [row[0] for row in rows]):
        (images_dir / f"{image_id}.jpg").write_bytes(image_files.jpeg(40 + index, 30 + index))
    return root


@pytest.mark.ml
def test_loads_train_split_sorted_with_labels_groups_and_hashes(tmp_path: Path) -> None:
    write_split(
        tmp_path,
        "train",
        [("img_b", "100.0"), ("img_a", "0.0")],
        groups=[("img_a", "grp_1"), ("img_b", "grp_1")],
    )

    samples = load_split(tmp_path, "train")

    assert [s.image_id for s in samples] == ["img_a", "img_b"]
    assert [s.label for s in samples] == [0.0, 100.0]
    assert {s.group_id for s in samples} == {"grp_1"}
    photo = tmp_path / "train" / "images" / "img_a.jpg"
    assert samples[0].sha256 == hashlib.sha256(photo.read_bytes()).hexdigest()
    assert samples[0].path == photo


@pytest.mark.ml
def test_test_split_has_no_labels_and_no_groups(tmp_path: Path) -> None:
    write_split(tmp_path, "test", [("img_a", "")])

    samples = load_split(tmp_path, "test")

    photo = tmp_path / "test" / "images" / "img_a.jpg"
    assert samples == [
        Sample(
            image_id="img_a",
            path=photo,
            sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),
            label=None,
            group_id=None,
        )
    ]


@pytest.mark.ml
def test_photo_listed_but_missing_from_disk_is_rejected(tmp_path: Path) -> None:
    write_split(
        tmp_path,
        "train",
        [("img_a", "10.0"), ("img_b", "20.0")],
        groups=[("img_a", "grp_1"), ("img_b", "grp_2")],
        images=["img_a"],
    )

    with pytest.raises(ValueError, match="img_b"):
        load_split(tmp_path, "train")


@pytest.mark.ml
def test_photo_on_disk_but_missing_from_csv_is_rejected(tmp_path: Path) -> None:
    write_split(
        tmp_path,
        "train",
        [("img_a", "10.0")],
        groups=[("img_a", "grp_1")],
        images=["img_a", "img_stray"],
    )

    with pytest.raises(ValueError, match="img_stray"):
        load_split(tmp_path, "train")


@pytest.mark.ml
def test_training_photo_without_a_group_is_rejected(tmp_path: Path) -> None:
    write_split(tmp_path, "train", [("img_a", "10.0")], groups=[("img_other", "grp_1")])

    with pytest.raises(ValueError, match="without a group"):
        load_split(tmp_path, "train")


@pytest.mark.ml
def test_group_for_a_photo_missing_from_the_csv_is_rejected(tmp_path: Path) -> None:
    write_split(
        tmp_path,
        "train",
        [("img_a", "10.0")],
        groups=[("img_a", "grp_1"), ("img_ghost", "grp_2")],
    )

    with pytest.raises(ValueError, match="img_ghost"):
        load_split(tmp_path, "train")


@pytest.mark.ml
def test_label_that_is_not_a_number_names_the_photo(tmp_path: Path) -> None:
    write_split(tmp_path, "train", [("img_a", "полный")], groups=[("img_a", "grp_1")])

    with pytest.raises(ValueError, match="label of img_a is not a number"):
        load_split(tmp_path, "train")


@pytest.mark.ml
def test_duplicate_image_id_is_rejected(tmp_path: Path) -> None:
    write_split(
        tmp_path,
        "train",
        [("img_a", "10.0"), ("img_a", "20.0")],
        groups=[("img_a", "grp_1")],
    )

    with pytest.raises(ValueError, match="duplicate image id"):
        load_split(tmp_path, "train")


@pytest.mark.ml
def test_unexpected_columns_are_rejected(tmp_path: Path) -> None:
    write_split(tmp_path, "train", [("img_a", "10.0")], groups=[("img_a", "grp_1")])
    (tmp_path / "train" / "train.csv").write_text("image_id,load\nimg_a,10.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected columns"):
        load_split(tmp_path, "train")


@pytest.mark.ml
def test_label_outside_the_allowed_range_is_rejected(tmp_path: Path) -> None:
    write_split(tmp_path, "train", [("img_a", "150.0")], groups=[("img_a", "grp_1")])

    with pytest.raises(ValueError, match="within \\[0, 100\\]"):
        load_split(tmp_path, "train")


@pytest.mark.ml
def test_missing_images_folder_is_reported_as_missing_file(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()

    with pytest.raises(FileNotFoundError, match="images folder not found"):
        load_split(tmp_path, "train")


@pytest.mark.data
@pytest.mark.skipif(not (DATA_DIR / "train").is_dir(), reason="dataset is not present")
def test_real_dataset_loads_with_expected_counts() -> None:
    train = load_split(DATA_DIR, "train")
    test = load_split(DATA_DIR, "test")

    assert len(train) == 716
    assert len(test) == 307
    assert all(s.label is not None and s.group_id is not None for s in train)
    assert len({s.sha256 for s in train + test}) == len(train) + len(test)
