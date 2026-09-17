import csv
import hashlib
from pathlib import Path

import pytest

from loadvision.application.ports import LabeledImage
from loadvision.domain.assessment import LoadPercent
from loadvision.ml.data.confirmed import build_dataset
from loadvision.ml.data.manifest import load_split
from tests import image_files
from tests.fakes.image_storage import InMemoryImageStorage

BASE_PHOTOS = 6


def make_base(root: Path) -> Path:
    """Writes a small dataset shaped like the organizers': train with groups, test without."""
    for split in ("train", "test"):
        (root / split / "images").mkdir(parents=True)
    train_rows, group_rows = [], []
    for index in range(BASE_PHOTOS):
        image_id = f"img_base_{index:02d}"
        (root / "train" / "images" / f"{image_id}.jpg").write_bytes(
            image_files.jpeg(40 + index, 30)
        )
        train_rows.append((image_id, float(index * 20 % 101)))
        group_rows.append((image_id, f"grp_{index // 2:02d}"))
    (root / "train" / "train.csv").write_text(
        "image_id,load_pct\n" + "".join(f"{i},{v:g}\n" for i, v in train_rows), encoding="utf-8"
    )
    (root / "train" / "train_groups.csv").write_text(
        "image_id,group_id\n" + "".join(f"{i},{g}\n" for i, g in group_rows), encoding="utf-8"
    )
    (root / "test" / "images" / "img_test_0.jpg").write_bytes(image_files.jpeg(40, 30))
    (root / "test" / "test.csv").write_text("image_id\nimg_test_0\n", encoding="utf-8")
    return root


def confirmed(storage: InMemoryImageStorage, raw: bytes, load: float) -> LabeledImage:
    sha = hashlib.sha256(raw).hexdigest()
    storage.files[sha] = raw
    return LabeledImage(image_sha256=sha, load_pct=LoadPercent(load))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return make_base(tmp_path / "data")


@pytest.mark.ml
def test_a_new_photo_is_added_as_a_scene_of_its_own(base: Path, tmp_path: Path) -> None:
    storage = InMemoryImageStorage()
    label = confirmed(storage, image_files.jpeg(64, 48), 55.0)

    report = build_dataset(base, tmp_path / "built", [label], storage)

    samples = load_split(tmp_path / "built", "train")
    assert report.as_counts() == {"added": 1, "relabelled": 0, "unchanged": 0, "missing": 0}
    assert len(samples) == BASE_PHOTOS + 1
    [fresh] = [s for s in samples if s.sha256 == label.image_sha256]
    assert fresh.label == 55.0
    # Two photos confirmed through the service are not known to show the same truck, so each
    # one is its own scene and cannot land on both sides of a fold.
    assert fresh.group_id not in {s.group_id for s in samples if s.sha256 != label.image_sha256}


@pytest.mark.ml
def test_a_photo_already_in_the_dataset_is_relabelled_in_place(base: Path, tmp_path: Path) -> None:
    storage = InMemoryImageStorage()
    known = load_split(base, "train")[0]
    label = LabeledImage(image_sha256=known.sha256, load_pct=LoadPercent(77.0))

    report = build_dataset(base, tmp_path / "built", [label], storage)

    samples = {s.image_id: s for s in load_split(tmp_path / "built", "train")}
    assert report.as_counts() == {"added": 0, "relabelled": 1, "unchanged": 0, "missing": 0}
    assert len(samples) == BASE_PHOTOS
    # Same row, same scene, new label: a confirmation corrects the answer, it does not add a photo.
    assert samples[known.image_id].label == 77.0
    assert samples[known.image_id].group_id == known.group_id


@pytest.mark.ml
def test_a_confirmation_that_repeats_the_known_label_changes_nothing(
    base: Path, tmp_path: Path
) -> None:
    storage = InMemoryImageStorage()
    known = load_split(base, "train")[0]
    label = LabeledImage(image_sha256=known.sha256, load_pct=LoadPercent(known.label or 0.0))

    report = build_dataset(base, tmp_path / "built", [label], storage)

    assert report.as_counts() == {"added": 0, "relabelled": 0, "unchanged": 1, "missing": 0}


@pytest.mark.ml
def test_a_label_whose_photo_the_storage_lost_is_reported_not_raised(
    base: Path, tmp_path: Path
) -> None:
    storage = InMemoryImageStorage()
    present = confirmed(storage, image_files.jpeg(64, 48), 30.0)
    gone = LabeledImage(image_sha256="a" * 64, load_pct=LoadPercent(10.0))

    report = build_dataset(base, tmp_path / "built", [present, gone], storage)

    # One broken row must not cancel a retraining that has every other label in hand.
    assert report.as_counts() == {"added": 1, "relabelled": 0, "unchanged": 0, "missing": 1}
    assert report.missing == ("a" * 64,)


@pytest.mark.ml
def test_bytes_that_do_not_match_their_hash_are_refused(base: Path, tmp_path: Path) -> None:
    storage = InMemoryImageStorage()
    storage.files["b" * 64] = image_files.jpeg(40, 30)
    label = LabeledImage(image_sha256="b" * 64, load_pct=LoadPercent(50.0))

    with pytest.raises(ValueError, match="sha256"):
        build_dataset(base, tmp_path / "built", [label], storage)


@pytest.mark.ml
@pytest.mark.parametrize(
    ("raw", "suffix"),
    [(image_files.png(64, 48), ".png"), (image_files.webp(64, 48), ".webp")],
)
def test_a_photo_keeps_the_format_it_was_uploaded_in(
    base: Path, tmp_path: Path, raw: bytes, suffix: str
) -> None:
    storage = InMemoryImageStorage()
    label = confirmed(storage, raw, 40.0)

    build_dataset(base, tmp_path / "built", [label], storage)

    samples = {s.sha256: s for s in load_split(tmp_path / "built", "train")}
    assert samples[label.image_sha256].path.suffix == suffix


@pytest.mark.ml
def test_the_base_dataset_is_left_untouched(base: Path, tmp_path: Path) -> None:
    storage = InMemoryImageStorage()
    label = confirmed(storage, image_files.jpeg(64, 48), 90.0)
    before = (base / "train" / "train.csv").read_text(encoding="utf-8")

    build_dataset(base, tmp_path / "built", [label], storage)

    assert (base / "train" / "train.csv").read_text(encoding="utf-8") == before
    assert len(load_split(base, "train")) == BASE_PHOTOS


@pytest.mark.ml
def test_the_test_split_comes_along_unchanged(base: Path, tmp_path: Path) -> None:
    storage = InMemoryImageStorage()

    build_dataset(base, tmp_path / "built", [], storage)

    assert [s.image_id for s in load_split(tmp_path / "built", "test")] == ["img_test_0"]


@pytest.mark.ml
def test_the_same_labels_always_build_the_same_dataset(base: Path, tmp_path: Path) -> None:
    storage = InMemoryImageStorage()
    labels = [confirmed(storage, image_files.jpeg(60 + i, 45), float(i * 11)) for i in range(4)]

    build_dataset(base, tmp_path / "one", labels, storage)
    build_dataset(base, tmp_path / "two", list(reversed(labels)), storage)

    def rows(root: Path) -> list[list[str]]:
        with (root / "train" / "train.csv").open(newline="", encoding="utf-8") as file:
            return list(csv.reader(file))

    assert rows(tmp_path / "one") == rows(tmp_path / "two")


@pytest.mark.ml
def test_the_same_photo_confirmed_twice_is_refused(base: Path, tmp_path: Path) -> None:
    storage = InMemoryImageStorage()
    label = confirmed(storage, image_files.jpeg(64, 48), 20.0)
    other = LabeledImage(image_sha256=label.image_sha256, load_pct=LoadPercent(80.0))

    with pytest.raises(ValueError, match="repeat a photo"):
        build_dataset(base, tmp_path / "built", [label, other], storage)
