"""Stage that turns the dataset folders into one checked table."""

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from loadvision.imaging.quality import brightness
from loadvision.ml.data.manifest import Sample, load_split
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.ml.training.embedding_cache import read_image

MANIFEST_FILE: Final = "manifest.csv"

_COLUMNS: Final[tuple[str, ...]] = (
    "image_id",
    "split",
    "sha256",
    "label",
    "group_id",
    "width",
    "height",
    "aspect",
    "brightness",
)

# A photo darker than this is a night shot with no light in the body; a sixth of the dataset.
_DARK: Final = 60.0


@dataclass(frozen=True, slots=True)
class BuildManifest:
    """Reads `train` and `test` into one table and writes it into the run folder.

    Everything downstream reads photos through this table, so a half-copied dataset fails in
    the first stage of a run instead of in the middle of training.

    The table also carries what the report slices photos by: size, proportions and brightness.
    Slices are defined by what is known before a prediction, never by the error, otherwise
    measuring them would be circular.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "build_manifest"

    @property
    def version(self) -> str:
        """Stage version."""
        return "1"

    def inputs(self, ctx: RunContext) -> Mapping[str, str]:
        """What the result depends on."""
        return {"data_dir": str(ctx.data_dir)}

    def run(self, ctx: RunContext) -> StageResult:
        """Reads both splits and writes `manifest.csv`."""
        train = load_split(ctx.data_dir, "train")
        test = load_split(ctx.data_dir, "test")
        rows = [
            _describe(sample, split)
            for split, samples in (("train", train), ("test", test))
            for sample in samples
        ]
        path = ctx.run_dir / MANIFEST_FILE
        _write(path, rows)
        dark = sum(1 for row in rows if float(row["brightness"]) < _DARK)
        return StageResult(
            artifacts={"manifest": path},
            counts={"train": len(train), "test": len(test), "dark": dark},
            metrics={
                "labelled_share": len(train) / len(rows),
                "dark_share": dark / len(rows),
            },
        )

    def check(self, ctx: RunContext, result: StageResult) -> None:
        """Verifies the table covers every photo exactly once.

        Raises:
            StageCheckError: If the file is missing, a count is zero, or the rows and the
                counts disagree.
        """
        path = result.artifacts.get("manifest")
        if path is None or not path.is_file():
            raise StageCheckError(f"{self.name}: manifest was not written")
        expected = result.counts["train"] + result.counts["test"]
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != expected:
            raise StageCheckError(f"{self.name}: {len(rows)} rows for {expected} photos")
        if not result.counts["train"] or not result.counts["test"]:
            raise StageCheckError(f"{self.name}: a split came out empty, {dict(result.counts)}")
        ids = {row["image_id"] for row in rows}
        if len(ids) != expected:
            raise StageCheckError(f"{self.name}: image ids repeat in the manifest")


def _describe(sample: Sample, split: str) -> dict[str, str]:
    # The very function the service uses on an uploaded photo: two definitions of brightness
    # would put the report and the review policy on different numbers.
    decoded = read_image(sample.path)
    width, height = decoded.width, decoded.height
    return {
        "image_id": sample.image_id,
        "split": split,
        "sha256": sample.sha256,
        "label": "" if sample.label is None else f"{sample.label:g}",
        "group_id": sample.group_id or "",
        "width": str(width),
        "height": str(height),
        "aspect": f"{width / height:.4f}",
        "brightness": f"{brightness(decoded):.2f}",
    }


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(_COLUMNS)
        writer.writerows([row[column] for column in _COLUMNS] for row in rows)
