"""Stage that turns the dataset folders into one checked table."""

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from loadvision.ml.data.manifest import Sample, load_split
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult

MANIFEST_FILE: Final = "manifest.csv"

_COLUMNS: Final = ("image_id", "split", "sha256", "label", "group_id")


@dataclass(frozen=True, slots=True)
class BuildManifest:
    """Reads `train` and `test` into one table and writes it into the run folder.

    Everything downstream reads photos through this table, so a half-copied dataset fails in
    the first stage of a run instead of in the middle of training.
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
        path = ctx.run_dir / MANIFEST_FILE
        _write(path, train, test)
        return StageResult(
            artifacts={"manifest": path},
            counts={"train": len(train), "test": len(test)},
            metrics={"labelled_share": len(train) / (len(train) + len(test))},
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


def _write(path: Path, train: list[Sample], test: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(_COLUMNS)
        for split, samples in (("train", train), ("test", test)):
            for sample in samples:
                writer.writerow(
                    [
                        sample.image_id,
                        split,
                        sample.sha256,
                        "" if sample.label is None else f"{sample.label:g}",
                        sample.group_id or "",
                    ]
                )
