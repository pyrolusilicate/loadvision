"""The answer file for the platform: one load per test photo, in the format it accepts."""

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

SUBMISSION_FILE: Final = "submission.csv"
COLUMNS: Final = ("image_id", "load_pct")
DECIMALS: Final = 1


def write_submission(
    path: Path, image_ids: Sequence[str], loads: Sequence[float] | NDArray[np.float64]
) -> None:
    """Writes the answer file.

    Args:
        path: File to write; its folder is created if missing.
        image_ids: Photo ids, in the order of the test split.
        loads: Predicted load of every photo.

    Raises:
        ValueError: If the two sequences have different lengths.
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        for image_id, value in zip(image_ids, loads, strict=True):
            writer.writerow([image_id, f"{float(value):.{DECIMALS}f}"])
