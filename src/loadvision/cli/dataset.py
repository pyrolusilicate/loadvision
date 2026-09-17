"""Reading the dataset for commands: a broken dataset becomes a one-line error."""

from pathlib import Path

from loadvision.cli.errors import CommandError
from loadvision.ml.data.manifest import Sample, load_split


def train_samples(data_dir: Path) -> list[Sample]:
    """Reads the labelled split.

    Args:
        data_dir: Dataset root with `train/` and `test/`.

    Returns:
        The photos of `train`, sorted by image id.

    Raises:
        CommandError: If the csv files and the photos disagree or a label is out of range.
        FileNotFoundError: If a csv or the images folder is missing.
    """
    try:
        return load_split(data_dir, "train")
    except ValueError as error:
        raise CommandError(f"dataset {data_dir}: {error}") from error
