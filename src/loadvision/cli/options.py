"""Options that several commands share, declared once."""

import argparse
from pathlib import Path
from typing import Final

DATA_DIR: Final = Path("data")


def add_data_option(parser: argparse.ArgumentParser) -> None:
    """Adds `--data`, the dataset root with `train/` and `test/`."""
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR,
        help="dataset root with train/ and test/ (default: %(default)s)",
    )


def add_device_option(parser: argparse.ArgumentParser, default: str) -> None:
    """Adds `--device`, the torch device the encoders run on.

    Args:
        parser: Parser of the command.
        default: Device when the option is omitted, such as ``cpu`` or ``mps``.
    """
    parser.add_argument(
        "--device",
        default=default,
        help="torch device, such as cpu, mps or cuda (default: %(default)s)",
    )
