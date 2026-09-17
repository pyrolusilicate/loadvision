"""Command line entry point: ``loadvision <group> <command>``."""

import argparse
import logging
from collections.abc import Sequence

from loadvision.cli.commands import data, eval, features, model, train
from loadvision.cli.errors import CommandError
from loadvision.errors import LoadvisionError

_logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Runs one command.

    Expected failures, such as a missing file, are logged in one line; anything else keeps its
    traceback.

    Args:
        argv: Arguments without the program name; by default the process arguments.

    Returns:
        Exit code: 0 on success, 1 for a reported failure. Invalid arguments exit with 2.
    """
    parser = argparse.ArgumentParser(
        prog="loadvision", description="Loadvision: cargo load estimation from photos."
    )
    groups = parser.add_subparsers(title="groups", required=True)
    data.add_commands(groups)
    features.add_commands(groups)
    eval.add_commands(groups)
    train.add_commands(groups)
    model.add_commands(groups)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return args.run(args)
    except (CommandError, LoadvisionError, OSError) as error:
        _logger.error("error: %s", error)
        return 1
