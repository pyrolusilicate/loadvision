"""Running one command without importing every other one."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from loadvision.cli.errors import CommandError

_logger = logging.getLogger(__name__)

type AddCommands = Callable[["argparse._SubParsersAction[argparse.ArgumentParser]"], None]


def run_alone(add_commands: AddCommands, command: str, argv: Sequence[str] | None = None) -> int:
    """Parses and runs one command the way the top-level CLI would, and nothing else.

    The GPU machine and the research runs need a couple of ML commands, and a broken import in
    an unrelated one must not keep them from starting: `python -m <command module> ...`.

    Args:
        add_commands: The command module's `add_commands`.
        command: Its name, the first word after `loadvision`.
        argv: Arguments after the name; the process arguments by default.

    Returns:
        Exit code: the command's own, or 1 for an expected failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(prog="loadvision")
    add_commands(parser.add_subparsers(required=True))
    args = parser.parse_args([command, *(sys.argv[1:] if argv is None else argv)])
    try:
        return args.run(args)
    except CommandError as error:
        _logger.error("error: %s", error)
        return 1
