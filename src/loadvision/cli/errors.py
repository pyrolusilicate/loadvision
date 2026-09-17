"""Errors a command reports in one line, without a traceback."""


class CommandError(Exception):
    """A command cannot do its job for a reason the user can fix, such as a missing git checkout."""
