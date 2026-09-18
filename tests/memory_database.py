"""The service database of a command, replaced by one in memory."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from loadvision.application.ports import UnitOfWorkFactory
from tests.fakes.in_memory_db import InMemoryStore


def use_memory_database(monkeypatch: pytest.MonkeyPatch, module: str) -> InMemoryStore:
    """Makes the commands of a module open an in-memory store instead of Postgres.

    Args:
        monkeypatch: Fixture that swaps the name.
        module: Command module that imports `unit_of_work_factory`, such as
            ``loadvision.cli.commands.jobs``.

    Returns:
        The store the commands will read and write.
    """
    store = InMemoryStore()

    @contextmanager
    def factory() -> Iterator[UnitOfWorkFactory]:
        yield store.unit_of_work

    monkeypatch.setattr(f"{module}.unit_of_work_factory", factory)
    return store
