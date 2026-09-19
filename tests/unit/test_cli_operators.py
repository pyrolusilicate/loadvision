import logging

import pytest

from loadvision.application.passwords import verify_password
from loadvision.cli.main import main
from tests.fakes.in_memory_db import InMemoryStore
from tests.memory_database import use_memory_database

PASSWORD = "correct horse battery"


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> InMemoryStore:
    return use_memory_database(monkeypatch, "loadvision.cli.commands.operators")


def type_passwords(monkeypatch: pytest.MonkeyPatch, *answers: str) -> None:
    replies = iter(answers)
    monkeypatch.setattr("loadvision.cli.commands.operators.getpass", lambda _prompt: next(replies))


@pytest.mark.unit
def test_an_account_is_created_with_the_typed_password(
    store: InMemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    type_passwords(monkeypatch, PASSWORD, PASSWORD)

    assert main(["operators", "add", "anna"]) == 0

    assert verify_password(PASSWORD, store.operators["anna"].password_hash)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("login", "answers", "message"),
    [
        ("anna", (PASSWORD, PASSWORD + "!"), "the passwords differ"),
        ("anna", ("short", "short"), "at least"),
        ("Anna", (PASSWORD, PASSWORD), "login must match"),
    ],
    ids=["differ", "short", "malformed-login"],
)
def test_an_account_is_refused(
    store: InMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    login: str,
    answers: tuple[str, str],
    message: str,
) -> None:
    type_passwords(monkeypatch, *answers)

    assert main(["operators", "add", login]) == 1

    assert message in caplog.text
    assert store.operators == {}


@pytest.mark.unit
def test_an_account_is_disabled_and_given_a_new_password(
    store: InMemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    type_passwords(monkeypatch, PASSWORD, PASSWORD, "a brand new one", "a brand new one")
    assert main(["operators", "add", "anna"]) == 0

    assert main(["operators", "disable", "anna"]) == 0
    assert store.operators["anna"].active is False

    assert main(["operators", "password", "anna"]) == 0
    assert store.operators["anna"].active is True
    assert verify_password("a brand new one", store.operators["anna"].password_hash)


@pytest.mark.unit
@pytest.mark.parametrize("command", ["disable", "password"])
def test_an_unknown_account_is_reported(
    store: InMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    command: str,
) -> None:
    type_passwords(monkeypatch, PASSWORD, PASSWORD)

    assert main(["operators", command, "nobody"]) == 1

    assert "no operator nobody" in caplog.text


@pytest.mark.unit
def test_accounts_are_listed(
    store: InMemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    assert main(["operators", "list"]) == 0
    assert "no accounts yet" in caplog.text

    type_passwords(monkeypatch, PASSWORD, PASSWORD)
    main(["operators", "add", "anna"])
    caplog.clear()

    assert main(["operators", "list"]) == 0

    header, row = (record.getMessage() for record in caplog.records)
    assert header.split() == ["login", "active", "created"]
    assert row.split()[:2] == ["anna", "yes"]
