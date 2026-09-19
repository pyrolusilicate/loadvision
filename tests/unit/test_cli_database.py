import logging

import pytest

from loadvision.cli.main import main
from loadvision.settings import DATABASE_URL_ENV


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv", [["operators", "list"], ["model", "list"], ["model", "rollback"], ["jobs", "list"]]
)
def test_a_command_without_the_database_says_so_in_one_line(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)

    with caplog.at_level(logging.INFO):
        code = main(argv)

    assert code == 1
    [record] = caplog.records
    assert record.getMessage() == (
        "error: DATABASE_URL is not set; create .env from .env.example and run "
        "uv run --env-file .env loadvision <command>"
    )
    assert record.exc_info is None
