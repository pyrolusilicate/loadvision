from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

ALEMBIC_INI = Path(__file__).parents[2] / "alembic.ini"


@pytest.mark.unit
def test_migrations_form_one_chain() -> None:
    # Two migrations written in parallel branches give two heads, and `upgrade head` refuses
    # to run; this catches it before the merge.
    scripts = ScriptDirectory.from_config(Config(str(ALEMBIC_INI)))
    assert scripts.get_heads() == ["0001"]
    assert [revision.revision for revision in scripts.walk_revisions()] == ["0001"]
