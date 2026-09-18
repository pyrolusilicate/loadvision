"""The commit a model package records, with and without a git checkout."""

import subprocess

import pytest

from loadvision.ml.pipeline.stages import build_package
from loadvision.settings import GIT_COMMIT_ENV

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.ml
def test_an_image_records_the_commit_it_was_built_from(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GIT_COMMIT_ENV, COMMIT)

    assert build_package._git_commit() == COMMIT


@pytest.mark.ml
def test_without_git_the_commit_is_unknown_instead_of_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_git(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    monkeypatch.delenv(GIT_COMMIT_ENV, raising=False)
    monkeypatch.setattr(build_package.subprocess, "run", no_git)

    assert build_package._git_commit() == "0" * 40
