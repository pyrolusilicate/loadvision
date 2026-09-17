"""Where a model package comes from: the code commit and the dataset it was built on."""

import subprocess
import tempfile
from pathlib import Path
from typing import Final

from loadvision.cli.errors import CommandError
from loadvision.ml.packaging.model_package import file_sha256
from loadvision.ml.pipeline.stage import RunContext
from loadvision.ml.pipeline.stages.build_manifest import BuildManifest

UNKNOWN_COMMIT: Final = "unknown"


def git_commit() -> str:
    """Returns the full hash of the checked-out commit.

    Raises:
        CommandError: If git is not installed or the current folder is not a git checkout.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError as error:
        raise CommandError("git is not installed") from error
    if result.returncode != 0:
        raise CommandError(f"cannot read the git commit: {result.stderr.strip()}")
    return result.stdout.strip()


def describe_commit() -> str:
    """Returns the short hash of the checked-out commit, or ``unknown`` outside a checkout.

    Research records only name the code that produced a row, so a run on a machine with a copy
    of the code and no git history is still worth recording. A model package, where the commit
    has to be exact, uses `git_commit` instead.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return UNKNOWN_COMMIT
    return result.stdout.strip() or UNKNOWN_COMMIT


def dataset_hash(data_dir: Path) -> str:
    """Returns the SHA-256 of the dataset manifest.

    The manifest is the `manifest.csv` the first pipeline stage writes, built here in a
    temporary folder, so a package built by a command and one built by a pipeline run describe
    the same dataset with the same hash.

    Args:
        data_dir: Dataset root with `train/` and `test/`.

    Raises:
        FileNotFoundError: If a split folder or csv is missing.
        CommandError: If the csv files and the photos disagree.
        StageCheckError: If the manifest comes out incomplete.
    """
    stage = BuildManifest()
    with tempfile.TemporaryDirectory() as temporary:
        ctx = RunContext(
            run_id="dataset-hash",
            runs_dir=Path(temporary),
            cache_dir=Path(temporary),
            data_dir=data_dir,
            config={},
            seed=0,
        )
        try:
            result = stage.run(ctx)
        except ValueError as error:
            raise CommandError(f"dataset {data_dir}: {error}") from error
        stage.check(ctx, result)
        return file_sha256(result.artifacts["manifest"])
