"""Writing and loading model packages."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

from pydantic import ValidationError

from loadvision.domain.model_version import ModelVersionId
from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.ml.packaging.manifest import MANIFEST_FILE, Manifest

# Out-of-fold predictions of the very models in the package, copied in at build time. The
# promotion gate compares a candidate with the active version on the rows they share, and
# refitting the active one to get them back would not be out-of-fold any more.
PACKAGE_OOF: Final = "metrics/oof.csv"


class ModelPackageError(LoadvisionError):
    """A model package cannot be used: the manifest is invalid or files do not match it.

    Args:
        detail: Which file or field failed and how, for logs.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.MODEL_PACKAGE_INVALID, detail)

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...]]:
        return (type(self), (self.detail,))


@dataclass(frozen=True, slots=True)
class ModelPackage:
    """A model package with a valid manifest and files that match their hashes.

    Attributes:
        root: Package folder.
        manifest: Parsed manifest.
    """

    root: Path
    manifest: Manifest

    @property
    def version(self) -> ModelVersionId:
        """Package version."""
        return self.manifest.model_version

    def path(self, name: str) -> Path:
        """Returns the location of a file listed in the manifest.

        Args:
            name: Package path from the manifest, such as a member's `head`.

        Raises:
            ModelPackageError: If the manifest does not list `name`.
        """
        if name not in self.manifest.files:
            raise ModelPackageError(f"{name} is not listed in {MANIFEST_FILE}")
        return self.root / name

    @classmethod
    def load(cls, root: Path) -> Self:
        """Reads the manifest and verifies every listed file.

        Files in the folder that the manifest does not list are ignored.

        Args:
            root: Package folder.

        Returns:
            The verified package.

        Raises:
            ModelPackageError: If the manifest is missing or invalid, or a listed file is
                missing, lies outside the package or has another hash. All file problems are
                reported together.
        """
        manifest_path = root / MANIFEST_FILE
        try:
            text = manifest_path.read_bytes()
        except OSError as error:
            raise ModelPackageError(f"cannot read {manifest_path}: {error}") from error
        try:
            manifest = Manifest.model_validate_json(text)
        except ValidationError as error:
            raise ModelPackageError(f"{manifest_path}: {_describe(error)}") from error
        problems = _check_files(root, manifest)
        if problems:
            raise ModelPackageError(f"{root}: {'; '.join(problems)}")
        return cls(root, manifest)


def write_package(root: Path, manifest: Manifest) -> ModelPackage:
    """Writes the manifest into a folder that already holds the package files, then loads it.

    Args:
        root: Package folder.
        manifest: Manifest whose `files` describe the folder contents.

    Returns:
        The package loaded back, so a package that cannot be loaded is never reported as built.

    Raises:
        ModelPackageError: If the written package does not load.
    """
    text = manifest.model_dump_json(indent=2)
    (root / MANIFEST_FILE).write_text(text + "\n", encoding="utf-8")
    return ModelPackage.load(root)


def hash_package_files(root: Path) -> dict[str, str]:
    """Hashes every file under the package folder except the manifest, for `Manifest.files`.

    Args:
        root: Package folder.

    Returns:
        SHA-256 by package path with ``/`` separators, sorted by path.
    """
    hashes = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file() and path != root / MANIFEST_FILE
    }
    return dict(sorted(hashes.items()))


def file_sha256(path: Path) -> str:
    """Returns the lowercase hex SHA-256 of a file, read in chunks."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _check_files(root: Path, manifest: Manifest) -> list[str]:
    problems: list[str] = []
    resolved_root = root.resolve()
    for name, expected in sorted(manifest.files.items()):
        path = root / name
        if not path.resolve().is_relative_to(resolved_root):
            problems.append(f"{name} points outside the package")
        elif not path.is_file():
            problems.append(f"{name} is missing")
        else:
            try:
                actual = file_sha256(path)
            except OSError as error:
                problems.append(f"cannot read {name}: {error}")
                continue
            if actual != expected:
                problems.append(f"{name} has sha256 {actual}, manifest says {expected}")
    return problems


def _describe(error: ValidationError) -> str:
    # Pydantic's own text adds documentation links and echoes input values; the location and
    # message are enough to find the field.
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors(include_url=False)
    )
