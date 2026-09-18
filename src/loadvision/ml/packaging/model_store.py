"""The models folder the service reads packages from: one folder per version."""

import shutil
from pathlib import Path

from loadvision.ml.packaging.model_package import ModelPackage


def place_package(package_dir: Path, models_dir: Path) -> ModelPackage:
    """Verifies a package and puts it where the service looks for the version it serves.

    The package is checked before it is copied, so the models folder never holds a package
    with a missing or changed file. A package that is already in its place is left alone.

    Args:
        package_dir: Package to place, such as the `package/` folder of a run.
        models_dir: Folder with one package folder per version.

    Returns:
        The package in its place in the models folder.

    Raises:
        ModelPackageError: If the package is missing, invalid or a file does not match its
            hash.
        FileExistsError: If another package of this version is already in the models folder.
        OSError: If the package cannot be copied.
    """
    package = ModelPackage.load(package_dir)
    target = models_dir / package.version.value
    if target.resolve() == package_dir.resolve():
        return package
    if target.exists():
        raise FileExistsError(
            f"{target} already holds a package of version {package.version.value}"
        )
    shutil.copytree(package_dir, target)
    return ModelPackage.load(target)
