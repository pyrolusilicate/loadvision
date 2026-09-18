"""Encoder checkpoints a model package carries, and how a clean clone gets them back.

A package keeps the untouched timm checkpoint of every encoder and the manifest records its
hash. The checkpoints are public on Hugging Face and take a gigabyte, so a package can travel
without them: `fetch_checkpoints` downloads each one by name and accepts it only if the hash
matches, which means the package runs exactly the weights it was built with.
"""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Final

from loadvision.ml.packaging.manifest import Manifest
from loadvision.ml.packaging.model_package import ModelPackageError, file_sha256, read_manifest
from loadvision.ml.registry import BACKBONES

# The file every timm repository on Hugging Face keeps its weights in.
CHECKPOINT_FILE: Final = "model.safetensors"
WEIGHTS_DIR: Final = "weights"

type Download = Callable[[str], Path]


def download_checkpoint(timm_name: str) -> Path:
    """Downloads a timm checkpoint into the Hugging Face cache.

    Args:
        timm_name: Checkpoint name in the timm model hub.

    Returns:
        Path of the file in the cache.
    """
    # Imported here: huggingface_hub comes with the serve group, and commands that never
    # download must work without it.
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(f"timm/{timm_name}", CHECKPOINT_FILE))


def place_checkpoint(root: Path, timm_name: str, download: Download = download_checkpoint) -> Path:
    """Copies a pretrained checkpoint into a package that is being built.

    Args:
        root: Package folder.
        timm_name: Checkpoint name in the timm model hub.
        download: Gets the checkpoint file; tests pass their own.

    Returns:
        Path of the copy inside the package.
    """
    target = root / WEIGHTS_DIR / f"{timm_name}.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        shutil.copyfile(download(timm_name), target)
    return target


def fetch_checkpoints(root: Path, download: Download = download_checkpoint) -> list[str]:
    """Downloads the checkpoints a package lists but does not hold, and checks their hashes.

    Only the manifest is read: a package without its weights does not pass
    `ModelPackage.load`, and that is the package this function is for.

    Args:
        root: Package folder.
        download: Gets a checkpoint file by timm name; tests pass their own.

    Returns:
        Package paths of the files that were downloaded, empty when all were in place.

    Raises:
        ModelPackageError: If the manifest is missing or invalid, a member names an encoder
            the registry does not know, or a downloaded file has another hash than the
            manifest records: the checkpoint on Hugging Face changed, and the package would
            no longer run the weights it was built with.
        OSError: If a file cannot be written.
    """
    manifest = read_manifest(root)
    fetched: list[str] = []
    for name, timm_name in _checkpoints(manifest).items():
        target = root / name
        expected = manifest.files[name]
        if target.is_file() and file_sha256(target) == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(download(timm_name), target)
        actual = file_sha256(target)
        if actual != expected:
            target.unlink()
            raise ModelPackageError(
                f"{name}: timm/{timm_name} now has sha256 {actual}, the package was built with "
                f"{expected}"
            )
        fetched.append(name)
    return fetched


def _checkpoints(manifest: Manifest) -> dict[str, str]:
    """Maps the package path of every checkpoint to its timm name; members may share one."""
    checkpoints: dict[str, str] = {}
    for member in manifest.members:
        if member.weights is None:
            continue
        spec = BACKBONES.get(member.backbone)
        if spec is None:
            raise ModelPackageError(f"unknown backbone {member.backbone!r}")
        checkpoints[member.weights] = spec.timm_name
    return checkpoints
