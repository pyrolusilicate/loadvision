"""Maps of what the model saw, kept on disk so the same photo is drawn once."""

import os
import re
import secrets
from pathlib import Path
from typing import Final

from loadvision.application.ports import ExplanationKind
from loadvision.domain.assessment import SHA256_HEX_PATTERN
from loadvision.domain.model_version import ModelVersionId

_SHA256_RE: Final = re.compile(SHA256_HEX_PATTERN)

# The same sharding as the photos: a folder never grows to hundreds of thousands of entries.
_SHARD_CHARS: Final = 2


class HeatmapStore:
    """Drawn maps on the local disk, one file per photo, model version and kind.

    A map depends on nothing else, so those three name the file. The store is asked before the
    explainer is built: a map drawn before a restart is read back without loading the encoder
    it was drawn with. Maps of a version that is gone can be deleted at any time; the next
    request draws them again.

    Args:
        root: Folder for the maps, such as `storage/heatmaps`; created on the first write.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def read(self, sha256: str, version: ModelVersionId, kind: ExplanationKind) -> bytes | None:
        """Returns the stored map, or None if it has not been drawn yet.

        Raises:
            ValueError: If `sha256` is not 64 lowercase hex characters.
        """
        try:
            return self.path(sha256, version, kind).read_bytes()
        except FileNotFoundError:
            return None

    def write(
        self, sha256: str, version: ModelVersionId, kind: ExplanationKind, png: bytes
    ) -> None:
        """Stores the map.

        The bytes go to a temporary file next to the target and are renamed into place, as
        photos are: a reader never sees half a map, and a failed write leaves nothing behind.

        Raises:
            ValueError: If `sha256` is not 64 lowercase hex characters.
            OSError: If the file cannot be written.
        """
        path = self.path(sha256, version, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{secrets.token_hex(4)}.partial")
        try:
            with partial.open("xb") as file:
                file.write(png)
                file.flush()
                os.fsync(file.fileno())
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)

    def path(self, sha256: str, version: ModelVersionId, kind: ExplanationKind) -> Path:
        """Returns where the map of this photo is kept.

        Args:
            sha256: Lowercase hex SHA-256 of the photo.
            version: Model version that draws.
            kind: Which map.

        Raises:
            ValueError: If `sha256` is not 64 lowercase hex characters; the check also keeps a
                crafted value such as ``../..`` from pointing outside the folder.
        """
        if _SHA256_RE.fullmatch(sha256) is None:
            raise ValueError("sha256 must be 64 lowercase hex characters")
        folder = self._root / version.value / sha256[:_SHARD_CHARS]
        return folder / f"{sha256}-{kind.value}.jpg"
