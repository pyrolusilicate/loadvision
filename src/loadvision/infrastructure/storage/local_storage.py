"""Content-addressed file storage for images and model packages."""

import os
import re
import secrets
from pathlib import Path
from typing import Final

from loadvision.domain.assessment import SHA256_HEX_PATTERN
from loadvision.imaging.prepared_image import PreparedImage

_SHA256_RE: Final = re.compile(SHA256_HEX_PATTERN)

# Photos are spread over 256 subfolders by the first two hash characters, so no single folder
# grows to hundreds of thousands of entries.
_SHARD_CHARS: Final = 2


class LocalImageStorage:
    """`ImageStorage` on the local disk: one file per photo, named by its SHA-256.

    Args:
        root: Folder for photos, such as `storage/images`; created on the first save.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, image: PreparedImage) -> None:
        """Stores the photo bytes; saving the same photo again changes nothing.

        The bytes go to a temporary file next to the target and are renamed into place, so a
        reader never sees a half-written photo, even if the process dies during the write.

        Args:
            image: Decoded photo whose original bytes are stored.

        Raises:
            OSError: If the file cannot be written; no temporary file is left behind.
        """
        path = self.path(image.sha256)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{secrets.token_hex(4)}.partial")
        try:
            with partial.open("xb") as file:
                file.write(image.raw)
                file.flush()
                os.fsync(file.fileno())
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)

    def load(self, sha256: str) -> bytes | None:
        """Returns the stored bytes, or None if the photo is unknown.

        Raises:
            ValueError: If `sha256` is not 64 lowercase hex characters.
        """
        try:
            return self.path(sha256).read_bytes()
        except FileNotFoundError:
            return None

    def path(self, sha256: str) -> Path:
        """Returns where the photo with this hash is stored.

        Args:
            sha256: Lowercase hex SHA-256 of the photo.

        Raises:
            ValueError: If `sha256` is not 64 lowercase hex characters; the check also keeps a
                crafted value such as ``../..`` from pointing outside the storage.
        """
        if _SHA256_RE.fullmatch(sha256) is None:
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return self._root / sha256[:_SHARD_CHARS] / sha256
