"""Maps of what the model saw, kept on disk so the same photo is drawn once."""

import os
import re
import secrets
from pathlib import Path
from typing import Final

from loadvision.application.ports import ExplanationKind, LoadExplainer
from loadvision.domain.assessment import SHA256_HEX_PATTERN
from loadvision.domain.model_version import ModelVersionId
from loadvision.imaging.prepared_image import PreparedImage

_SHA256_RE: Final = re.compile(SHA256_HEX_PATTERN)

# The same sharding as the photos: a folder never grows to hundreds of thousands of entries.
_SHARD_CHARS: Final = 2


class CachedExplainer:
    """`LoadExplainer` that keeps every map it draws.

    A map depends on nothing but the photo, the model version and the kind, so those three
    name the file and a second look costs a read instead of a pass of the encoder. Maps of a
    version that is gone can be deleted at any time; they are drawn again on demand.

    Args:
        inner: The explainer that draws.
        root: Folder for the maps, such as `storage/heatmaps`; created on the first write.
    """

    def __init__(self, inner: LoadExplainer, root: Path) -> None:
        self._inner = inner
        self._root = root

    @property
    def version(self) -> ModelVersionId:
        """Version of the model package the maps are drawn with."""
        return self._inner.version

    def explain(self, image: PreparedImage, kind: ExplanationKind) -> bytes:
        """Returns the stored map of this photo, drawing it first if there is none.

        Args:
            image: Decoded photo.
            kind: Which map to draw.

        Returns:
            PNG bytes.

        Raises:
            Exception: Whatever the inner explainer raises; a failed drawing stores nothing.
        """
        path = self.path(image.sha256, kind)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            pass
        png = self._inner.explain(image, kind)
        self._write(path, png)
        return png

    def path(self, sha256: str, kind: ExplanationKind) -> Path:
        """Returns where the map of this photo is kept.

        Args:
            sha256: Lowercase hex SHA-256 of the photo.
            kind: Which map.

        Raises:
            ValueError: If `sha256` is not 64 lowercase hex characters; the check also keeps a
                crafted value such as ``../..`` from pointing outside the folder.
        """
        if _SHA256_RE.fullmatch(sha256) is None:
            raise ValueError("sha256 must be 64 lowercase hex characters")
        folder = self._root / self.version.value / sha256[:_SHARD_CHARS]
        return folder / f"{sha256}-{kind.value}.png"

    @staticmethod
    def _write(path: Path, png: bytes) -> None:
        # Written beside the target and renamed into place, as photos are: a reader never sees
        # half a map, and a failed write leaves nothing behind.
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
