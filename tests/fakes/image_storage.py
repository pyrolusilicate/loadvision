"""In-memory ImageStorage for tests."""

from loadvision.imaging.prepared_image import PreparedImage


class InMemoryImageStorage:
    """Keeps photo bytes in a dict keyed by SHA-256."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def save(self, image: PreparedImage) -> None:
        self.files.setdefault(image.sha256, image.raw)

    def load(self, sha256: str) -> bytes | None:
        return self.files.get(sha256)
