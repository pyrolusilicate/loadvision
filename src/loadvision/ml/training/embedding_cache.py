"""Embeddings of dataset photos, computed once and cached between runs."""

import hashlib
import logging
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.preprocessing import Params, preprocess
from loadvision.ml.data.manifest import Sample
from loadvision.ml.models.backbones import PREPROCESSING_NAME, Backbone

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE: Final = 8

_MEDIA_TYPES: Final = {"JPEG": MediaType.JPEG, "PNG": MediaType.PNG, "WEBP": MediaType.WEBP}

# Any fixed non-uniform photo works; these sides are coprime with every patch size we use,
# so padding and scaling both take part in the fingerprint.
_FINGERPRINT_SHAPE: Final = (37, 53, 3)
_FINGERPRINT_PIXELS: Final = 37 * 53 * 3


def embed_samples(
    samples: Sequence[Sample],
    backbone: Backbone,
    *,
    cache_dir: Path,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> NDArray[np.float32]:
    """Returns one embedding per sample, computing only the photos that are not cached yet.

    The cache is keyed by photo hash, not by image id, so a photo keeps its embedding when the
    dataset is extended with confirmed labels from the service.

    Args:
        samples: Photos to embed.
        backbone: Loaded encoder.
        cache_dir: Folder for cache files, one per backbone.
        device: Torch device the encoder is on.
        batch_size: How many photos go through the encoder at once.

    Returns:
        Float32 array of shape (len(samples), backbone.dim), in the order of `samples`.
    """
    cache_path = cache_dir / f"{backbone.name}.npz"
    fingerprint = preprocessing_fingerprint(backbone.params)
    cached = _load_cache(cache_path, backbone.dim, fingerprint)

    missing = [sample for sample in samples if sample.sha256 not in cached]
    if missing:
        logger.info(
            "embedding %d of %d photos with %s on %s",
            len(missing),
            len(samples),
            backbone.name,
            device,
        )
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            vectors = _embed_batch(batch, backbone, device)
            for sample, vector in zip(batch, vectors, strict=True):
                cached[sample.sha256] = vector
        _save_cache(cache_path, cached, fingerprint)

    return np.stack([cached[sample.sha256] for sample in samples])


def _embed_batch(batch: Sequence[Sample], backbone: Backbone, device: str) -> NDArray[np.float32]:
    inputs = np.stack(
        [preprocess(PREPROCESSING_NAME, read_image(s.path), backbone.params) for s in batch]
    )
    with torch.inference_mode():
        output = backbone.encode(torch.from_numpy(inputs).to(device))
    return output.float().cpu().numpy().astype(np.float32)


def read_image(path: Path) -> PreparedImage:
    """Decodes a dataset photo the same way the service decodes an uploaded one.

    Args:
        path: Photo on disk.

    Returns:
        Decoded photo.

    Raises:
        ValueError: If the file is not one of the supported formats.
    """
    raw = path.read_bytes()
    with Image.open(BytesIO(raw)) as image:
        media_type = _MEDIA_TYPES.get(image.format or "")
        if media_type is None:
            raise ValueError(f"{path}: unsupported image format {image.format!r}")
        pixels = np.asarray(image.convert("RGB"), np.uint8)
    return PreparedImage(raw=raw, media_type=media_type, pixels=pixels)


def preprocessing_fingerprint(params: Params) -> str:
    """Identifies the preprocessing a cache was built with.

    The file name only carries the backbone, so a changed normalization, a changed input box or
    an edit inside `resize_pad` would silently mix old and new vectors in one cache. Running the
    preprocessing on a fixed synthetic photo catches all three.

    Args:
        params: Preprocessing parameters of the backbone.

    Returns:
        Lowercase hex SHA-256 of the preprocessed synthetic photo.
    """
    pixels = np.arange(_FINGERPRINT_PIXELS, dtype=np.uint8).reshape(_FINGERPRINT_SHAPE)
    # raw bytes are only hashed into PreparedImage.sha256, which this fingerprint never uses.
    probe = PreparedImage(raw=b"fingerprint", media_type=MediaType.JPEG, pixels=pixels)
    return hashlib.sha256(preprocess(PREPROCESSING_NAME, probe, params).tobytes()).hexdigest()


def _load_cache(path: Path, dim: int, fingerprint: str) -> dict[str, NDArray[np.float32]]:
    if not path.is_file():
        return {}
    with np.load(path) as stored:
        hashes: NDArray[Any] = stored["hashes"]
        vectors: NDArray[np.float32] = stored["vectors"]
        stored_fingerprint = str(stored["fingerprint"]) if "fingerprint" in stored else ""
    if stored_fingerprint != fingerprint:
        raise ValueError(f"{path}: cache was built with other preprocessing, delete it and rerun")
    if vectors.ndim != 2 or vectors.shape[1] != dim or len(hashes) != len(vectors):
        raise ValueError(f"{path}: cache does not match the backbone, delete it and rerun")
    return dict(zip((str(h) for h in hashes), vectors, strict=True))


def _save_cache(path: Path, cached: dict[str, NDArray[np.float32]], fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = sorted(cached)
    np.savez(
        path,
        hashes=np.array(hashes),
        vectors=np.stack([cached[h] for h in hashes]),
        fingerprint=np.array(fingerprint),
    )
