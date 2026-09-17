"""Stage that fills the embedding cache for every encoder of an experiment."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from loadvision.ml.data.manifest import load_split
from loadvision.ml.models.backbones import Backbone, load_backbone
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.ml.training.embedding_cache import embed_samples

EMBEDDINGS_DIR: Final = "embeddings"


@dataclass(frozen=True, slots=True)
class ExtractEmbeddings:
    """Runs every encoder over both splits once and keeps the vectors in the cache.

    The encoders are frozen, so a photo always gets the same vector and the cache survives
    between runs: only photos added since the last run are computed.

    Attributes:
        encoders: Registry keys of the encoders to fill.
        device: Torch device the encoders run on.
        load: How an encoder is built from its registry key; replaced in tests by a
            stand-in that needs no download.
    """

    encoders: Sequence[str]
    device: str = "cpu"
    load: Callable[[str, str], Backbone] = load_backbone

    @property
    def name(self) -> str:
        """Stage name."""
        return "extract_embeddings"

    @property
    def version(self) -> str:
        """Stage version."""
        return "1"

    def inputs(self, ctx: RunContext) -> Mapping[str, str]:
        """What the result depends on."""
        return {"encoders": ",".join(self.encoders), "data_dir": str(ctx.data_dir)}

    def run(self, ctx: RunContext) -> StageResult:
        """Embeds both splits with every encoder."""
        train = load_split(ctx.data_dir, "train")
        test = load_split(ctx.data_dir, "test")
        cache_dir = ctx.cache_dir / EMBEDDINGS_DIR

        counts: dict[str, int] = {"photos": len(train) + len(test), "encoders": len(self.encoders)}
        metrics: dict[str, float] = {}
        artifacts = {}
        for encoder in self.encoders:
            backbone = self.load(encoder, self.device)
            vectors = embed_samples(
                [*train, *test], backbone, cache_dir=cache_dir, device=self.device
            )
            counts[f"dim_{encoder}"] = backbone.dim
            metrics[f"norm_{encoder}"] = float(np.linalg.norm(vectors, axis=1).mean())
            artifacts[encoder] = cache_dir / f"{encoder}.npz"
        return StageResult(artifacts=artifacts, counts=counts, metrics=metrics)

    def check(self, ctx: RunContext, result: StageResult) -> None:
        """Verifies every photo has a finite vector of the expected width.

        Raises:
            StageCheckError: If a cache file is missing, a photo is uncovered, a width differs
                from the encoder's or a value is not finite.
        """
        train = load_split(ctx.data_dir, "train")
        test = load_split(ctx.data_dir, "test")
        hashes = {sample.sha256 for sample in (*train, *test)}
        cache_dir = ctx.cache_dir / EMBEDDINGS_DIR
        for encoder in self.encoders:
            path = cache_dir / f"{encoder}.npz"
            if not path.is_file():
                raise StageCheckError(f"{self.name}: {encoder} has no cache file")
            with np.load(path) as stored:
                stored_hashes = {str(value) for value in stored["hashes"]}
                vectors = stored["vectors"]
            missing = hashes - stored_hashes
            if missing:
                raise StageCheckError(
                    f"{self.name}: {encoder} is missing {len(missing)} photos of {len(hashes)}"
                )
            expected = result.counts[f"dim_{encoder}"]
            if vectors.shape[1] != expected:
                raise StageCheckError(
                    f"{self.name}: {encoder} vectors are {vectors.shape[1]} wide, "
                    f"expected {expected}"
                )
            if not np.isfinite(vectors).all():
                raise StageCheckError(f"{self.name}: {encoder} produced values that are not finite")
