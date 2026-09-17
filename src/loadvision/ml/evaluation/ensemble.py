"""Out-of-fold predictions of an ensemble: every member fitted and averaged the same way."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from loadvision.ml.data.manifest import Sample
from loadvision.ml.evaluation.cross_validation import out_of_fold
from loadvision.ml.evaluation.oof_heads import ridge_predictor
from loadvision.ml.models.backbones import load_backbone
from loadvision.ml.training.embedding_cache import embed_samples


@dataclass(frozen=True, slots=True, kw_only=True)
class Member:
    """One ensemble member: a frozen encoder and the ridge penalty of its head.

    Attributes:
        encoder: Registry key of the backbone.
        alpha: Ridge penalty of the head.
    """

    encoder: str
    alpha: float

    @classmethod
    def parse(cls, text: str) -> "Member":
        """Reads a member written as ``encoder:alpha``.

        Raises:
            ValueError: If the text has no penalty or the penalty is not a number.
        """
        encoder, separator, alpha = text.partition(":")
        if not separator or not encoder:
            raise ValueError(f"member must be written as encoder:alpha, got {text!r}")
        return cls(encoder=encoder, alpha=float(alpha))

    def __str__(self) -> str:
        """Returns the member as ``encoder:alpha``."""
        return f"{self.encoder}:{self.alpha:g}"


@dataclass(frozen=True, slots=True, kw_only=True)
class MemberOutOfFold:
    """What one member predicted out of fold.

    Attributes:
        member: The member.
        predictions: One prediction per photo.
        mae: Mean absolute error of these predictions.
    """

    member: Member
    predictions: NDArray[np.float64]
    mae: float


@dataclass(frozen=True, slots=True, kw_only=True)
class EnsembleOutOfFold:
    """What the ensemble predicted out of fold.

    Attributes:
        predictions: Mean of the member predictions, one per photo.
        members: Result of every member, in the order they were given.
    """

    predictions: NDArray[np.float64]
    members: tuple[MemberOutOfFold, ...]


def ensemble_out_of_fold(
    samples: Sequence[Sample],
    labels: NDArray[np.float64],
    folds: Sequence[int],
    members: Sequence[Member],
    *,
    device: str,
    cache_dir: Path,
) -> EnsembleOutOfFold:
    """Fits every member out of fold and averages their predictions.

    Args:
        samples: Labelled photos.
        labels: Reference loads, in the order of `samples`.
        folds: Fold number of every photo.
        members: Encoders and ridge penalties to average.
        device: Torch device the encoders run on.
        cache_dir: Folder with cached embeddings.

    Returns:
        The ensemble predictions and the result of every member.

    Raises:
        ValueError: If `members` is empty.
        KeyError: If an encoder is not in the registry.
    """
    if not members:
        raise ValueError("an ensemble needs at least one member")
    parts: list[MemberOutOfFold] = []
    for member in members:
        backbone = load_backbone(member.encoder, device)
        features = embed_samples(samples, backbone, cache_dir=cache_dir, device=device).astype(
            np.float64
        )
        result = out_of_fold(features, labels, folds, ridge_predictor(member.alpha))
        parts.append(
            MemberOutOfFold(
                member=member, predictions=result.predictions, mae=result.metrics["mae"]
            )
        )
    predictions = np.mean([part.predictions for part in parts], axis=0)
    return EnsembleOutOfFold(predictions=predictions, members=tuple(parts))
