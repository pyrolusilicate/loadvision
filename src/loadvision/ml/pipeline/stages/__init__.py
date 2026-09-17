"""Pipeline stages and the factory that builds them from an experiment config."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from loadvision.ml.pipeline.stage import Stage
from loadvision.ml.pipeline.stages.build_manifest import BuildManifest
from loadvision.ml.pipeline.stages.evaluate import Evaluate
from loadvision.ml.pipeline.stages.extract_embeddings import ExtractEmbeddings
from loadvision.ml.pipeline.stages.predict_test import PredictTest
from loadvision.ml.pipeline.stages.train_folds import Chain, Member, TrainFolds

__all__ = [
    "BuildManifest",
    "Evaluate",
    "ExtractEmbeddings",
    "PredictTest",
    "TrainFolds",
    "build_pipeline",
]

DEFAULT_FOLDS: Final = Path("configs/splits/folds.json")


def build_pipeline(config: Mapping[str, Any], *, device: str = "cpu") -> list[Stage]:
    """Builds the full pipeline from an experiment config.

    This is the composition root of a run: stages get their parameters here and never read the
    config themselves, so every stage can be built directly in a test.

    Args:
        config: Experiment config with `members` and, optionally, `postprocess` and `folds`.
        device: Torch device the encoders run on.

    Returns:
        Stages in execution order.

    Raises:
        ValueError: If the config names no members or a member is malformed.
    """
    members = _members(config)
    chain = _chain(config)
    folds_file = Path(str(config.get("folds", DEFAULT_FOLDS)))
    encoders = sorted({encoder for encoder, _ in members})
    return [
        BuildManifest(),
        ExtractEmbeddings(encoders=encoders, device=device),
        TrainFolds(members=members, chain=chain, folds_file=folds_file, device=device),
        Evaluate(),
        PredictTest(members=members, chain=chain, device=device),
    ]


def _members(config: Mapping[str, Any]) -> Sequence[Member]:
    raw = config.get("members")
    if not raw:
        raise ValueError("experiment config needs a non-empty 'members' list")
    members: list[Member] = []
    for item in raw:
        try:
            members.append((str(item["encoder"]), float(item["alpha"])))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"member {item!r} needs 'encoder' and numeric 'alpha'") from error
    return members


def _chain(config: Mapping[str, Any]) -> Chain:
    steps: list[tuple[str, Mapping[str, float]]] = []
    for item in config.get("postprocess", ()):
        if isinstance(item, str):
            steps.append((item, {}))
            continue
        try:
            params = {str(key): float(value) for key, value in item.get("params", {}).items()}
            steps.append((str(item["name"]), params))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"postprocess entry {item!r} needs 'name'") from error
    return steps
