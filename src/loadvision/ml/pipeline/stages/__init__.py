"""Pipeline stages and the factory that builds them from an experiment config."""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from loadvision.ml.pipeline.stage import Stage
from loadvision.ml.pipeline.stages.build_manifest import BuildManifest
from loadvision.ml.pipeline.stages.build_package import BuildPackage
from loadvision.ml.pipeline.stages.evaluate import Evaluate
from loadvision.ml.pipeline.stages.extract_embeddings import ExtractEmbeddings
from loadvision.ml.pipeline.stages.predict_test import PredictTest
from loadvision.ml.pipeline.stages.train_folds import Chain, Member, TrainFolds
from loadvision.ml.training.head_trainer import DEFAULT_HEAD

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CHAIN",
    "DEFAULT_MEMBERS",
    "BuildManifest",
    "BuildPackage",
    "Evaluate",
    "ExtractEmbeddings",
    "PredictTest",
    "TrainFolds",
    "build_pipeline",
]

DEFAULT_FOLDS: Final = Path("configs/splits/folds.json")
DEFAULT_VERSION: Final = "v1"

# The configuration we ship, written down in code and not only in a JSON file, so that
# "what is our best known setup" has one answer a reader can find. Every number here is a
# measurement; the table behind them is in docs/ml/methods.md.
#
# Four members, all weights Apache-2.0: two encoders looking at the scene as a whole and two
# looking at a 4x3 grid of it, because the label is the share of the floor taken by cargo and
# a single scene vector cannot see that.
DEFAULT_MEMBERS: Final = (
    ("pe_spatial_vitb16", 50.0),
    ("pe_spatial_vitb16_grid4x3_padmean", 1000.0),
    ("siglip2_vitb16_grid4x3", 1000.0),
    ("dinov2_vitb14", 300.0),
)

# A median calibration per population of photos, then rounding to 5. The calibration is worth
# 1.90 MAE on its own; the rounding matches the grid the expert labelled on, where 97% of the
# labels are multiples of 5.
DEFAULT_CHAIN: Final = (
    ("median_bins_by_aspect", {"bins": 16.0, "low": 0.70, "high": 0.80}),
    ("round_to", {"step": 5.0}),
)


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
        ValueError: If a member or a postprocessing step is malformed. A config that names
            neither gets `DEFAULT_MEMBERS` and `DEFAULT_CHAIN`, which is the shipped setup.
    """
    members = _members(config)
    chain = _chain(config)
    folds_file = Path(str(config.get("folds", DEFAULT_FOLDS)))
    encoders = sorted({member.encoder for member in members})
    return [
        BuildManifest(),
        ExtractEmbeddings(encoders=encoders, device=device),
        TrainFolds(members=members, chain=chain, folds_file=folds_file, device=device),
        Evaluate(),
        PredictTest(members=members, chain=chain, device=device),
        BuildPackage(
            members=members,
            chain=chain,
            package_version=str(config.get("version", DEFAULT_VERSION)),
            folds_file=folds_file,
            device=device,
        ),
    ]


def _members(config: Mapping[str, Any]) -> Sequence[Member]:
    if "members" not in config:
        logger.info("config names no members, using the shipped ensemble")
        return [
            Member(encoder=name, head=DEFAULT_HEAD, params={"alpha": alpha})
            for name, alpha in DEFAULT_MEMBERS
        ]
    # An empty list is a mistake, not a request for the defaults: leaving the key out asks for
    # them, writing it out and emptying it asks for an ensemble of nobody.
    raw = config["members"]
    if not raw:
        raise ValueError("experiment config needs a non-empty 'members' list")
    members: list[Member] = []
    for item in raw:
        try:
            params = {str(key): float(value) for key, value in item.get("params", {}).items()}
            members.append(
                Member(
                    encoder=str(item["encoder"]),
                    head=str(item.get("head", DEFAULT_HEAD)),
                    params=params,
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"member {item!r} needs 'encoder' and, optionally, 'head' and numeric 'params'"
            ) from error
    return members


def _chain(config: Mapping[str, Any]) -> Chain:
    if "postprocess" not in config:
        logger.info("config names no postprocessing, using the shipped chain")
        return [(name, dict(params)) for name, params in DEFAULT_CHAIN]
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
