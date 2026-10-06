"""Picks the line "send this photo to a person" of a model package, and writes it into it.

The line is a share of the band a package measures on itself, and the share does not carry
over between packages: on one of ours the line 0.05 lets through a body of potato sacks that
another one sends to a person, and the line that catches it there would send 18.6% of real
photos to a person here. So every package gets its own line, picked by one rule from two
folders and kept in its manifest, rather than a number in the service settings that a person
has to remember to change with every model.

`--real` holds photos the service must answer on: the held-out photos of the contest are the
honest choice, because no model has seen them. `--foreign` holds photos it must refuse or ask
about: another kind of body, a tractor with no trailer, a road, a wall.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.imaging.validator import ImageLimits, prepare_image
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.manifest import Manifest
from loadvision.ml.packaging.model_package import (
    ModelPackage,
    ModelPackageError,
    write_package,
)

# Largest share of real photos the line may send to a person.
DEFAULT_BUDGET: Final = 0.01
# Generous on purpose: these are photos from a folder, not requests from the API.
LIMITS: Final = ImageLimits(
    max_bytes=50_000_000,
    max_pixels=50_000_000,
    min_side=64,
    media_types=frozenset(MediaType),
)
SUFFIXES: Final = (".jpg", ".jpeg", ".png", ".webp")
# Shares are read to the hundredth: the band itself is measured to about that.
_STEP: Final = 0.01
# How far above the most unusual real photo the line stands, in shares of the band. A photo we
# have never seen may land a little further than any we have.
_MARGIN: Final = 0.03

_logger = logging.getLogger(__name__)


class Scorer(Protocol):
    """What the rule needs from a loaded package: its band and the distance of one photo."""

    @property
    def band(self) -> tuple[float, float] | None:
        """The far edge of the training photos and the distance of empty frames, or None."""
        ...

    def ood_score(self, image: PreparedImage) -> float | None:
        """Distance from a photo to the training ones, or None when none is measured."""
        ...


@dataclass(frozen=True, slots=True)
class Measured:
    """What the two folders look like on the scale of one package.

    Attributes:
        real: Share of every photo the service must answer on.
        foreign: Share of every photo it must not answer on.
    """

    real: NDArray[np.float64]
    foreign: NDArray[np.float64]


def photos_in(folder: Path) -> list[Path]:
    """Returns the photos of a folder, in a stable order.

    Raises:
        FileNotFoundError: If the folder holds no photo this module can read.
    """
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    found = [path for path in files if path.suffix.lower() in SUFFIXES]
    # Said out loud: a folder where half the files are quietly ignored looks like a folder
    # where every photo passed.
    for path in files:
        if path not in found:
            _logger.warning("skipped, not a photo: %s", path.name)
    if not found:
        raise FileNotFoundError(f"{folder} holds no {', '.join(SUFFIXES)}")
    return found


def shares(paths: Sequence[Path], scorer: Scorer, band: tuple[float, float]) -> NDArray[np.float64]:
    """Returns where the photos land in the band, as shares.

    Only the distance is computed, never a whole prediction: the two are the same number to the
    last bit and the distance alone costs a sixth of the work, which over a folder of three
    hundred photos is two minutes instead of eleven.

    Args:
        paths: Photos to measure.
        scorer: Loaded package.
        band: The two ends of the package's scale.

    Returns:
        One share per photo that could be read and placed.
    """
    edge, empty = band
    values: list[float] = []
    for path in paths:
        try:
            image = prepare_image(path.read_bytes(), LIMITS)
        # A folder of files, not an API request: one unreadable file must not stop the run.
        except Exception as error:
            _logger.warning("skipped, %s: %s", path.name, error)
            continue
        score = scorer.ood_score(image)
        if score is not None:
            values.append((score - edge) / (empty - edge))
    return np.asarray(values, dtype=np.float64)


def measure(scorer: Scorer, real: Path, foreign: Path) -> Measured | None:
    """Places both folders on the scale of a package, or returns None when it has no scale."""
    band = scorer.band
    if band is None:
        return None
    return Measured(
        real=shares(photos_in(real), scorer, band),
        foreign=shares(photos_in(foreign), scorer, band),
    )


def line_for(measured: Measured, budget: float = DEFAULT_BUDGET) -> float:
    """Returns the line this package should use, rounded to a hundredth of the band.

    The service exists to notice a photo that is not ours, so catching the foreign frames comes
    first and the cost is the constraint, not the other way round. The line is put just under
    the nearest foreign frame; where that would send more than `budget` of the real photos to a
    person, it retreats to the highest share the budget allows and lets some foreign frames
    slip under it.

    An earlier rule ignored the foreign folder and only kept clear of the real photos, with a
    margin on top. It read well and let a stock photo of somebody else's loaded body through by
    a hundredth of the band: the margin had walked the line straight past it.

    Args:
        measured: Shares of both folders.
        budget: Largest share of real photos that may go to a person, from 0 to 1.

    Returns:
        The share at which the line stands.

    Raises:
        ValueError: If no real photo was measured.
    """
    if measured.real.size == 0:
        raise ValueError("no real photo was measured; the cost of a line cannot be read")
    # The photo that decides the cost is the one just inside the budget, not the very last one:
    # a single odd photo out of three hundred must not drag the line past everything else.
    affordable = float(np.quantile(measured.real, 1.0 - budget))
    if measured.foreign.size == 0:
        return _rounded(affordable + _MARGIN)
    # Just under the nearest foreign frame, by one step of the scale the shares are read to.
    wanted = _rounded(float(measured.foreign.min())) - _STEP
    return max(wanted, _rounded(affordable))


def record_line(
    root: Path,
    real: Path,
    foreign: Path,
    *,
    device: str = "cpu",
    budget: float = DEFAULT_BUDGET,
) -> float | None:
    """Picks the line of a package that is already written and keeps it in its manifest.

    The package is loaded the way the service loads it, so the band the line is placed in is
    the one the service will measure.

    Args:
        root: Package folder.
        real: Photos the service must answer on.
        foreign: Photos it must refuse or ask about.
        device: Torch device the encoder runs on.
        budget: Largest share of real photos that may go to a person.

    Returns:
        The line written, or None when the package measures no distances and has no line.

    Raises:
        FileNotFoundError: If a folder holds no photo.
        ModelPackageError: If the package does not load.
    """
    package = ModelPackage.load(root)
    if package.manifest.ood is None:
        return None
    measured = measure(PackagePredictor(package, device=device), real, foreign)
    if measured is None:
        return None
    line = line_for(measured, budget)
    # Outside the band the line means nothing: the package cannot tell the two folders apart
    # on its own scale. A retraining must not fail over it; the service keeps its own line.
    if not -1 < line < 1:
        _logger.warning(
            "no line is kept for %s: %+.2f falls outside its band", package.version.value, line
        )
        return None
    caught = int((measured.foreign >= line).sum())
    flagged = int((measured.real >= line).sum())
    _logger.info(
        "line of %s: %+.2f, foreign caught %d of %d, real flagged %d of %d",
        package.version.value,
        line,
        caught,
        measured.foreign.size,
        flagged,
        measured.real.size,
    )
    write_line(root, line)
    return line


def write_line(root: Path, line: float) -> None:
    """Keeps a picked line in the manifest of a package.

    The manifest is the only file rewritten: the line is not among the files it lists, so their
    hashes stay valid.

    Args:
        root: Package folder.
        line: Share of the band the line stands at.

    Raises:
        ModelPackageError: If the package does not load or measures no distances.
    """
    package = ModelPackage.load(root)
    if package.manifest.ood is None:
        raise ModelPackageError(f"{root}: package measures no distances, it has no line to keep")
    data = package.manifest.model_dump(mode="json")
    data["ood"]["review_share"] = line
    write_package(root, Manifest.model_validate(data))


def _rounded(share: float) -> float:
    """Rounds a share up to the hundredth of the band the report speaks in."""
    return float(np.ceil(share / _STEP) * _STEP)
