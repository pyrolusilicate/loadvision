"""Picks the line "send this photo to a person" for a model package, from measurements.

The line is a share of the band a package measures on itself, and the share does not carry
over between packages: on one of ours the line 0.05 lets through a body of potato sacks that
another one sends to a person, and the line that catches it there would send 18.6% of real
photos to a person here. So every package gets its own line, and this is the tool that finds
it instead of a person reading a table.

It needs two folders. `--real` holds photos the service must answer on - the held-out photos
of the contest are the honest choice, because no model has seen them. `--foreign` holds photos
it must refuse or ask about: another kind of body, a tractor with no trailer, a road, a wall.

The rule is written down and has one knob. Catching the foreign frames comes first: the line
goes just under the nearest of them. The knob is `--budget`, the share of real photos that may
go to a person for it; where the line would cost more than that, it retreats and the report
says how many foreign frames then slip under it. Where none slip under, the package separates
them, even if the two groups overlap by a hundredth and no perfect gap exists.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from ood_scale import LIMITS, photos_in

from loadvision.imaging.validator import prepare_image
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.model_package import ModelPackage

# Shares are read to the hundredth: the band itself is measured to about that.
_STEP: Final = 0.01
# How far above the most unusual real photo the line stands, in shares of the band. A photo we
# have never seen may land a little further than any we have.
_MARGIN: Final = 0.03

_logger = logging.getLogger("ood_line")


@dataclass(frozen=True, slots=True)
class Measured:
    """What the two folders look like on the scale of one package.

    Attributes:
        real: Share of every photo the service must answer on.
        foreign: Share of every photo it must not answer on.
    """

    real: np.ndarray
    foreign: np.ndarray


def shares(folder: Path, predictor: PackagePredictor, band: tuple[float, float]) -> np.ndarray:
    """Returns where the photos of a folder land in the band, as shares.

    Only the distance is computed, never a whole prediction: the two are the same number to the
    last bit and the distance alone costs a sixth of the work, which over a folder of three
    hundred photos is two minutes instead of eleven.

    Args:
        folder: Photos to measure.
        predictor: Loaded package.
        band: The two ends of the package's scale.

    Returns:
        One share per photo that could be read and placed.
    """
    edge, empty = band
    values: list[float] = []
    for path in photos_in(folder):
        try:
            image = prepare_image(path.read_bytes(), LIMITS)
        # A folder of files, not an API request: one unreadable file must not stop the run.
        except Exception as error:
            _logger.warning("пропущено, %s: %s", path.name, error)
            continue
        score = predictor.ood_score(image)
        if score is not None:
            values.append((score - edge) / (empty - edge))
    return np.asarray(values, dtype=np.float64)


def line_for(measured: Measured, budget: float) -> float:
    """Returns the line this package should use, rounded to a hundredth of the band.

    The service exists to notice a photo that is not ours, so catching the foreign frames comes
    first and the cost is the constraint, not the other way round. The line is put just under
    the nearest foreign frame; where that would send more than `budget` of the real photos to a
    person, it retreats to the highest share the budget allows and the report says how many
    foreign frames slipped under it.

    An earlier rule ignored the foreign folder and only kept clear of the real photos, with a
    margin on top. It read well and let a stock photo of somebody else's loaded body through by
    a hundredth of the band: the margin had walked the line straight past it.

    Args:
        measured: Shares of both folders.
        budget: Largest share of real photos that may go to a person, from 0 to 1.

    Returns:
        The share at which the line stands.
    """
    # The photo that decides the cost is the one just inside the budget, not the very last one:
    # a single odd photo out of three hundred must not drag the line past everything else.
    affordable = float(np.quantile(measured.real, 1.0 - budget))
    if measured.foreign.size == 0:
        return _rounded(affordable + _MARGIN)
    # Just under the nearest foreign frame, by one step of the scale the shares are read to.
    wanted = _rounded(float(measured.foreign.min())) - _STEP
    return max(wanted, _rounded(affordable))


def _rounded(share: float) -> float:
    """Rounds a share up to the hundredth of the band the report speaks in."""
    return float(np.ceil(share / _STEP) * _STEP)


def report(measured: Measured, line: float, budget: float) -> None:
    """Writes what the line does to both folders and whether the package can tell them apart."""
    real, foreign = measured.real, measured.foreign
    _logger.info("\nнастоящие фото: %d", real.size)
    for name, value in (
        ("медиана", float(np.median(real))),
        ("95-й процентиль", float(np.quantile(real, 0.95))),
        ("максимум", float(real.max())),
    ):
        _logger.info("  %-16s %+.3f", name, value)

    _logger.info("\nцена черты на настоящих фото:")
    for candidate in np.arange(round(real.min(), 2), real.max() + 0.1, 0.05):
        flagged = int((real >= candidate).sum())
        _logger.info(
            "  %+.2f: %3d из %d (%.1f%%)", candidate, flagged, real.size, 100 * flagged / real.size
        )

    caught = int((foreign >= line).sum())
    _logger.info("\nчерта при допуске ложных срабатываний %.0f%%: %+.2f", 100 * budget, line)
    _logger.info("  настоящих фото на проверку: %d из %d", int((real >= line).sum()), real.size)
    _logger.info("  чужих кадров поймано:       %d из %d", caught, foreign.size)

    # The gap is context, not the verdict: a line that catches every foreign frame at a price
    # the budget allows is a working line even where the two groups overlap by a hundredth.
    apart = float(foreign.min() - real.max())
    _logger.info("\nзазор между самым необычным настоящим и самым «своим» чужим: %+.3f", apart)
    if caught < foreign.size:
        _logger.warning(
            "  %d чужих кадров прошли под чертой: допуск %.0f%% не позволяет опустить её ниже.",
            foreign.size - caught,
            100 * budget,
        )
        _logger.warning("  Либо поднимайте допуск, либо этот пакет не различает их - смотри зазор.")
    elif apart <= 0:
        _logger.info(
            "  отрицательный: идеально их не разделить, но черта ловит всех чужих, и цена выше"
        )
    else:
        _logger.info("  положительный: черту можно ставить где угодно внутри зазора")


def main() -> int:
    """Measures both folders on a package and prints the line it should use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="model package folder")
    parser.add_argument(
        "--real", type=Path, required=True, help="photos the service must answer on"
    )
    parser.add_argument(
        "--foreign", type=Path, required=True, help="photos it must refuse or ask about"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.01,
        help="share of real photos that may go to a person (default: %(default)s)",
    )
    parser.add_argument("--device", default="cpu", help="torch device (default: %(default)s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    predictor = PackagePredictor(ModelPackage.load(args.package), device=args.device)
    band = predictor.band
    if band is None:
        _logger.error("пакет %s не меряет расстояний, черту ставить не от чего", args.package.name)
        return 1
    measured = Measured(
        real=shares(args.real, predictor, band), foreign=shares(args.foreign, predictor, band)
    )
    _logger.info("\nпакет %s", args.package.name)
    report(measured, line_for(measured, args.budget), args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
