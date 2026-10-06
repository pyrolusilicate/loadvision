"""Prints the line "send this to a person" a model package should use, and what it costs.

The rule itself lives in `loadvision.ml.inference.review_line`: `build-finetuned` and the
retraining worker pick the line with it and keep it in the package. This tool shows the
measurements behind the number and, with `--write`, records it into a package built before the
line travelled with it.

It needs two folders. `--real` holds photos the service must answer on - the held-out photos
of the contest are the honest choice, because no model has seen them. `--foreign` holds photos
it must refuse or ask about: another kind of body, a tractor with no trailer, a road, a wall.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.inference.review_line import (
    DEFAULT_BUDGET,
    Measured,
    line_for,
    measure,
    write_line,
)
from loadvision.ml.packaging.model_package import ModelPackage

_logger = logging.getLogger("ood_line")


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
        default=DEFAULT_BUDGET,
        help="share of real photos that may go to a person (default: %(default)s)",
    )
    parser.add_argument("--device", default="cpu", help="torch device (default: %(default)s)")
    parser.add_argument(
        "--write", action="store_true", help="record the line into the package manifest"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    predictor = PackagePredictor(ModelPackage.load(args.package), device=args.device)
    measured = measure(predictor, args.real, args.foreign)
    if measured is None:
        _logger.error("пакет %s не меряет расстояний, черту ставить не от чего", args.package.name)
        return 1
    _logger.info("\nпакет %s", args.package.name)
    line = line_for(measured, args.budget)
    report(measured, line, args.budget)
    if args.write:
        write_line(args.package, line)
        _logger.info("\nчерта %+.2f записана в манифест %s", line, args.package.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
