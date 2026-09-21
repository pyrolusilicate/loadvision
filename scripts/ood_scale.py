"""Where photos land on the scale of a model package, and what the service would say about them.

The service places its two lines - "show this to a person" and "show no prediction at all" -
inside a band the package measures on itself: from the far edge of its training photos to the
distance at which frames holding nothing land. This prints that band, every photo's place in
it and the verdict, so a new model or a new kind of photo can be checked without guessing.
"""

import argparse
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from loadvision.imaging.prepared_image import MediaType
from loadvision.imaging.validator import ImageLimits, prepare_image
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.model_package import ModelPackage
from loadvision.settings import load_settings

# Generous on purpose: this reads photos from anywhere, not requests from the API.
LIMITS: Final = ImageLimits(
    max_bytes=50_000_000,
    max_pixels=50_000_000,
    min_side=64,
    media_types=frozenset(MediaType),
)
SUFFIXES: Final = (".jpg", ".jpeg", ".png", ".webp")

_logger = logging.getLogger("ood_scale")


@dataclass(frozen=True, slots=True)
class Placed:
    """One photo on the scale of the package.

    Attributes:
        name: File name, for the report.
        load_pct: What the model answered.
        score: Distance from the photo to the training ones.
        share: Where that distance stands in the band, 0 at the training edge and 1 at the
            frames with nothing in them.
        verdict: What the service would do with this photo.
    """

    name: str
    load_pct: float
    score: float
    share: float
    verdict: str


def photos_in(folder: Path) -> list[Path]:
    """Returns the photos of a folder, in a stable order.

    Raises:
        FileNotFoundError: If the folder holds no photo this tool can read.
    """
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    found = [path for path in files if path.suffix.lower() in SUFFIXES]
    # Said out loud: a folder where half the files are quietly ignored looks like a folder
    # where every photo passed.
    for path in files:
        if path not in found:
            _logger.warning("пропущено, формат не читаем: %s", path.name)
    if not found:
        raise FileNotFoundError(f"{folder} holds no {', '.join(SUFFIXES)}")
    return found


def place(
    paths: list[Path], predictor: PackagePredictor, unusual: float, unreadable: float
) -> Iterator[Placed]:
    """Scores every photo and says what the service would do with it.

    Args:
        paths: Photos to score.
        predictor: Loaded package.
        unusual: Share of the band from which a photo goes to a person.
        unreadable: Share from which the pages show no prediction.

    Yields:
        One `Placed` per photo that could be read; unreadable files are logged and skipped.
    """
    for path in paths:
        try:
            image = prepare_image(path.read_bytes(), LIMITS)
        # A folder of files, not an API request: one unreadable file must not stop the run.
        except Exception as error:
            _logger.warning("  %s: %s", path.name, error)
            continue
        [prediction] = predictor.predict([image])
        score, edge, empty = (
            prediction.ood_score,
            prediction.ood_threshold,
            prediction.ood_unusable,
        )
        if score is None or edge is None or empty is None or empty <= edge:
            _logger.warning("  %s: the package measures no distances", path.name)
            continue
        share = (score - edge) / (empty - edge)
        verdict = (
            "нечитаемо" if share >= unreadable else ("на проверку" if share >= unusual else "норма")
        )
        yield Placed(
            name=path.name,
            load_pct=prediction.load_pct.value,
            score=score,
            share=share,
            verdict=verdict,
        )


def main() -> int:
    """Prints the band of a package and where the photos of a folder fall in it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", type=Path, help="folder with photos to place")
    parser.add_argument(
        "--package", type=Path, default=Path("storage/models/v4"), help="model package folder"
    )
    parser.add_argument("--device", default="cpu", help="torch device (default: %(default)s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = photos_in(args.photos)
    package = ModelPackage.load(args.package)
    predictor = PackagePredictor(package, device=args.device)
    review = load_settings().review

    placed = sorted(
        place(paths, predictor, review.unusual_band_share, review.unreadable_band_share),
        key=lambda row: row.share,
    )
    _logger.info("\nпакет %s, %d фото", args.package.name, len(placed))
    _logger.info(
        "линии: на проверку от доли %.2f, нечитаемо от %.2f",
        review.unusual_band_share,
        review.unreadable_band_share,
    )
    _logger.info("\n%-34s %8s %8s %7s  %s", "фото", "загрузка", "скор", "доля", "вердикт")
    for row in placed:
        _logger.info(
            "%-34s %7.1f%% %8.4f %7.2f  %s",
            row.name[:34],
            row.load_pct,
            row.score,
            row.share,
            row.verdict,
        )
    counted: dict[str, int] = {}
    for row in placed:
        counted[row.verdict] = counted.get(row.verdict, 0) + 1
    _logger.info("\nитого: %s", ", ".join(f"{name} {count}" for name, count in counted.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
