"""What conditional postprocessing steps look at, read from the photos themselves."""

from collections.abc import Sequence

import numpy as np

from loadvision.imaging.quality import aspect
from loadvision.ml.data.manifest import Sample
from loadvision.ml.postprocess import ASPECT, Facts
from loadvision.ml.training.embedding_cache import read_image


def photo_facts(samples: Sequence[Sample]) -> Facts:
    """Reads the facts of every photo, in the order of `samples`.

    The pipeline takes these from the `manifest.csv` of its run; commands and the package
    builder have no run, so they decode the photos. Both go through the same decoder and the
    same definition of proportions, so a chain fitted here behaves the same inside the package.

    Args:
        samples: Photos, in prediction order.

    Returns:
        Fact name to one number per photo.

    Raises:
        LoadvisionError: If a photo cannot be decoded.
    """
    return {ASPECT: np.array([aspect(read_image(s.path)) for s in samples], dtype=np.float64)}
