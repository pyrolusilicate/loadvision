"""Frames that hold no truck body at all, used to measure the scale of a model package.

The distance a package reports means nothing on its own: one encoder calls 0.1 far away and
another calls 0.3 close by, and fine-tuning moves the scale again. A package already carries
one point of its own scale - the threshold, which sits at the far edge of its training photos.
These frames give the second point: whatever a model answers for a grey wall or pure noise is
where "this is not a truck body" starts on that model's scale. With two points the service
places its lines by itself, instead of carrying a number somebody measured once by hand.

The frames are built in code rather than stored: they must be the same on every machine and in
every release, or two services would measure two different scales.
"""

import hashlib
from typing import Final

import numpy as np
from numpy.typing import NDArray

from loadvision.imaging.prepared_image import MediaType, PreparedImage

# The shape of a photo of a truck body: portrait, and large enough that a preprocessing which
# crops or pads behaves the way it does on a real photo.
FRAME_HEIGHT: Final = 800
FRAME_WIDTH: Final = 600
# Fixed, so the noise frame is the same frame everywhere.
NOISE_SEED: Final = 0


def _flat(level: int) -> NDArray[np.uint8]:
    return np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), level, dtype=np.uint8)


def unusable_frames() -> list[PreparedImage]:
    """Returns the frames a working model must place far from its training photos.

    A closed lens, a frame shot in the dark, a blown-out frame and sensor noise: four ways a
    photo arrives with nothing in it. They are deliberately the easy cases - the point is to
    find where "nothing" starts, not to catch every bad photo.

    Returns:
        Four photos, ready for a predictor.
    """
    noise = np.random.default_rng(NOISE_SEED).integers(
        0, 256, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
    )
    return [
        PreparedImage(b"grey-frame", MediaType.PNG, _flat(128)),
        PreparedImage(b"dark-frame", MediaType.PNG, _flat(8)),
        PreparedImage(b"blown-frame", MediaType.PNG, _flat(250)),
        PreparedImage(b"noise-frame", MediaType.PNG, noise),
    ]


def frames_fingerprint() -> str:
    """Identifies the set of frames a measurement was made with.

    A measurement is only worth keeping while the frames behind it are the same ones. Adding a
    frame or changing the noise seed must throw the stored number away, not silently keep it.

    Returns:
        Lowercase hex SHA-256 of the pixels of every frame, in order.
    """
    digest = hashlib.sha256()
    for frame in unusable_frames():
        digest.update(frame.pixels.tobytes())
    return digest.hexdigest()
