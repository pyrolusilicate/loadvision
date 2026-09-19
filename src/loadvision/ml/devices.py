"""The torch device a model runs on when the settings leave the choice to the machine."""

from typing import Final

import torch

# The settings value that lets the machine choose.
AUTO_DEVICE: Final = "auto"


def resolve_device(device: str) -> str:
    """Returns the torch device to run on.

    `auto` takes the GPU when torch sees one and the CPU otherwise, so one settings file serves
    a laptop, the images and the GPU server. Any other value, such as ``cpu``, ``cuda`` or
    ``mps``, is returned as given.

    Args:
        device: Device from the settings.
    """
    if device != AUTO_DEVICE:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
