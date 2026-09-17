"""Loading pretrained encoders together with the preprocessing that matches them."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, cast

import timm
import torch
from timm.data.config import resolve_model_data_config

from loadvision.imaging.preprocessing import Params
from loadvision.ml.registry import BACKBONES, BackboneSpec

PREPROCESSING_NAME: Final = "resize_pad"

_CHANNELS: Final = ("r", "g", "b")


@dataclass(frozen=True, slots=True, kw_only=True)
class Backbone:
    """A loaded encoder and everything needed to feed it a photo.

    Attributes:
        name: Registry key; goes into the model package manifest as `Member.backbone`.
        module: Encoder in eval mode on the requested device, without a classification head.
        params: Preprocessing parameters for `PREPROCESSING_NAME`, taken from the checkpoint,
            so training and the service normalize a photo exactly the same way.
        dim: Length of one embedding vector.
        spec: Registry entry the encoder was built from.
    """

    name: str
    module: torch.nn.Module
    params: Params
    dim: int
    spec: BackboneSpec

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Turns a batch of prepared photos into one vector each.

        Args:
            batch: Float tensor of shape (photos, 3, height, width) on the encoder's device.

        Returns:
            Float tensor of shape (photos, `dim`).
        """
        if self.spec.pooling == "pooled":
            return self.module(batch)
        forward_features = cast(
            Callable[[torch.Tensor], torch.Tensor], self.module.forward_features
        )
        tokens = forward_features(batch)
        prefix = int(getattr(self.module, "num_prefix_tokens", 0))
        patches = tokens[:, prefix:, :]
        rows = self.spec.height // self.spec.patch
        cols = self.spec.width // self.spec.patch
        grid = patches.reshape(batch.shape[0], rows, cols, -1).permute(0, 3, 1, 2)
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            grid, (self.spec.grid_rows, self.spec.grid_cols)
        )
        return pooled.flatten(1)


def load_backbone(name: str, device: str) -> Backbone:
    """Loads a pretrained encoder from the registry.

    Args:
        name: Registry key from `loadvision.ml.registry.BACKBONES`.
        device: Torch device, such as "mps", "cuda" or "cpu".

    Returns:
        Encoder ready for `embed_images`.

    Raises:
        KeyError: If the name is not in the registry.
    """
    spec: BackboneSpec = BACKBONES[name]
    module = timm.create_model(
        spec.timm_name,
        pretrained=True,
        num_classes=0,
        img_size=(spec.height, spec.width),
    )
    module.eval().to(device)
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    config = resolve_model_data_config(module)
    params: dict[str, float] = {"height": spec.height, "width": spec.width}
    for index, channel in enumerate(_CHANNELS):
        params[f"mean_{channel}"] = float(config["mean"][index])
        params[f"std_{channel}"] = float(config["std"][index])

    width = cast(int, module.num_features)
    dim = width if spec.pooling == "pooled" else width * spec.grid_rows * spec.grid_cols
    return Backbone(name=name, module=module, params=params, dim=dim, spec=spec)
