"""Loading pretrained encoders together with the preprocessing that matches them."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import timm
import torch
from timm.data.config import resolve_model_data_config

from loadvision.imaging.prepared_image import PreparedImage
from loadvision.imaging.preprocessing import Params, preprocess
from loadvision.imaging.views import View, get
from loadvision.ml.registry import BACKBONES, BackboneSpec

_CHANNELS: Final = ("r", "g", "b")


@dataclass(frozen=True, slots=True, kw_only=True)
class Backbone:
    """A loaded encoder and everything needed to feed it a photo.

    Attributes:
        name: Registry key; goes into the model package manifest as `Member.backbone`.
        module: Encoder in eval mode on the requested device, without a classification head.
        preprocessing: Key in the preprocessing registry, from the registry entry.
        params: Parameters of that preprocessing, taken from the checkpoint, so training and
            the service normalize a photo exactly the same way.
        dim: Length of one embedding vector.
        spec: Registry entry the encoder was built from.
    """

    name: str
    module: torch.nn.Module
    preprocessing: str
    params: Params
    dim: int
    spec: BackboneSpec

    def tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """Runs the encoder and returns its tokens, before any pooling.

        Kept apart from `pool` so that members sharing a checkpoint and an input box run the
        encoder once and then pool its tokens each in their own way. That is the whole cost of
        a prediction, so sharing it halves the latency of the current ensemble.

        Args:
            batch: Float tensor of shape (photos, 3, height, width) on the encoder's device.

        Returns:
            Token tensor of the encoder, prefix tokens included.
        """
        forward_features = cast(
            Callable[[torch.Tensor], torch.Tensor], self.module.forward_features
        )
        return forward_features(batch)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Turns encoder tokens into one vector per photo, the way this entry asks for.

        Args:
            tokens: Output of `tokens`.

        Returns:
            Float tensor of shape (photos, `dim`).
        """
        if self.spec.pooling == "pooled":
            forward_head = cast(Callable[[torch.Tensor], torch.Tensor], self.module.forward_head)
            return forward_head(tokens)
        prefix = int(getattr(self.module, "num_prefix_tokens", 0))
        patches = tokens[:, prefix:, :]
        rows = self.spec.height // self.spec.patch
        cols = self.spec.width // self.spec.patch
        grid = patches.reshape(tokens.shape[0], rows, cols, -1).permute(0, 3, 1, 2)
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            grid, (self.spec.grid_rows, self.spec.grid_cols)
        )
        return pooled.flatten(1)

    def unmirror(self, vectors: torch.Tensor) -> torch.Tensor:
        """Puts the columns of a grid back in reading order after a mirrored view.

        A mirror swaps left and right, so cell (row, column) of the mirrored photo holds what
        cell (row, last - column) of the original does. Averaging without this would blend the
        left of the body with its right and throw away the geometry the grid exists for. An
        encoder that pools into a single vector has no columns, so nothing happens.

        Args:
            vectors: Output of `pool` for a mirrored view.

        Returns:
            The same vectors with grid columns reversed.
        """
        if self.spec.pooling == "pooled":
            return vectors
        width = vectors.shape[1] // (self.spec.grid_rows * self.spec.grid_cols)
        grid = vectors.reshape(-1, width, self.spec.grid_rows, self.spec.grid_cols)
        return grid.flip(-1).reshape(vectors.shape[0], -1)

    @property
    def views(self) -> tuple[View, ...]:
        """Views this entry averages over; a single identity view means no augmentation."""
        return tuple(get(name) for name in self.spec.views)

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Turns a batch of prepared photos into one vector each, for a single view.

        Args:
            batch: Float tensor of shape (photos, 3, height, width) on the encoder's device.

        Returns:
            Float tensor of shape (photos, `dim`).
        """
        return self.pool(self.tokens(batch))

    def prepare(self, images: Sequence[PreparedImage], view: View) -> torch.Tensor:
        """Turns photos into the encoder's input for one view.

        Args:
            images: Decoded photos.
            view: How to look at them.

        Returns:
            Float tensor of shape (photos, 3, height, width) on the CPU.
        """
        batch = np.stack(
            [preprocess(self.preprocessing, view.make(image), self.params) for image in images]
        )
        return torch.from_numpy(batch)

    def embed(self, images: Sequence[PreparedImage], device: str) -> torch.Tensor:
        """Embeds photos, averaging over every view this entry declares.

        This is the one place the whole chain from a decoded photo to an embedding lives, so
        training and the service cannot drift apart on it.

        Args:
            images: Decoded photos.
            device: Torch device the encoder is on.

        Returns:
            Float tensor of shape (photos, `dim`).
        """
        total: torch.Tensor | None = None
        for view in self.views:
            vectors = self.encode(self.prepare(images, view).to(device))
            if view.mirrored:
                vectors = self.unmirror(vectors)
            total = vectors if total is None else total + vectors
        assert total is not None  # the registry refuses an empty view list
        return total / len(self.views)


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
    return Backbone(
        name=name,
        module=module,
        preprocessing=spec.preprocessing,
        params=params,
        dim=dim,
        spec=spec,
    )
