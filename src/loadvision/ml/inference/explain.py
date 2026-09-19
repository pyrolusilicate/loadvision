"""Maps of what the encoder looks at when it answers: attention and contribution per patch."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Self, cast

import numpy as np
import torch
from numpy.typing import NDArray

from loadvision.imaging.prepared_image import PreparedImage
from loadvision.imaging.views import IDENTITY, get
from loadvision.ml.inference.predictor import RidgeHead
from loadvision.ml.models.backbones import Backbone, load_backbone
from loadvision.ml.packaging.manifest import Runtime
from loadvision.ml.packaging.model_package import ModelPackage, ModelPackageError

# The strongest percent of patches sets the colour scale, so one hot patch does not wash out
# the rest of the map.
_SCALE_PERCENTILE: Final = 99.0


@dataclass(frozen=True, slots=True)
class Explanation:
    """Per-patch maps of one photo, on the patch grid of the encoder's input box.

    Attributes:
        attention: How much the class token looks at every patch in the last block, averaged
            over heads; 0 to 1, where 1 is the most looked-at patch.
        contribution: How much every patch pushes the answer up (positive) or down (negative),
            from -1 to 1; None when it was not asked for.
        box_height: Height of the input box the grid covers, in pixels.
        box_width: Width of the input box, in pixels.
    """

    attention: NDArray[np.float32]
    contribution: NDArray[np.float32] | None
    box_height: int
    box_width: int


class Explainer:
    """Explains the answers of one torch member of a model package.

    Attention costs about nothing on top of the forward pass. Contribution needs a backward
    pass, which roughly doubles the time, so it is asked for per call.

    The explainer keeps its own copy of the encoder: it installs hooks for the length of a call,
    and must not disturb the predictor answering other requests at the same time.

    Args:
        backbone: Encoder of the member, with the weights the package carries.
        head: Head of that member.
        device: Torch device the encoder is on.
    """

    def __init__(self, backbone: Backbone, head: RidgeHead, device: str) -> None:
        self._backbone = backbone
        self._device = device
        self._module = cast(Any, backbone.module)
        self._prefix = int(getattr(self._module, "num_prefix_tokens", 0))
        self._rows = backbone.spec.height // backbone.spec.patch
        self._cols = backbone.spec.width // backbone.spec.patch
        self._mean = torch.tensor(head.mean, dtype=torch.float32, device=device)
        self._scale = torch.tensor(head.scale, dtype=torch.float32, device=device)
        self._weights = torch.tensor(head.weights, dtype=torch.float32, device=device)
        self._intercept = head.intercept

    @classmethod
    def from_package(cls, package: ModelPackage, device: str = "cpu") -> Self:
        """Loads the first torch member of a package.

        Args:
            package: Verified model package.
            device: Torch device.

        Returns:
            The explainer.

        Raises:
            ModelPackageError: If the package has no torch member.
        """
        member = next((m for m in package.manifest.members if m.runtime is Runtime.TORCH), None)
        if member is None:
            raise ModelPackageError("the package has no torch member to explain")
        weights = None if member.weights is None else package.path(member.weights)
        backbone = load_backbone(member.backbone, device, weights)
        return cls(backbone, RidgeHead.load(package.path(member.head)), device)

    def explain(self, image: PreparedImage, *, contribution: bool = True) -> Explanation:
        """Computes the maps of one photo.

        Args:
            image: Decoded photo.
            contribution: Also compute the signed contribution map; about twice as slow.

        Returns:
            The maps on the patch grid.
        """
        pixels = self._backbone.prepare([image], get(IDENTITY)).to(self._device)
        seen: dict[str, torch.Tensor] = {}
        blocks = self._module.blocks
        hooks = [blocks[-1].attn.register_forward_pre_hook(self._keep(seen, "attn"))]
        if contribution:
            hooks.append(blocks[-1].register_forward_pre_hook(self._keep(seen, "block", True)))
        try:
            if contribution:
                self._answer(pixels.requires_grad_(True)).sum().backward()
            else:
                with torch.inference_mode():
                    self._backbone.tokens(pixels)
        finally:
            for hook in hooks:
                hook.remove()
        with torch.inference_mode():
            attention = self._attention(seen["attn"].detach())
        signed = self._contribution(seen["block"]) if contribution else None
        return Explanation(
            attention=attention,
            contribution=signed,
            box_height=self._backbone.spec.height,
            box_width=self._backbone.spec.width,
        )

    def _answer(self, pixels: torch.Tensor) -> torch.Tensor:
        pooled = self._backbone.pool(self._backbone.tokens(pixels))
        return (pooled - self._mean) / self._scale @ self._weights + self._intercept

    @staticmethod
    def _keep(
        seen: dict[str, torch.Tensor], name: str, grad: bool = False
    ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...]], None]:
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if grad:
                inputs[0].retain_grad()
            seen[name] = inputs[0]

        return hook

    def _attention(self, x: torch.Tensor) -> NDArray[np.float32]:
        """Recomputes the last block's attention weights; the fused kernel does not return them."""
        attn = self._module.blocks[-1].attn
        n = x.shape[1]
        qkv = attn.qkv(x).reshape(1, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k = attn.q_norm(qkv[0]), attn.k_norm(qkv[1])
        weights = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
        grid = weights[0, :, 0, self._prefix :].mean(0).float().cpu().numpy()
        return _normalized(grid.reshape(self._rows, self._cols), signed=False)

    def _contribution(self, x: torch.Tensor) -> NDArray[np.float32]:
        """Gradient times activation of the patch tokens entering the last block."""
        assert x.grad is not None  # retained by the hook, filled by backward
        relevance = (x.grad * x).sum(-1)[0, self._prefix :].detach().float().cpu().numpy()
        return _normalized(relevance.reshape(self._rows, self._cols), signed=True)


def _normalized(grid: NDArray[Any], *, signed: bool) -> NDArray[np.float32]:
    scale = float(np.percentile(np.abs(grid), _SCALE_PERCENTILE))
    if scale <= 0:
        return np.zeros_like(grid, dtype=np.float32)
    low = -1.0 if signed else 0.0
    return np.clip(grid / scale, low, 1.0).astype(np.float32)
