"""Name-to-factory registries for backbones and heads used by experiment configs."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

# A key names one pretrained checkpoint at one input size forever: it is written into a model
# package manifest, and the service resolves the package by it long after the experiment.
DEFAULT_BACKBONE: Final = "dinov2_vitb14"

# Letterboxing into the encoder box; `imaging/preprocessing.py` holds the alternatives.
DEFAULT_PREPROCESSING: Final = "resize_pad"

type Pooling = Literal["pooled", "grid"]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackboneSpec:
    """Pretrained encoder and the input box its embeddings are computed in.

    Attributes:
        timm_name: Checkpoint name in the timm model hub.
        patch: Side of one patch in pixels; the box must be a whole number of patches.
        height: Height of the input box in pixels.
        width: Width of the input box in pixels.
        preprocessing: Key in the preprocessing registry of `imaging/preprocessing.py`. It is
            part of the entry because the model package stores it per ensemble member, so
            training must be able to choose it the same way.
        pooling: How patch tokens become one vector. "pooled" is the encoder's own pooling and
            describes the scene as a whole; "grid" averages tokens inside a coarse grid, which
            keeps where the cargo is and how much floor it covers.
        grid_rows: Rows of the grid, used by "grid" pooling.
        grid_cols: Columns of the grid, used by "grid" pooling.
    """

    timm_name: str
    patch: int
    height: int
    width: int
    preprocessing: str = DEFAULT_PREPROCESSING
    pooling: Pooling = "pooled"
    grid_rows: int = 1
    grid_cols: int = 1

    def __post_init__(self) -> None:
        """Checks the box against the patch grid.

        Raises:
            ValueError: If a side is not a positive multiple of `patch`.
        """
        for side, value in (("height", self.height), ("width", self.width)):
            if value <= 0 or value % self.patch:
                raise ValueError(
                    f"{self.timm_name}: {side} {value} is not a multiple of the "
                    f"{self.patch}px patch"
                )
        if self.pooling != "grid":
            return
        # Torch on MPS pools only into a grid that divides the patch grid evenly.
        for side, cells, patches in (
            ("rows", self.grid_rows, self.height // self.patch),
            ("cols", self.grid_cols, self.width // self.patch),
        ):
            if cells < 1 or patches % cells:
                raise ValueError(
                    f"{self.timm_name}: grid {side} {cells} does not divide the {patches} "
                    f"patches of that side"
                )


# The box is 3:4 and not square: dataset photos are 3:4 at the median, so a square box would
# spend a quarter of the input on black bars, and patches are cheaper spent on the cargo.
# A checkpoint may appear more than once: the box is part of the key, because the same encoder
# at another resolution is a different candidate and gets its own embedding cache.
BACKBONES: Final = MappingProxyType(
    {
        "dinov2_vitb14": BackboneSpec(
            timm_name="vit_base_patch14_reg4_dinov2.lvd142m", patch=14, height=518, width=392
        ),
        "dinov3_vitb16": BackboneSpec(
            timm_name="vit_base_patch16_dinov3.lvd1689m", patch=16, height=512, width=384
        ),
        # DINOv3 was pretrained at 256, so this entry keeps its native scale for comparison.
        "dinov3_vitb16_256": BackboneSpec(
            timm_name="vit_base_patch16_dinov3.lvd1689m", patch=16, height=256, width=192
        ),
        "pe_spatial_vitb16": BackboneSpec(
            timm_name="vit_pe_spatial_base_patch16_512.fb", patch=16, height=512, width=384
        ),
        "siglip2_vitb16": BackboneSpec(
            timm_name="vit_base_patch16_siglip_512.v2_webli", patch=16, height=512, width=384
        ),
        "siglip2_vitb16_grid4x3": BackboneSpec(
            timm_name="vit_base_patch16_siglip_512.v2_webli",
            patch=16,
            height=512,
            width=384,
            pooling="grid",
            grid_rows=4,
            grid_cols=3,
        ),
        # Grid variants keep where the cargo is: the label is the share of floor it covers, and
        # a single scene vector cannot tell three boxes in the middle from a half-full body.
        "dinov3_vitb16_grid4x3": BackboneSpec(
            timm_name="vit_base_patch16_dinov3.lvd1689m",
            patch=16,
            height=512,
            width=384,
            pooling="grid",
            grid_rows=4,
            grid_cols=3,
        ),
        "pe_spatial_vitb16_grid2x2": BackboneSpec(
            timm_name="vit_pe_spatial_base_patch16_512.fb",
            patch=16,
            height=512,
            width=384,
            pooling="grid",
            grid_rows=2,
            grid_cols=2,
        ),
        # Same encoders with a neutral padding instead of black: half of the dataset has
        # proportions far from the box, so a wide band of the input is padding.
        "dinov3_vitb16_padmean": BackboneSpec(
            timm_name="vit_base_patch16_dinov3.lvd1689m",
            patch=16,
            height=512,
            width=384,
            preprocessing="resize_pad_mean",
        ),
        "pe_spatial_vitb16_padmean": BackboneSpec(
            timm_name="vit_pe_spatial_base_patch16_512.fb",
            patch=16,
            height=512,
            width=384,
            preprocessing="resize_pad_mean",
        ),
        "pe_spatial_vitb16_grid4x3_padmean": BackboneSpec(
            timm_name="vit_pe_spatial_base_patch16_512.fb",
            patch=16,
            height=512,
            width=384,
            preprocessing="resize_pad_mean",
            pooling="grid",
            grid_rows=4,
            grid_cols=3,
        ),
        "pe_spatial_vitb16_grid4x3": BackboneSpec(
            timm_name="vit_pe_spatial_base_patch16_512.fb",
            patch=16,
            height=512,
            width=384,
            pooling="grid",
            grid_rows=4,
            grid_cols=3,
        ),
    }
)
