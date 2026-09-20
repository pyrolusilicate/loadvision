"""Fine-tuning an encoder end to end on the load percent, measured out of fold."""

import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Final, Self, cast

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file
from timm.optim._param_groups import param_groups_layer_decay

from loadvision.imaging.crops import far_end
from loadvision.imaging.prepared_image import PreparedImage
from loadvision.imaging.preprocessing import preprocess
from loadvision.ml.data.manifest import Sample
from loadvision.ml.models.backbones import Backbone, load_backbone
from loadvision.ml.models.depth import DEPTH_DIM, depth_grids, mirror_grids
from loadvision.ml.training.embedding_cache import read_image

logger = logging.getLogger(__name__)

_PERCENT: Final = 100.0

_FULL: Final = "full"
_FAR: Final = "far"
_DEPTH: Final = "depth"

ENCODER_FILE: Final = "encoder.safetensors"
HEAD_FILE: Final = "head.npz"

type Loader = Callable[[str, str], Backbone]


@dataclass(frozen=True, slots=True, kw_only=True)
class FinetuneSettings:
    """How one encoder is fine-tuned.

    The defaults follow the usual recipe for adapting a pretrained ViT to a few hundred
    photos: AdamW, layer-wise learning-rate decay so the early layers barely move, a short
    warmup and a cosine decay. The head starts at the mean label and learns ten times faster
    than the encoder, because it starts from nothing.

    Attributes:
        encoder: Key in `ml.registry.BACKBONES`; decides the checkpoint, the input box, the
            preprocessing and whether tokens are pooled into one vector or a grid.
        epochs: Passes over the training photos of a fold.
        batch_size: Photos per optimizer step.
        lr: Learning rate of the last encoder block.
        head_lr_scale: How many times faster the head learns than the last block.
        layer_decay: Factor applied per block going down; 0.75 leaves the first block at a
            few percent of `lr`.
        weight_decay: AdamW decay of the encoder weights; the head has none.
        warmup_epochs: Epochs of linear warmup before the cosine decay.
        flip: Mirror training photos at random; the share of the floor does not change.
        seed: Seed of the first fold; fold N uses `seed + N`.
        workers: Processes decoding photos for the loader; 0 decodes in the main process.
        far_view: Also feed the encoder a zoomed crop of the far end of the body and give the
            head both vectors; costs a second run of the encoder.
        depth: Also give the head a coarse grid of relative depth from a frozen Depth
            Anything V2 Small, computed once per photo.
        checkpointing: Recompute block activations in the backward pass instead of keeping
            them: a third slower, a fraction of the memory, for inputs that would not fit.
    """

    encoder: str
    epochs: int = 15
    batch_size: int = 16
    lr: float = 5e-5
    head_lr_scale: float = 10.0
    layer_decay: float = 0.75
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    flip: bool = True
    seed: int = 0
    workers: int = 4
    far_view: bool = False
    depth: bool = False
    checkpointing: bool = False

    def __post_init__(self) -> None:
        """Validates the numbers.

        Raises:
            ValueError: If a count or a rate is not positive, or `layer_decay` is outside
                (0, 1].
        """
        for name in ("epochs", "batch_size", "lr", "head_lr_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.layer_decay <= 1:
            raise ValueError("layer_decay must be in (0, 1]")
        if self.weight_decay < 0 or self.warmup_epochs < 0 or self.workers < 0:
            raise ValueError("weight_decay, warmup_epochs and workers must not be negative")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        """Reads settings from a JSON config.

        Raises:
            ValueError: If `encoder` is missing, a key is unknown or a value is invalid. An
                unknown key is an error because a typo would otherwise switch a setting off.
        """
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(unknown)}")
        if "encoder" not in raw:
            raise ValueError("settings need an 'encoder'")
        return cls(**dict(raw))


class LoadRegressor(torch.nn.Module):
    """An encoder from the registry with a linear head that answers a share of 1.

    Pooling is the registry entry's own, through `Backbone.encode`, so a grid entry keeps its
    grid and the fine-tuned model feeds its head exactly what the frozen pipeline does. With
    the far view the same encoder also embeds the crop, and with depth the depth grid joins
    the head input.

    Args:
        backbone: Loaded encoder, trainable.
        start: Initial answer as a share of 1, usually the mean training label.
        far_view: Whether the input carries the crop of the far end.
        depth: Whether the input carries the depth grid.
    """

    def __init__(
        self, backbone: Backbone, start: float, *, far_view: bool = False, depth: bool = False
    ) -> None:
        super().__init__()
        self._backbone = backbone
        self.encoder = backbone.module
        self.far_view = far_view
        self.depth = depth
        width = backbone.dim * (2 if far_view else 1) + (DEPTH_DIM if depth else 0)
        self.head = torch.nn.Linear(width, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.constant_(self.head.bias, start)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Returns one share of 1 per photo."""
        parts = [self._backbone.encode(inputs[_FULL])]
        if self.far_view:
            parts.append(self._backbone.encode(inputs[_FAR]))
        if self.depth:
            parts.append(inputs[_DEPTH].to(parts[0].dtype))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


type FoldDone = Callable[[int, LoadRegressor, NDArray[np.float64]], None]


class _Photos(torch.utils.data.Dataset[tuple[dict[str, torch.Tensor], torch.Tensor]]):
    """Photos decoded by the service decoder and prepared by the registry preprocessing."""

    def __init__(
        self,
        samples: Sequence[Sample],
        labels: NDArray[np.float64],
        backbone: Backbone,
        *,
        flip: bool,
        far_view: bool = False,
        depth: NDArray[np.float32] | None = None,
    ) -> None:
        self._paths = [sample.path for sample in samples]
        self._labels = labels
        self._preprocessing = backbone.preprocessing
        self._params = dict(backbone.params)
        self._flip = flip
        self._far_view = far_view
        self._depth = depth

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        image = read_image(self._paths[index])
        inputs = {_FULL: self._prepare(image)}
        if self._far_view:
            inputs[_FAR] = self._prepare(far_end(image))
        if self._depth is not None:
            inputs[_DEPTH] = torch.from_numpy(self._depth[index : index + 1])
        # The crop is centred, so mirroring the photo mirrors the crop as well.
        if self._flip and bool(torch.rand(()) < 0.5):
            inputs = {
                key: mirror_grids(value) if key == _DEPTH else torch.flip(value, dims=(2,))
                for key, value in inputs.items()
            }
        if _DEPTH in inputs:
            inputs[_DEPTH] = inputs[_DEPTH][0]
        label = torch.tensor(self._labels[index] / _PERCENT, dtype=torch.float32)
        return inputs, label

    def _prepare(self, image: PreparedImage) -> torch.Tensor:
        return torch.from_numpy(preprocess(self._preprocessing, image, self._params)).float()


def fit(
    samples: Sequence[Sample],
    labels: NDArray[np.float64],
    settings: FinetuneSettings,
    *,
    device: str,
    load: Loader | None = None,
) -> LoadRegressor:
    """Fine-tunes a fresh copy of the encoder on the given photos.

    Args:
        samples: Training photos.
        labels: Their loads, 0 to 100.
        settings: How to fine-tune.
        device: Torch device, such as "cuda", "mps" or "cpu".
        load: Builds the encoder from its registry key; the pretrained loader by default.

    Returns:
        The fine-tuned model in eval mode.

    Raises:
        ValueError: If there are no photos or the labels do not match them.
    """
    if not samples or len(samples) != len(labels):
        raise ValueError(f"cannot fine-tune on {len(samples)} photos and {len(labels)} labels")
    _seed(settings.seed)
    backbone = (load or load_backbone)(settings.encoder, device)
    backbone.module.train()
    backbone.module.requires_grad_(True)
    if settings.checkpointing:
        cast(Any, backbone.module).set_grad_checkpointing(True)
    model = LoadRegressor(
        backbone,
        start=float(np.mean(labels)) / _PERCENT,
        far_view=settings.far_view,
        depth=settings.depth,
    ).to(device)

    photos = _Photos(
        samples,
        labels,
        backbone,
        flip=settings.flip,
        far_view=settings.far_view,
        depth=depth_grids(samples, device) if settings.depth else None,
    )
    loader = torch.utils.data.DataLoader(
        photos,
        batch_size=settings.batch_size,
        shuffle=True,
        drop_last=len(photos) > settings.batch_size,
        num_workers=settings.workers,
        generator=torch.Generator().manual_seed(settings.seed),
    )
    optimizer = _optimizer(model, settings)
    steps = settings.epochs * len(loader)
    schedule = _schedule(optimizer, steps, round(settings.warmup_epochs * len(loader)))

    for epoch in range(settings.epochs):
        total, count = 0.0, 0
        for inputs, targets in loader:
            with _autocast(device):
                answers = model(_to(inputs, device))
            loss = torch.nn.functional.l1_loss(answers.float(), targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            total += float(loss.item()) * len(targets)
            count += len(targets)
        logger.info(
            "epoch %d/%d  train MAE %.2f", epoch + 1, settings.epochs, _PERCENT * total / count
        )
    model.eval()
    return model


@torch.inference_mode()
def predict(
    model: LoadRegressor,
    samples: Sequence[Sample],
    *,
    device: str,
    batch_size: int,
    workers: int = 0,
) -> NDArray[np.float64]:
    """Predicts the load of every photo, clipped to 0-100.

    Args:
        model: Fine-tuned model.
        samples: Photos to predict.
        device: Torch device the model is on.
        batch_size: Photos per forward pass.
        workers: Processes decoding photos.

    Returns:
        One load per photo, in the order of `samples`.
    """
    photos = _Photos(
        samples,
        np.zeros(len(samples)),
        model._backbone,
        flip=False,
        far_view=model.far_view,
        depth=depth_grids(samples, device) if model.depth else None,
    )
    loader = torch.utils.data.DataLoader(photos, batch_size=batch_size, num_workers=workers)
    answers: list[torch.Tensor] = []
    for inputs, _ in loader:
        with _autocast(device):
            answers.append(model(_to(inputs, device)).float().cpu())
    shares = torch.cat(answers).numpy().astype(np.float64)
    return np.clip(shares * _PERCENT, 0.0, _PERCENT)


def save_model(model: LoadRegressor, folder: Path) -> tuple[Path, Path]:
    """Writes a fine-tuned model in the two files a model package carries for a member.

    The encoder goes out as a timm state dict, the form `load_backbone` reads from a package
    instead of downloading the pretrained checkpoint. The head goes out as the ridge head the
    service already reads: no standardization, and the weights scaled from a share of 1 to
    percent, so `RidgeHead.apply` answers exactly what this model does.

    Args:
        model: Fine-tuned model.
        folder: Where to write; created if missing.

    Returns:
        Paths of the encoder file and the head file.
    """
    folder.mkdir(parents=True, exist_ok=True)
    encoder = folder / ENCODER_FILE
    state = {
        key: value.detach().float().cpu().contiguous()
        for key, value in model.encoder.state_dict().items()
    }
    save_file(state, str(encoder))
    weights = model.head.weight.detach().float().cpu().numpy().ravel().astype(np.float64)
    bias = float(model.head.bias.detach().float().cpu().item())
    head = folder / HEAD_FILE
    np.savez(
        head,
        mean=np.zeros_like(weights),
        scale=np.ones_like(weights),
        weights=weights * _PERCENT,
        intercept=np.float64(bias * _PERCENT),
    )
    return encoder, head


def out_of_fold(
    samples: Sequence[Sample],
    labels: NDArray[np.float64],
    folds: Sequence[int],
    settings: FinetuneSettings,
    *,
    device: str,
    only: int | None = None,
    load: Loader | None = None,
    on_fold: FoldDone | None = None,
) -> NDArray[np.float64]:
    """Fine-tunes one model per fold and predicts the fold it never saw.

    Every fold starts from the pretrained weights again. Fine-tuning once on all photos and
    then scoring the folds would let the encoder see every validation photo, and the number
    would look better than any model we could ship.

    Args:
        samples: Labelled photos.
        labels: Their loads.
        folds: Fold number of every photo.
        settings: How to fine-tune.
        device: Torch device.
        only: Fold to run alone, for a quick check; all folds when None.
        load: Builds the encoder from its registry key.
        on_fold: Called after every fold with its model and the predictions so far, while the
            model is still in memory: the caller saves it and predicts the test split with it,
            so a run never has to be repeated to get either, and a stopped machine keeps the
            folds it finished.

    Returns:
        One prediction per photo; NaN for photos of folds that were not run.

    Raises:
        ValueError: If the lengths disagree or `only` is not a fold number.
    """
    numbers = np.asarray(folds)
    if not len(samples) == len(labels) == len(numbers):
        raise ValueError("samples, labels and folds must have one entry per photo")
    present = sorted({int(value) for value in numbers.tolist()})
    if only is not None and only not in present:
        raise ValueError(f"fold {only} is not one of {present}")

    predictions = np.full(len(samples), np.nan, dtype=np.float64)
    for fold in present if only is None else [only]:
        validation = numbers == fold
        training = [sample for sample, held in zip(samples, validation, strict=True) if not held]
        held_out = [sample for sample, held in zip(samples, validation, strict=True) if held]
        logger.info("fold %d: training on %d photos", fold, len(training))
        model = fit(
            training,
            labels[~validation],
            replace(settings, seed=settings.seed + fold),
            device=device,
            load=load,
        )
        predictions[validation] = predict(
            model, held_out, device=device, batch_size=settings.batch_size, workers=settings.workers
        )
        if on_fold is not None:
            on_fold(fold, model, predictions)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return predictions


def _optimizer(model: LoadRegressor, settings: FinetuneSettings) -> torch.optim.Optimizer:
    encoder = model.encoder
    if hasattr(encoder, "group_matcher"):
        groups = param_groups_layer_decay(
            encoder, weight_decay=settings.weight_decay, layer_decay=settings.layer_decay
        )
    else:
        # An encoder without timm's block layout gets one group; the tests use such a one.
        groups = [{"params": list(encoder.parameters()), "weight_decay": settings.weight_decay}]
    groups = [group for group in groups if group["params"]]
    for group in groups:
        group["lr"] = settings.lr * float(group.get("lr_scale", 1.0))
    groups.append(
        {
            "params": list(model.head.parameters()),
            "weight_decay": 0.0,
            "lr": settings.lr * settings.head_lr_scale,
        }
    )
    return torch.optim.AdamW(groups, lr=settings.lr)


def _schedule(
    optimizer: torch.optim.Optimizer, steps: int, warmup: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _to(inputs: Mapping[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def _autocast(device: str) -> AbstractContextManager[object]:
    # bf16 on the GPU halves memory and time; the L4 supports it. MPS and CPU stay in fp32.
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
