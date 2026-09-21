"""Heads over a fine-tuned encoder: how the number is made from what the encoder saw.

The shipped head is one linear layer over one vector per photo, and that vector is the
encoder's class token: the 1036 patch tokens of a 518x392 frame never reach it. The label,
though, is the share of the floor that cargo takes - a quantity that adds up over pieces of
the frame. `Occupancy` computes it that way instead of guessing it: one score per patch for
"is this floor", another for "is it taken", and the share is their weighted mean.

`Bins` answers with a distribution over the grid the expert actually marked on, which is what
a label quantized to 24 values and noisy by up to 10 points asks for. `Mlp` is the control:
if a hidden layer helps, the linear head was short of capacity, and if it does not, that
question is closed.
"""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import torch

# Huber turns quadratic below this error, in shares of 1. Five points is the grid the expert
# marked on, so smaller differences are noise and deserve a gentler gradient than L1 gives.
HUBER_DELTA: Final = 0.05

# The grid of the label: 0, 5, ... 100, as shares of 1.
BIN_COUNT: Final = 21

# Half-width of the bump a label is spread into, in shares of 1. The markers disagree by up to
# 10 points on near-identical frames, so a label is evidence for its neighbours too.
BIN_SPREAD: Final = 0.075


class Head(torch.nn.Module):
    """What every head over the encoder can do: answer, and score its own answer.

    Attributes:
        wants_tokens: Whether the head reads one token per patch instead of one vector per
            photo. `LoadRegressor` looks at this to decide what to hand over.
    """

    wants_tokens: bool = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns one share of 1 per photo."""
        raise NotImplementedError

    def loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Scores this head's answers against the labels, both as shares of 1."""
        raise NotImplementedError

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Runs the module; typed so callers see a tensor, not `Any`."""
        return super().__call__(features)


def bin_centres(device: torch.device | str = "cpu") -> torch.Tensor:
    """Returns the centre of every bin, as shares of 1."""
    return torch.linspace(0.0, 1.0, BIN_COUNT, device=device)


def _regression_loss(kind: str, answers: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Scores answers against labels, both as shares of 1."""
    if kind == "huber":
        return torch.nn.functional.huber_loss(answers, targets, delta=HUBER_DELTA)
    return torch.nn.functional.l1_loss(answers, targets)


def _logit(share: float) -> float:
    """Inverse sigmoid, kept away from the ends so a mean label of 0 or 1 cannot explode."""
    bounded = min(max(share, 1e-3), 1 - 1e-3)
    return math.log(bounded / (1 - bounded))


class Plain(Head):
    """One linear layer over one vector per photo: what the shipped models carry.

    `weight` and `bias` read through to that layer, because the package writer exports this
    head as the ridge head the service already loads.

    Args:
        width: Length of the vector the encoder hands over.
        start: Answer before any training, usually the mean training label as a share of 1.
        loss: Which regression loss to score with, `l1` or `huber`.
    """

    wants_tokens = False

    def __init__(self, width: int, start: float, *, loss: str = "l1") -> None:
        super().__init__()
        self.layer = torch.nn.Linear(width, 1)
        torch.nn.init.zeros_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, start)
        self._loss = loss

    @property
    def in_features(self) -> int:
        """Length of the vector this head reads."""
        return self.layer.in_features

    @property
    def weight(self) -> torch.Tensor:
        """Weights of the single layer."""
        return self.layer.weight

    @property
    def bias(self) -> torch.Tensor:
        """Bias of the single layer."""
        return self.layer.bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns one share of 1 per photo."""
        return self.layer(features).squeeze(1)

    def loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Scores this head's answers against the labels."""
        return _regression_loss(self._loss, self(features).float(), targets)


class Mlp(Head):
    """A hidden layer between the vector and the answer, as a test of capacity.

    Args:
        width: Length of the vector the encoder hands over.
        start: Answer before any training, as a share of 1.
        loss: Which regression loss to score with.
        hidden: Width of the hidden layer.
    """

    wants_tokens = False

    def __init__(self, width: int, start: float, *, loss: str = "l1", hidden: int = 256) -> None:
        super().__init__()
        last = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.constant_(last.bias, start)
        self.body = torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.GELU(), last)
        self._loss = loss

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns one share of 1 per photo."""
        return self.body(features).squeeze(1)

    def loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Scores this head's answers against the labels."""
        return _regression_loss(self._loss, self(features).float(), targets)


class Occupancy(Head):
    """Counts the share of the floor that is taken, instead of guessing it from one vector.

    Every patch of the frame gets two scores: how much of the floor it is, and how taken it
    is. The answer is the second averaged by the first, which is the definition of the label
    written into the architecture rather than learned from 716 photos. Both scores are bounded
    by construction, so the answer cannot leave 0-1 and needs no clipping.

    At the start the weights are zero and the bias of the occupancy score is the mean label,
    so the head answers that mean for every photo - the same warm start the linear head gets.

    Args:
        width: Length of one patch token.
        start: Answer before any training, as a share of 1.
        loss: Which regression loss to score with.
    """

    wants_tokens = True

    def __init__(self, width: int, start: float, *, loss: str = "l1") -> None:
        super().__init__()
        self.where = torch.nn.Linear(width, 1)
        self.taken = torch.nn.Linear(width, 1)
        torch.nn.init.zeros_(self.where.weight)
        torch.nn.init.zeros_(self.where.bias)
        torch.nn.init.zeros_(self.taken.weight)
        torch.nn.init.constant_(self.taken.bias, _logit(start))
        self._loss = loss

    def maps(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the floor weight and the occupancy of every patch, both 0 to 1.

        Args:
            tokens: Patch tokens, shape (photos, patches, width).

        Returns:
            Two tensors of shape (photos, patches): the weights, which sum to 1 per photo,
            and the occupancy of each patch.
        """
        weight = torch.softmax(self.where(tokens).squeeze(-1).float(), dim=1)
        return weight, torch.sigmoid(self.taken(tokens).squeeze(-1).float())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns one share of 1 per photo, from the patch tokens of the frame."""
        weight, taken = self.maps(features)
        return (weight * taken).sum(dim=1)

    def loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Scores this head's answers against the labels."""
        return _regression_loss(self._loss, self(features), targets)


class Bins(Head):
    """Answers with a distribution over the grid the expert marked on.

    A label of 45 is evidence that the load is near 45, not that it is exactly 45: the grid is
    five points wide and two markers disagree by up to ten. So the target is a bump around the
    label rather than a single bin, the head is trained to match that bump, and the answer is
    the mean of the distribution it predicts.

    Args:
        width: Length of the vector the encoder hands over.
        start: Answer before any training, as a share of 1.
        loss: Ignored; this head carries its own.
    """

    wants_tokens = False
    centres: torch.Tensor

    def __init__(self, width: int, start: float, *, loss: str = "l1") -> None:
        super().__init__()
        del loss
        self.layer = torch.nn.Linear(width, BIN_COUNT)
        centres = bin_centres()
        torch.nn.init.zeros_(self.layer.weight)
        with torch.no_grad():
            self.layer.bias.copy_(-0.5 * ((centres - start) / BIN_SPREAD) ** 2)
        self.register_buffer("centres", centres)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns one share of 1 per photo, the mean of the predicted distribution."""
        shares = torch.softmax(self.layer(features).float(), dim=1)
        return shares @ self.centres.to(dtype=shares.dtype)

    def loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Scores the predicted distribution against a bump around the label."""
        centres = self.centres.to(targets.dtype)
        wanted = torch.softmax(
            -0.5 * ((centres[None, :] - targets[:, None]) / BIN_SPREAD) ** 2, dim=1
        )
        logged = torch.log_softmax(self.layer(features).float(), dim=1)
        return -(wanted * logged).sum(dim=1).mean()


HEADS: Final[Mapping[str, type[Head]]] = MappingProxyType(
    {"linear": Plain, "mlp": Mlp, "occupancy": Occupancy, "bins": Bins}
)

LOSSES: Final = ("l1", "huber")


def make_head(kind: str, width: int, start: float, *, loss: str = "l1") -> Head:
    """Builds the head a config names.

    Args:
        kind: Key in `HEADS`.
        width: Length of what the encoder hands over: the vector, or one patch token.
        start: Answer before any training, as a share of 1.
        loss: Which regression loss to score with, for the heads that take one.

    Returns:
        The head, warm started to answer `start` for every photo.

    Raises:
        ValueError: If the kind or the loss is unknown.
    """
    if kind not in HEADS:
        raise ValueError(f"unknown head {kind!r}, known: {', '.join(sorted(HEADS))}")
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}, known: {', '.join(LOSSES)}")
    return HEADS[kind](width, start, loss=loss)
