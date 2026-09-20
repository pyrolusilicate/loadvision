"""What a photo says about the cargo besides how much floor it takes.

The load percent alone does not weigh anything: a floor covered by one layer of sacks and the
same floor stacked to the roof look alike to it. Two more things are needed, and both are on
the photo. How high the cargo is stacked comes from the depth map the package already carries:
cargo stands nearer to the camera than the far wall, so the grid shows how far up it reaches.
What kind of cargo it is comes from a small vision-language model, asked to choose from a
fixed list rather than to describe the scene in its own words.

Neither is a measurement, and neither is in the dataset - it carries no weights at all. They
narrow the assumptions of `domain.cargo_weight` from one number for every photo to one per
scene, and they are replaceable: a caller who knows the shipment passes its own.
"""

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from loadvision.domain.cargo_weight import MAX_STACK_SHARE, CargoKind
from loadvision.imaging.prepared_image import PreparedImage

# Apache-2.0, 500M parameters, about 260 MB: answers on a CPU in seconds, which is what a
# button that estimates weight can afford.
SIGHT_MODEL: Final = "HuggingFaceTB/SmolVLM-500M-Instruct"

# The question and the answers it may choose from. A closed list is the point: asked freely,
# the model calls half a body of boxes "nothing".
_CHOICES: Final[dict[str, CargoKind]] = {
    "empty": CargoKind.EMPTY,
    "mail sacks": CargoKind.SACKS,
    "cardboard parcels": CargoKind.PARCELS,
    "pallets or roll cages": CargoKind.PALLETS,
    "several kinds at once": CargoKind.MIXED,
}
_QUESTION: Final = (
    "What is loaded in this truck body? Choose exactly one line from the list and repeat it "
    "with no other words:\n" + "\n".join(_CHOICES)
)
_ANSWER_TOKENS: Final = 12

# A cell of the depth grid counts as cargo when it is this much nearer than the emptiest part
# of the frame, which is the far wall or the roof.
_NEAR_MARGIN: Final = 0.12
# The lowest rows are the floor right under the camera; they are near in every photo.
_FLOOR_ROWS: Final = 2


@dataclass(frozen=True, slots=True)
class CargoSeen:
    """What the photo says about the cargo itself.

    Attributes:
        kind: Which of the known kinds of cargo it is.
        stack_share: Share of the body's height the cargo reaches, 0 to `MAX_STACK_SHARE`.
    """

    kind: CargoKind
    stack_share: float


def stack_share(depth_grid: NDArray[np.float32], rows: int, cols: int) -> float:
    """Estimates how high the cargo is stacked, from the depth grid of the photo.

    The grid holds relative depth, larger meaning nearer. The emptiest cells - the far wall
    and the roof - set the baseline; a cell noticeably nearer than that holds something. The
    answer is how far up those cells reach, as a share of the frame's height.

    Args:
        depth_grid: Flat grid of relative depth, values 0 to 1.
        rows: Rows of that grid, from the top of the frame down.
        cols: Columns of it.

    Returns:
        Share of the body's height the cargo takes, never above `MAX_STACK_SHARE`.
    """
    grid = np.asarray(depth_grid, dtype=np.float64).reshape(rows, cols)
    middle = grid[:, cols // 4 : cols - cols // 4]  # the walls run along the sides
    floor = middle[rows - _FLOOR_ROWS :].mean()
    far = float(np.percentile(middle, 10))
    if floor <= far:
        return 0.0
    # A row counts when its middle is nearer than the far wall by a clear margin.
    nearer = middle.mean(axis=1) > far + _NEAR_MARGIN * (floor - far)
    usable = nearer[: rows - _FLOOR_ROWS]
    if not usable.any():
        return 0.0
    highest = int(np.argmax(usable))  # rows run from the top, so the first one is the tallest
    return min(MAX_STACK_SHARE, (rows - highest) / rows)


class CargoSight:
    """Names the kind of cargo on a photo, by asking a small model to pick from a list.

    The model is loaded once and answers on the CPU in seconds: this runs behind an operator's
    button, not on the path of an assessment.

    Args:
        model: Loaded vision-language model.
        processor: Its processor.
    """

    def __init__(self, model: Any, processor: Any) -> None:
        self._model = model
        self._processor = processor

    @classmethod
    def load(cls, source: str = SIGHT_MODEL, device: str = "cpu") -> "CargoSight":
        """Loads the model from the hub or from a folder a package carries."""
        # Imported here: a service that never estimates weight needs none of this.
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor: Any = AutoProcessor.from_pretrained(source)
        model: Any = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.float32)
        return cls(model.to(device).eval(), processor)

    def kind(self, image: PreparedImage) -> CargoKind:
        """Returns which kind of cargo the photo shows.

        Args:
            image: Decoded photo.

        Returns:
            The kind; `CargoKind.MIXED` when the answer matches none of the choices, because
            the mixed density is the widest and the least wrong.
        """
        import torch
        from PIL import Image

        picture = Image.fromarray(np.asarray(image.pixels)).convert("RGB")
        picture.thumbnail((512, 512))
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _QUESTION}]}
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[picture], return_tensors="pt")
        with torch.inference_mode():
            produced = self._model.generate(
                **inputs, max_new_tokens=_ANSWER_TOKENS, do_sample=False
            )
        answer = self._processor.batch_decode(
            produced[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return _match(answer)


def _match(answer: str) -> CargoKind:
    """Maps a free-text answer onto the closed list; anything unknown counts as mixed."""
    lowered = answer.strip().lower()
    for choice, kind in _CHOICES.items():
        if choice in lowered:
            return kind
    # The model sometimes answers with one word of the line it meant.
    for choice, kind in _CHOICES.items():
        if any(word in lowered for word in choice.split() if len(word) > 3):
            return kind
    return CargoKind.MIXED
