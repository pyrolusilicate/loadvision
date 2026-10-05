"""Scenarios a run is judged by, on top of the average metric.

An average hides the cases a reviewer cares about: a nearly empty body with two boxes in the
depth, a body packed loosely, a night shot. Each of those is a slice here, and every run
reports its numbers separately, so an improvement that helps on average but breaks a scenario
is visible instead of hidden.

A slice is defined only by what is known before a prediction: the reference label, the size of
the photo, its proportions, its brightness, the size of its scene. Defining one by the error
would make measuring it circular.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

DARK_BRIGHTNESS: Final = 60.0
USUAL_ASPECT: Final = (0.70, 0.80)


@dataclass(frozen=True, slots=True, kw_only=True)
class Photo:
    """What a slice may look at, all of it known before the model runs.

    Attributes:
        label: Reference load.
        brightness: Mean brightness of the photo, 0 to 255.
        aspect: Width divided by height.
        scene_size: How many photos the scene of this photo holds.
    """

    label: float
    brightness: float
    aspect: float
    scene_size: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Slice:
    """One scenario.

    Attributes:
        name: Short key used in metrics and in the report.
        description: What the scenario is, in Russian, for the report and the slides.
        belongs: Whether a photo falls into this scenario.
    """

    name: str
    description: str
    belongs: Callable[[Photo], bool]


SLICES: Final[tuple[Slice, ...]] = (
    Slice(
        name="empty",
        description="Почти пустой кузов, метка не выше 5. Несколько коробок в глубине модель "
        "читает как «груз дошёл до середины»",
        belongs=lambda photo: photo.label <= 5,
    ),
    Slice(
        name="low",
        description="Слабо загружен, метка от 5 до 20",
        belongs=lambda photo: 5 < photo.label <= 20,
    ),
    Slice(
        name="middle",
        description="Середина, метка от 20 до 80. Самый тяжёлый диапазон",
        belongs=lambda photo: 20 < photo.label < 80,
    ),
    Slice(
        name="high",
        description="Сильно загружен, метка от 80 до 95",
        belongs=lambda photo: 80 <= photo.label < 95,
    ),
    Slice(
        name="full",
        description="Забит под завязку, метка от 95. Неплотная укладка читается моделью как 80-90",
        belongs=lambda photo: photo.label >= 95,
    ),
    Slice(
        name="dark",
        description=f"Тёмный кадр, средняя яркость ниже {DARK_BRIGHTNESS:.0f} из 255",
        belongs=lambda photo: photo.brightness < DARK_BRIGHTNESS,
    ),
    Slice(
        name="bright",
        description=f"Обычная освещённость, яркость не ниже {DARK_BRIGHTNESS:.0f}",
        belongs=lambda photo: photo.brightness >= DARK_BRIGHTNESS,
    ),
    Slice(
        name="odd_aspect",
        description=f"Нестандартные пропорции, вне {USUAL_ASPECT[0]:.2f}-{USUAL_ASPECT[1]:.2f}",
        belongs=lambda photo: not USUAL_ASPECT[0] <= photo.aspect <= USUAL_ASPECT[1],
    ),
    Slice(
        name="multi_scene",
        description="Сцена из нескольких фото: один кузов снят несколько раз",
        belongs=lambda photo: photo.scene_size > 1,
    ),
)


def names() -> tuple[str, ...]:
    """Names of every scenario, in catalogue order."""
    return tuple(item.name for item in SLICES)


def assign(photos: Sequence[Photo]) -> Mapping[str, list[int]]:
    """Groups photos by scenario.

    A photo may belong to several scenarios, and that is on purpose: a dark nearly empty body
    should show up in both.

    Args:
        photos: What is known about every photo, in the order of the predictions.

    Returns:
        Scenario name to the positions of its photos; empty scenarios are left out.
    """
    groups: dict[str, list[int]] = {}
    for index, photo in enumerate(photos):
        for item in SLICES:
            if item.belongs(photo):
                groups.setdefault(item.name, []).append(index)
    return groups


def describe(name: str) -> str:
    """Prose description of a scenario, for the report.

    Args:
        name: Scenario name.

    Returns:
        The description.

    Raises:
        KeyError: If there is no such scenario.
    """
    for item in SLICES:
        if item.name == name:
            return item.description
    raise KeyError(name)
