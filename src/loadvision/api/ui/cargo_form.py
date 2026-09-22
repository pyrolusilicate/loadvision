"""What the weight block under a photo asks, and what it shows once it has been answered.

A photo says how much of the floor is taken, and nothing else a weight needs. The size of the
body cannot be read off it - there is nothing in the frame of a known size to measure against -
and neither can what a cubic metre of this cargo weighs. Both are chosen by the operator, the
answer is an interval, and the page says out loud that it is an estimate by a typical density
rather than a weighing.

The block posts and the answer is stored, because the table of densities is ours to change and
a number an operator acted on must not change under him afterwards.
"""

from dataclasses import dataclass
from typing import Final

from loadvision.domain.cargo_weight import (
    BODIES,
    BodySize,
    CargoDensity,
    CargoKind,
    WeightEstimate,
    density_of,
)
from loadvision.domain.weighing import Weighing

# The body an operator picks when the list has nothing like it; then the three sizes are typed.
CUSTOM_BODY: Final = "custom"
# How the bodies of `BODIES` are named on the page, in the order they are shown.
BODY_NAMES: Final = {
    "t5": "5 тонн",
    "t10": "10 тонн",
    "t20": "20 тонн",
    CUSTOM_BODY: "Иной габарит",
}
# How the kinds of cargo are named on the page. EMPTY is not among them on purpose: how empty
# the body is, is what the load percent already says, and picking "empty" on a photo of a half
# loaded body would answer with the weight of a cargo nobody claimed was there.
KIND_NAMES: Final = {
    CargoKind.ROLL_CAGES: "Роллтейнеры",
    CargoKind.PLASTIC_BOXES: "Пластиковые контейнеры",
    CargoKind.SACKS: "Мешки",
    CargoKind.PALLETS: "Паллеты",
    CargoKind.MIXED: "Смешанный",
}
# Sizes an operator may type, in metres. Wide enough for any lorry, tight enough that a typo
# of a hundred metres does not turn into a hundred tonnes.
_MIN_SIDE: Final = 0.5
_MAX_SIDE: Final = 25.0

_FIRST_BODY: Final = next(iter(BODIES))


@dataclass(frozen=True, slots=True)
class Chosen:
    """What an operator picked in the form.

    Attributes:
        body: Key of the body, or `CUSTOM_BODY`.
        size: Inner size of it, whether from the list or typed in.
        kind: Kind of cargo, or None when nothing usable was chosen.
    """

    body: str
    size: BodySize
    kind: CargoKind | None


@dataclass(frozen=True, slots=True)
class WeightBlock:
    """What the page needs to draw the block, with the stored answer when there is one.

    Attributes:
        body: Key of the body to show selected.
        size: Sizes to show in the three fields.
        kind: Kind of cargo to show selected.
        density: What a cubic metre of it is taken to weigh; the page shows the numbers, so
            nobody has to take the answer on trust.
        estimate: The stored answer, or None while nobody has weighed this assessment.
    """

    body: str
    size: BodySize
    kind: CargoKind
    density: CargoDensity
    estimate: WeightEstimate | None


def chosen_from(body: str, kind: str, length: str, width: str, height: str) -> Chosen:
    """Reads the form of the block, falling back to the defaults on anything unusable.

    A page is not an API: a field left empty or filled in by hand must come back with the block
    drawn, not with the whole page refused. What decides whether anything is weighed at all is
    `kind`, and it is None until the operator has chosen one of `KIND_NAMES`.

    Args:
        body: Key from `BODIES`, or `CUSTOM_BODY`.
        kind: Value of `CargoKind` from the list on the page.
        length: Length in metres, used only with `CUSTOM_BODY`.
        width: Width in metres, likewise.
        height: Height in metres, likewise.

    Returns:
        What was chosen, with `kind` None when nothing was.
    """
    picked = body if body in BODY_NAMES else _FIRST_BODY
    size = _typed(length, width, height) if picked == CUSTOM_BODY else BODIES[picked]
    if size is None:
        picked, size = _FIRST_BODY, BODIES[_FIRST_BODY]
    known = {value.value for value in KIND_NAMES}
    return Chosen(body=picked, size=size, kind=CargoKind(kind) if kind in known else None)


def stored_block(weighing: Weighing | None) -> WeightBlock:
    """What the form shows: the stored weighing, or the defaults when nobody has weighed yet."""
    if weighing is None:
        return WeightBlock(
            body=_FIRST_BODY,
            size=BODIES[_FIRST_BODY],
            kind=CargoKind.MIXED,
            density=density_of(CargoKind.MIXED),
            estimate=None,
        )
    return WeightBlock(
        body=weighing.body,
        size=weighing.size,
        kind=weighing.kind,
        density=density_of(weighing.kind),
        estimate=weighing.estimate,
    )


def _typed(length: str, width: str, height: str) -> BodySize | None:
    """Reads three sides an operator typed, or None when any of them is not a usable number."""
    sides = []
    for value in (length, width, height):
        try:
            side = float(value.replace(",", "."))
        except ValueError:
            return None
        if not _MIN_SIDE <= side <= _MAX_SIDE:
            return None
        sides.append(side)
    return BodySize(length=sides[0], width=sides[1], height=sides[2])
