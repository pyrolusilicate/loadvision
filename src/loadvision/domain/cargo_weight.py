"""Weight of the cargo, estimated from the load percent and the size of the body.

There is no weight in the dataset, so nothing here is learned: this is arithmetic over the
load the model predicts. A photo shows how much floor is taken, not what stands on it, so the
answer is an interval around a density that the caller chooses, and it is an estimate, never a
measurement. Everything that cannot be seen on a photo - the body, how high the cargo is
stacked, what it weighs per cubic metre - is an argument with a default, so a caller who knows
the shipment better replaces it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Self

_PERCENT: Final = 100.0

# Cargo rarely reaches the ceiling: the share of the body's height a stack usually takes.
DEFAULT_STACK_SHARE: Final = 0.6
MAX_STACK_SHARE = 0.90


class CargoKind(StrEnum):
    """What is in the body, as far as a photo tells them apart.

    The kinds are the ones our own photos actually show, counted by hand over 128 of them: a
    mesh roll cage on wheels is the unit of a postal run, a solid plastic box is the heavier
    one beside it, and pallets of shrink-wrapped goods are rarer and denser still. MIXED is
    the honest answer when the photo shows several at once, and it is the commonest of all.
    """

    EMPTY = "empty"
    ROLL_CAGES = "roll_cages"
    PLASTIC_BOXES = "plastic_boxes"
    SACKS = "sacks"
    PALLETS = "pallets"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True, kw_only=True)
class BodySize:
    """Inner size of the cargo body, in metres.

    Attributes:
        length: Along the body, from the doors to the far wall.
        width: Across the body.
        height: Floor to ceiling.
    """

    length: float
    width: float
    height: float

    def __post_init__(self) -> None:
        """Checks the sides.

        Raises:
            ValueError: If a side is not a positive finite number of metres.
        """
        for name in ("length", "width", "height"):
            side = getattr(self, name)
            if not 0 < side < 100:
                raise ValueError(f"{name} must be a size in metres, got {side}")

    @property
    def volume(self) -> float:
        """Volume of the empty body, in cubic metres."""
        return self.length * self.width * self.height


@dataclass(frozen=True, slots=True, kw_only=True)
class CargoDensity:
    """What a cubic metre of this cargo weighs, in kilograms.

    Attributes:
        low: Lightest the cargo plausibly is; gives the lower bound of the estimate.
        typical: What it usually is; gives the number itself.
        high: Heaviest it plausibly is; gives the upper bound.
    """

    low: float
    typical: float
    high: float

    def __post_init__(self) -> None:
        """Checks the three numbers.

        Raises:
            ValueError: If a density is not positive or they are out of order.
        """
        if not 0 < self.low <= self.typical <= self.high:
            raise ValueError(
                f"densities must be positive and ordered low <= typical <= high, got "
                f"{self.low}, {self.typical}, {self.high}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightEstimate:
    """What the cargo on a photo weighs, as far as a photo can tell.

    Attributes:
        kilograms: The estimate itself, at the typical density.
        low: Lower bound, at the lightest density.
        high: Upper bound, at the heaviest density.
        cubic_metres: Volume the cargo is taken to occupy; the estimate divided by density.
    """

    kilograms: float
    low: float
    high: float
    cubic_metres: float

    @classmethod
    def of(cls, cubic_metres: float, density: CargoDensity) -> Self:
        """Weighs a volume at the three densities."""
        return cls(
            kilograms=round(cubic_metres * density.typical, 1),
            low=round(cubic_metres * density.low, 1),
            high=round(cubic_metres * density.high, 1),
            cubic_metres=round(cubic_metres, 2),
        )


# Kilograms per cubic metre by kind of cargo. These are assumptions, wide on purpose, not
# measurements: the dataset carries no weights at all. Two public numbers bound them. Freight
# under 160 kg/m3 counts as light and over 480 as very dense, and postal cargo sits at the
# light end. A nestable postal roll cage weighs about 100 kg empty and holds roughly a cubic
# metre, so its own steel already sets the floor for a cage that carries parcels.
#
# One weighbridge record per photo is enough to replace the lot with the client's own numbers.
DENSITIES: Final[Mapping[CargoKind, "CargoDensity"]] = {
    CargoKind.ROLL_CAGES: CargoDensity(low=150.0, typical=220.0, high=320.0),
    CargoKind.PLASTIC_BOXES: CargoDensity(low=200.0, typical=290.0, high=420.0),
    CargoKind.SACKS: CargoDensity(low=150.0, typical=250.0, high=380.0),
    CargoKind.PALLETS: CargoDensity(low=200.0, typical=310.0, high=480.0),
    CargoKind.MIXED: CargoDensity(low=120.0, typical=230.0, high=400.0),
}

# The bodies an operator picks from, inner size in metres. A photo cannot give the size of the
# body - there is nothing in the frame of a known size to measure against - so it is chosen,
# not computed, and a body outside the list is typed in by hand.
# The three the task names: "the test set holds lorries rated 5, 10 and 20 tonnes among
# others". The sizes are the usual inner ones for each class, the twenty tonner being a
# semi-trailer.
BODIES: Final[Mapping[str, "BodySize"]] = {
    "t5": BodySize(length=6.2, width=2.45, height=2.4),
    "t10": BodySize(length=8.0, width=2.45, height=2.6),
    "t20": BodySize(length=13.6, width=2.45, height=2.7),
}

# The smallest of the three, so a shipment nobody described is not overstated.
DEFAULT_BODY: Final = BODIES["t5"]

# Parcels and sacks of a postal run, kilograms per cubic metre. Mail is light for its volume,
# and the spread is wide on purpose: that is why the answer is an interval.
DEFAULT_DENSITY: Final = DENSITIES[CargoKind.MIXED]


def density_of(kind: CargoKind) -> CargoDensity:
    """Returns the density to weigh this kind of cargo with.

    Args:
        kind: What the photo shows.

    Returns:
        Its density; an empty body weighs nothing, so it takes the mixed one and a load of
        zero does the rest.
    """
    return DENSITIES.get(kind, DEFAULT_DENSITY)


def estimate_weight(
    load_pct: float,
    *,
    body: BodySize = DEFAULT_BODY,
    density: CargoDensity = DEFAULT_DENSITY,
    stack_share: float = DEFAULT_STACK_SHARE,
) -> WeightEstimate:
    """Estimates what the cargo weighs, from the share of the floor it takes.

    The load percent says how much of the floor is occupied. Multiplied by the floor area it
    gives an area, by the height of a stack a volume, and by a density a weight. Every step
    after the first is an assumption of the caller, which is why the answer carries an
    interval and the volume it was computed from.

    Args:
        load_pct: Load of the body, 0 to 100, as the model answers it.
        body: Inner size of that body; a common delivery truck by default.
        density: What a cubic metre of this cargo weighs; postal parcels by default.
        stack_share: Share of the body's height the cargo is stacked to, 0 to 1.

    Returns:
        The estimate, in kilograms, with its bounds and the volume behind it.

    Raises:
        ValueError: If the load is outside 0-100 or the stack share outside 0-1.
    """
    if not 0 <= load_pct <= _PERCENT:
        raise ValueError(f"load_pct must be 0 to 100, got {load_pct}")
    if not 0 < stack_share <= 1:
        raise ValueError(f"stack_share must be above 0 and at most 1, got {stack_share}")
    occupied = body.volume * (load_pct / _PERCENT) * stack_share
    return WeightEstimate.of(occupied, density)
