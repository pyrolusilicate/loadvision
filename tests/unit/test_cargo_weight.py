import pytest

from loadvision.domain.cargo_weight import (
    DEFAULT_BODY,
    DEFAULT_DENSITY,
    BodySize,
    CargoDensity,
    estimate_weight,
)

# A body of exactly one cubic metre and a cargo of exactly 100 kg per cubic metre make the
# arithmetic readable: at half load and half height the cargo is 0.25 m3 and 25 kg.
UNIT_BODY = BodySize(length=1.0, width=1.0, height=1.0)
UNIT_DENSITY = CargoDensity(low=50.0, typical=100.0, high=200.0)


@pytest.mark.unit
def test_the_estimate_is_the_occupied_volume_times_the_density() -> None:
    estimate = estimate_weight(50, body=UNIT_BODY, density=UNIT_DENSITY, stack_share=0.5)

    assert estimate.cubic_metres == pytest.approx(0.25)
    assert estimate.kilograms == pytest.approx(25.0)
    assert (estimate.low, estimate.high) == (pytest.approx(12.5), pytest.approx(50.0))


@pytest.mark.unit
def test_an_empty_body_weighs_nothing_and_a_full_one_weighs_the_most() -> None:
    empty = estimate_weight(0, body=UNIT_BODY, density=UNIT_DENSITY)
    full = estimate_weight(100, body=UNIT_BODY, density=UNIT_DENSITY)

    assert empty.kilograms == 0.0
    assert full.kilograms > empty.kilograms
    # The stack does not reach the ceiling, so even a full body is not the whole volume.
    assert full.cubic_metres < UNIT_BODY.volume


@pytest.mark.unit
def test_more_load_never_weighs_less() -> None:
    weights = [estimate_weight(load).kilograms for load in range(0, 101, 5)]

    assert weights == sorted(weights)


@pytest.mark.unit
def test_a_bigger_body_holds_more_at_the_same_load() -> None:
    small = estimate_weight(40, body=BodySize(length=3.0, width=2.0, height=2.0))
    large = estimate_weight(40, body=BodySize(length=6.0, width=2.4, height=2.5))

    assert large.kilograms > 2 * small.kilograms


@pytest.mark.unit
def test_the_defaults_answer_a_plausible_lorry_load() -> None:
    estimate = estimate_weight(100, body=DEFAULT_BODY, density=DEFAULT_DENSITY)

    # A 16 m3 body of parcels: hundreds of kilograms, not tens and not tens of tonnes.
    assert 500 < estimate.kilograms < 3000
    assert estimate.low < estimate.kilograms < estimate.high


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"load_pct": -1}, "load_pct"),
        ({"load_pct": 101}, "load_pct"),
        ({"load_pct": 50, "stack_share": 0}, "stack_share"),
        ({"load_pct": 50, "stack_share": 1.5}, "stack_share"),
    ],
)
def test_impossible_arguments_are_refused(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_weight(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("side", ["length", "width", "height"])
def test_a_body_without_a_real_side_is_refused(side: str) -> None:
    sides = {"length": 4.0, "width": 2.0, "height": 2.0} | {side: 0.0}

    with pytest.raises(ValueError, match=side):
        BodySize(**sides)


@pytest.mark.unit
def test_densities_out_of_order_are_refused() -> None:
    with pytest.raises(ValueError, match="ordered"):
        CargoDensity(low=200.0, typical=100.0, high=300.0)
