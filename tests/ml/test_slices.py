import pytest

from loadvision.ml.evaluation.slices import SLICES, Photo, assign, describe, names


def photo(
    label: float = 50.0, brightness: float = 120.0, aspect: float = 0.75, scene_size: int = 1
) -> Photo:
    return Photo(label=label, brightness=brightness, aspect=aspect, scene_size=scene_size)


@pytest.mark.ml
def test_every_scenario_has_a_description() -> None:
    assert len(names()) == len(SLICES)
    assert all(item.description for item in SLICES)
    assert len(set(names())) == len(names())


@pytest.mark.ml
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (0.0, "empty"),
        (5.0, "empty"),
        (10.0, "low"),
        (50.0, "middle"),
        (90.0, "high"),
        (100.0, "full"),
    ],
)
def test_load_scenarios_cover_the_whole_range(label: float, expected: str) -> None:
    groups = assign([photo(label=label)])

    assert expected in groups


@pytest.mark.ml
def test_every_photo_falls_into_exactly_one_load_scenario() -> None:
    load_names = {"empty", "low", "middle", "high", "full"}

    for label in range(0, 101):
        groups = assign([photo(label=float(label))])
        assert len(load_names & set(groups)) == 1, label


@pytest.mark.ml
def test_a_dark_photo_is_dark_and_not_bright() -> None:
    groups = assign([photo(brightness=40.0)])

    assert "dark" in groups
    assert "bright" not in groups


@pytest.mark.ml
def test_scenarios_may_overlap() -> None:
    # A dark nearly empty body belongs to both, and that is the point of slicing.
    groups = assign([photo(label=2.0, brightness=30.0)])

    assert {"empty", "dark"} <= set(groups)


@pytest.mark.ml
def test_unusual_proportions_are_flagged() -> None:
    assert "odd_aspect" in assign([photo(aspect=0.45)])
    assert "odd_aspect" not in assign([photo(aspect=0.75)])


@pytest.mark.ml
def test_a_scene_of_several_photos_is_flagged() -> None:
    assert "multi_scene" in assign([photo(scene_size=3)])
    assert "multi_scene" not in assign([photo(scene_size=1)])


@pytest.mark.ml
def test_positions_point_back_at_the_photos() -> None:
    photos = [photo(label=2.0), photo(label=50.0), photo(label=1.0)]

    groups = assign(photos)

    assert groups["empty"] == [0, 2]


@pytest.mark.ml
def test_an_unknown_scenario_has_no_description() -> None:
    with pytest.raises(KeyError):
        describe("no_such_case")
