import numpy as np
import pytest

from loadvision.ml.evaluation.blend import blend, subsets
from loadvision.ml.pipeline.stages.train_folds import OofRow


def rows(raws: list[float], folds: list[int] | None = None) -> dict[str, OofRow]:
    folds = folds or [index % 2 for index in range(len(raws))]
    return {
        f"img_{index}": OofRow(fold=fold, label=50.0, raw=raw, predicted=raw)
        for index, (raw, fold) in enumerate(zip(raws, folds, strict=True))
    }


@pytest.mark.ml
def test_members_are_averaged_photo_by_photo() -> None:
    result = blend({"a": rows([10.0, 20.0, 30.0]), "b": rows([30.0, 40.0, 50.0])})

    np.testing.assert_allclose(result.raw, [20.0, 30.0, 40.0])
    assert result.ids == ("img_0", "img_1", "img_2")


@pytest.mark.ml
def test_runs_over_other_photos_are_refused() -> None:
    with pytest.raises(ValueError, match="other photos"):
        blend({"a": rows([10.0, 20.0]), "b": rows([10.0, 20.0, 30.0])})


@pytest.mark.ml
def test_runs_on_other_folds_are_refused() -> None:
    # The calibration is fitted fold by fold: a photo in two folds would leak into its own curve.
    with pytest.raises(ValueError, match="another fold"):
        blend({"a": rows([10.0, 20.0], [0, 1]), "b": rows([10.0, 20.0], [1, 0])})


@pytest.mark.ml
def test_every_combination_is_listed_once() -> None:
    assert subsets(["a", "b", "c"]) == [
        ("a",),
        ("b",),
        ("c",),
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "b", "c"),
    ]


@pytest.mark.unit
def test_the_median_join_ignores_a_single_wild_member() -> None:
    runs = {
        "sane_a": rows([40.0, 60.0]),
        "sane_b": rows([42.0, 62.0]),
        "wild": rows([100.0, 0.0]),
    }

    averaged = blend(runs, "mean")
    middled = blend(runs, "median")

    assert averaged.raw.tolist() == pytest.approx([60.666667, 40.666667], abs=1e-4)
    assert middled.raw.tolist() == pytest.approx([42.0, 60.0])
    assert middled.how == "median"
    assert averaged.how == "mean"


@pytest.mark.unit
def test_two_members_have_the_same_mean_and_median() -> None:
    runs = {"one": rows([40.0, 10.0]), "two": rows([50.0, 20.0])}

    assert blend(runs, "median").raw.tolist() == pytest.approx(blend(runs, "mean").raw.tolist())


@pytest.mark.unit
def test_the_default_join_is_still_the_mean_every_earlier_number_was_measured_with() -> None:
    runs = {"a": rows([40.0]), "b": rows([44.0]), "c": rows([90.0])}

    assert blend(runs).raw.tolist() == pytest.approx([58.0])


@pytest.mark.unit
def test_an_unknown_join_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown join"):
        blend({"one": rows([40.0])}, "trimmed")  # type: ignore[arg-type]
