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
