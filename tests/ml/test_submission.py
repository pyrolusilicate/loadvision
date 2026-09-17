from pathlib import Path

import numpy as np
import pytest

from loadvision.ml.data.submission import COLUMNS, write_submission


@pytest.mark.ml
def test_answer_file_has_the_header_and_one_row_per_photo(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"

    write_submission(path, ["img_b", "img_a"], [12.34, 100.0])

    assert path.read_text(encoding="utf-8").splitlines() == [
        ",".join(COLUMNS),
        "img_b,12.3",
        "img_a,100.0",
    ]


@pytest.mark.ml
def test_numpy_values_are_written_like_plain_numbers(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "submission.csv"

    write_submission(path, ["img_a"], np.array([55.0]))

    assert path.read_text(encoding="utf-8").splitlines()[1] == "img_a,55.0"


@pytest.mark.ml
def test_a_load_without_a_photo_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_submission(tmp_path / "submission.csv", ["img_a"], [10.0, 20.0])
