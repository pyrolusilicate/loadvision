import pytest

from loadvision.application.paging import page_count, page_number, slice_page


@pytest.mark.unit
@pytest.mark.parametrize(("total", "pages"), [(0, 1), (1, 1), (20, 1), (21, 2), (40, 2), (41, 3)])
def test_pages_cover_every_row_and_an_empty_list_has_one(total: int, pages: int) -> None:
    assert page_count(total, 20) == pages


@pytest.mark.unit
@pytest.mark.parametrize(("requested", "shown"), [(-3, 1), (0, 1), (1, 1), (2, 2), (3, 2), (99, 2)])
def test_a_requested_page_stays_within_the_list(requested: int, shown: int) -> None:
    assert page_number(requested, total=30, size=20) == shown


@pytest.mark.unit
def test_a_slice_holds_one_page_and_its_place() -> None:
    page = slice_page(list(range(45)), requested=3, size=20)

    assert page.items == tuple(range(40, 45))
    assert (page.number, page.pages, page.total, page.offset) == (3, 3, 45, 40)
