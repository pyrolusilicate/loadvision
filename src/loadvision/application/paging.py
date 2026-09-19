"""One page of a long list, for pages that show a list in parts."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class Page[T]:
    """The rows of one page and where the page stands in the list.

    Attributes:
        items: Rows of this page.
        number: Page number, from 1.
        size: Most rows a page holds.
        total: Rows in the whole list.
    """

    items: tuple[T, ...]
    number: int
    size: int
    total: int

    @property
    def pages(self) -> int:
        """How many pages the list takes; an empty list still shows one page."""
        return page_count(self.total, self.size)

    @property
    def offset(self) -> int:
        """Rows before this page."""
        return (self.number - 1) * self.size


def page_count(total: int, size: int) -> int:
    """How many pages of `size` rows `total` rows take, at least one."""
    return max(1, -(-total // size))


def page_number(requested: int, total: int, size: int) -> int:
    """Returns the page to show: the requested one, within the pages the list has.

    A number past the end, such as an old link after rows were labelled, shows the last page.
    """
    return min(max(requested, 1), page_count(total, size))


def slice_page[T](rows: Sequence[T], requested: int, size: int) -> Page[T]:
    """Cuts one page out of a list that is already in memory."""
    number = page_number(requested, len(rows), size)
    start = (number - 1) * size
    return Page(items=tuple(rows[start : start + size]), number=number, size=size, total=len(rows))
