"""The pager under a long list: where the list stands and the addresses of its neighbours."""

from dataclasses import dataclass
from typing import Final
from urllib.parse import urlencode

from loadvision.application.paging import Page

# Photos come as a grid of tiles, four to a row on a wide screen; tables come as rows.
TILES_PER_PAGE: Final = 24
ROWS_PER_PAGE: Final = 20


@dataclass(frozen=True, slots=True, kw_only=True)
class Pager:
    """What the pager shows.

    Attributes:
        number: Current page, from 1.
        pages: Pages in the list.
        total: Rows in the list.
        previous: Address of the page before, or None on the first.
        next: Address of the page after, or None on the last.
    """

    number: int
    pages: int
    total: int
    previous: str | None
    next: str | None


def pager(page: Page[object], path: str, **params: str) -> Pager:
    """Builds the pager of a page.

    Args:
        page: The page shown.
        path: Address of the list without a query.
        **params: Other query values the neighbours keep, such as the order.
    """

    def address(number: int) -> str:
        return f"{path}?{urlencode({**params, 'page': number})}"

    return Pager(
        number=page.number,
        pages=page.pages,
        total=page.total,
        previous=address(page.number - 1) if page.number > 1 else None,
        next=address(page.number + 1) if page.number < page.pages else None,
    )
