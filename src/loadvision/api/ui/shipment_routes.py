"""Looking up a shipment by its number."""

from typing import Annotated, Final
from urllib.parse import quote

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from loadvision.api.ui.pages import catalog_error, page
from loadvision.api.ui.pagination import ROWS_PER_PAGE, pager
from loadvision.api.ui.services import ui_services_of
from loadvision.api.ui.session import SignedIn
from loadvision.api.ui.templates import UI_PREFIX
from loadvision.application.paging import slice_page
from loadvision.errors import ErrorCode, LoadvisionError

router = APIRouter()

# Answers to a typed number that belong on the search page, next to the number to fix.
_SEARCH_ERRORS: Final = frozenset({ErrorCode.SHIPMENT_NOT_FOUND, ErrorCode.INVALID_SHIPMENT_ID})


@router.get("/shipments")
def search(request: Request, session: SignedIn, number: str = "") -> Response:
    """Shows the search form, or opens the shipment whose number was typed."""
    typed = number.strip()
    if typed:
        # The number goes into a path: "/" or "?" typed by mistake must not change the address.
        return RedirectResponse(f"{UI_PREFIX}/shipments/{quote(typed, safe='')}", status_code=303)
    return page(request, "shipments.html", session, history=None, number="")


@router.get("/shipments/{shipment_id}")
async def shipment(
    request: Request,
    session: SignedIn,
    shipment_id: str,
    number: Annotated[int, Query(alias="page")] = 1,
) -> Response:
    """Shows every assessment of the shipment, newest first.

    An unknown or malformed number stays on the search page with the reason; an unknown one
    also gets a link to assess the first photo of that shipment.
    """
    try:
        history = await run_in_threadpool(ui_services_of(request).get_shipment, shipment_id)
    except LoadvisionError as error:
        if error.code not in _SEARCH_ERRORS:
            raise
        status, message, hint = catalog_error(error.code)
        return page(
            request,
            "shipments.html",
            session,
            status_code=status,
            history=None,
            number=shipment_id,
            error=(message, hint),
            unknown=error.code is ErrorCode.SHIPMENT_NOT_FOUND,
        )
    # A shipment holds a handful of photos, so its history is paged in memory.
    rows = slice_page(history.assessments, number, ROWS_PER_PAGE)
    return page(
        request,
        "shipments.html",
        session,
        history=history,
        number=history.shipment_id.value,
        rows=rows,
        pager=pager(rows, f"{UI_PREFIX}/shipments/{quote(history.shipment_id.value, safe='')}"),
    )
