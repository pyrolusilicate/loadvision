"""Looking up a shipment by its number."""

from urllib.parse import quote

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from loadvision.api.ui.pages import page
from loadvision.api.ui.services import ui_services_of
from loadvision.api.ui.session import SignedIn
from loadvision.api.ui.templates import UI_PREFIX

router = APIRouter()


@router.get("/shipments")
def search(request: Request, session: SignedIn, number: str = "") -> Response:
    """Shows the search form, or opens the shipment whose number was typed."""
    typed = number.strip()
    if typed:
        # The number goes into a path: "/" or "?" typed by mistake must not change the address.
        return RedirectResponse(f"{UI_PREFIX}/shipments/{quote(typed, safe='')}", status_code=303)
    return page(request, "shipments.html", session, history=None)


@router.get("/shipments/{shipment_id}")
async def shipment(request: Request, session: SignedIn, shipment_id: str) -> Response:
    """Shows every assessment of the shipment, newest first."""
    history = await run_in_threadpool(ui_services_of(request).get_shipment, shipment_id)
    return page(request, "shipments.html", session, history=history)
