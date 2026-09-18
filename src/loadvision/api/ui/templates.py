"""Page templates and the filters they format values with."""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Final

from starlette.templating import Jinja2Templates

from loadvision.api.xml.schemas import LOAD_PCT_PLACES, format_decimal
from loadvision.domain.timestamps import format_utc_seconds

UI_PREFIX: Final = "/ui"

_HERE: Final = Path(__file__).parent
_STYLESHEET: Final = _HERE / "static" / "panel.css"
_VERSION_CHARS: Final = 12

templates: Final = Jinja2Templates(directory=_HERE / "templates")


def _stylesheet() -> str:
    # The content hash in the address makes a browser fetch the new stylesheet after a
    # release. Not the file time: files installed from a wheel all carry the same date.
    version = hashlib.sha256(_STYLESHEET.read_bytes()).hexdigest()[:_VERSION_CHARS]
    return f"{UI_PREFIX}/static/panel.css?v={version}"


def _load(value: float) -> str:
    # The same rounding as the XML answers, so a page and the API never disagree.
    return format_decimal(value, LOAD_PCT_PLACES)


def human_time(value: datetime) -> str:
    """Writes a time for a person: UTC to the second, with the zone spelled out."""
    return format_utc_seconds(value).replace("T", " ").replace("Z", " UTC")


templates.env.globals["ui"] = UI_PREFIX
templates.env.globals["stylesheet"] = _stylesheet
templates.env.filters["load"] = _load
templates.env.filters["time"] = human_time
