"""Page templates and the filters they format values with."""

import hashlib
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final

from starlette.templating import Jinja2Templates

from loadvision.api.xml.schemas import LOAD_PCT_PLACES, format_decimal
from loadvision.application.ports import ExplanationKind
from loadvision.domain.assessment import ReviewReason
from loadvision.domain.timestamps import format_utc_seconds

UI_PREFIX: Final = "/ui"
# Where an operator lands: assessing a photo is the first thing the service is for.
HOME_PAGE: Final = f"{UI_PREFIX}/assess"

_HERE: Final = Path(__file__).parent
_STYLESHEET: Final = _HERE / "static" / "panel.css"
_VERSION_CHARS: Final = 12

# What an assessment card says about each reason to check the prediction.
REVIEW_NOTES: Final = MappingProxyType(
    {
        ReviewReason.UNUSUAL_SCENE: (
            "Фото не похоже на снимки кузовов, на которых училась модель: закрыт объектив, "
            "засвет, в кадре не кузов или непривычный ракурс."
        ),
        ReviewReason.LOW_CONFIDENCE: "Модели ансамбля сильно расходятся в оценке.",
        ReviewReason.TOO_DARK: "В кадре слишком темно, груз не разглядеть.",
        ReviewReason.BLURRY: "Фото размыто.",
    }
)

# What a card can show in place of the photo, in the order of the switch above it.
PHOTO_VIEW: Final = ""
VIEW_NAMES: Final = MappingProxyType(
    {
        PHOTO_VIEW: "Фото",
        ExplanationKind.ATTENTION.value: "Внимание",
        ExplanationKind.CONTRIBUTION.value: "Вклад в ответ",
    }
)
VIEW_NOTES: Final = MappingProxyType(
    {
        ExplanationKind.ATTENTION.value: (
            "видно, куда модель смотрит. Поднимает место процент или опускает, карта не говорит."
        ),
        ExplanationKind.CONTRIBUTION.value: (
            "видно, какие места тянут процент вверх, а какие вниз."
        ),
    }
)

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
templates.env.globals["review_notes"] = REVIEW_NOTES
templates.env.globals["view_names"] = VIEW_NAMES
templates.env.globals["view_notes"] = VIEW_NOTES
templates.env.globals["stylesheet"] = _stylesheet
templates.env.filters["load"] = _load
templates.env.filters["time"] = human_time
