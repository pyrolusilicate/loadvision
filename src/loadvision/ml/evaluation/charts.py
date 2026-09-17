"""Inline SVG charts for research reports: no libraries, no files to serve."""

import html
from typing import Final

_WIDTH: Final = 720
_HEIGHT: Final = 230
_PAD_LEFT: Final = 48
_PAD_BOTTOM: Final = 36
_PAD_TOP: Final = 12
_MAX_LABELS: Final = 24
_SHOWN_LABELS: Final = 12


def bars(
    items: list[tuple[str, float]],
    *,
    unit: str = "",
    decimals: int = 2,
    zero_line: bool = False,
) -> str:
    """One-series bar chart as inline SVG, with a native tooltip on every bar.

    Args:
        items: Label and value of every bar, in the order they are drawn.
        unit: Suffix of the value in the tooltip.
        decimals: Digits after the point on the axis and in tooltips.
        zero_line: Draw bars from zero, so negative values hang below it.

    Returns:
        The `<svg>` element as text.
    """
    values = [value for _, value in items]
    low = min(0.0, min(values)) if zero_line else 0.0
    high = max(values) or 1.0
    span = (high - low) or 1.0
    plot_w, plot_h = _WIDTH - _PAD_LEFT - 8, _HEIGHT - _PAD_BOTTOM - _PAD_TOP
    slot = plot_w / len(items)
    bar_w = max(2.0, slot * 0.72)

    def y_of(value: float) -> float:
        return _PAD_TOP + plot_h * (1 - (value - low) / span)

    out = [f'<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" role="img">']
    for fraction in (0.0, 0.5, 1.0):
        value = low + span * fraction
        y = y_of(value)
        out.append(
            f'<line class="grid" x1="{_PAD_LEFT}" y1="{y:.1f}" x2="{_WIDTH - 8}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{_PAD_LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">'
            f"{value:.{decimals}f}</text>"
        )
    base = y_of(0.0) if zero_line else y_of(low)
    for index, (label, value) in enumerate(items):
        x = _PAD_LEFT + slot * index + (slot - bar_w) / 2
        top = min(y_of(value), base)
        bar_h = max(abs(y_of(value) - base), 1.0)
        out.append(
            f'<rect class="bar" x="{x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{bar_h:.1f}" rx="3"><title>{html.escape(label)}: '
            f"{value:.{decimals}f}{unit}</title></rect>"
        )
        if len(items) <= _MAX_LABELS or index % max(1, len(items) // _SHOWN_LABELS) == 0:
            out.append(
                f'<text class="tick" x="{x + bar_w / 2:.1f}" y="{_HEIGHT - _PAD_BOTTOM + 16}" '
                f'text-anchor="middle">{html.escape(label)}</text>'
            )
    out.append("</svg>")
    return "\n".join(out)


def card(title: str, subtitle: str, body: str, note: str = "") -> str:
    """Wraps a chart in a titled card with an optional note under it."""
    tail = f'<p class="note">{note}</p>' if note else ""
    return (
        f'<section class="card"><h3>{title}</h3><p class="sub">{subtitle}</p>{body}{tail}</section>'
    )


def stat(value: str, label: str, hint: str = "") -> str:
    """One big number with its name and a hint."""
    return (
        f'<div class="stat"><div class="v">{value}</div><div class="l">{label}</div>'
        f'<div class="h">{hint}</div></div>'
    )
