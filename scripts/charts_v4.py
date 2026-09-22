"""Как отвечает Loadvision v4: попадание в метку, форма ошибки и потеря по диапазонам."""

import collections
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OOF = Path("runs/finetune/run_20260919_183257_877c1e/oof.csv")
OUT = Path("runs/charts/v4.png")

INK = "#1b2a33"
MUTED = "#5b6a73"
MAIN = "#3d7ea6"
WARM = "#c98b3a"
SOFT = "#d3dce2"
BAND = "#e8eef2"
GRID = "#eef2f4"

# Границы диапазонов те же, что в ml/evaluation/metrics.py: [низ, верх), последний включает 100.
RANGES = (
    ("пустые\n0-20%", 0, 20, False),
    ("наполовину\n20-80%", 20, 80, False),
    ("полные\n80-100%", 80, 100, True),
)


def dress(axes: plt.Axes, title: str) -> None:
    """Общее оформление панели: заголовок слева, тонкие оси, сетка по горизонтали."""
    axes.set_title(title, fontsize=13.5, color=INK, pad=12, loc="left")
    axes.tick_params(colors=MUTED, labelsize=11)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#c6d0d6")


def main() -> None:
    rows = list(csv.DictReader(OOF.open()))
    labels = [float(r["label"]) for r in rows]
    answers = [float(r["predicted"]) for r in rows]
    errors = [answer - label for answer, label in zip(answers, labels, strict=True)]
    total = len(rows)

    figure, (hit, shape, bands) = plt.subplots(
        1,
        3,
        figsize=(16.5, 4.5),
        width_ratios=[1.15, 1.0, 0.8],
        gridspec_kw={"wspace": 0.26},
    )

    # Попадание: сколько фото пришлось на каждую пару "метка - ответ".
    counts = collections.Counter(zip(labels, answers, strict=True))
    hit.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle=(0, (5, 4)), zorder=1)
    hit.fill_between([0, 100], [-10, 90], [10, 110], color=BAND, zorder=0)
    hit.scatter(
        [pair[0] for pair in counts],
        [pair[1] for pair in counts],
        s=[value * 7 for value in counts.values()],
        color=MAIN,
        alpha=0.62,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    hit.set_xlim(-6, 106)
    hit.set_ylim(-6, 106)
    hit.set_xlabel("метка эксперта, %", fontsize=12, color=MUTED)
    hit.set_ylabel("ответ модели, %", fontsize=12, color=MUTED)
    hit.text(
        3,
        99,
        "полоса ±10 пунктов\nразмер круга - сколько фото",
        fontsize=10.5,
        color=MUTED,
        va="top",
    )
    dress(hit, f"Ответ против метки, {total} фото")
    hit.grid(color=GRID, zorder=0)

    # Форма ошибки: промах со знаком, по сетке в 5 пунктов.
    inside = sum(abs(e) <= 10 for e in errors) / total * 100
    spread = collections.Counter(round(e / 5.0) * 5 for e in errors)
    edges = sorted(spread)
    shape.axvspan(-10, 10, color=BAND, zorder=0)
    shape.bar(
        edges,
        [spread[e] / total * 100 for e in edges],
        width=4.2,
        color=[MAIN if abs(e) <= 10 else WARM for e in edges],
        zorder=3,
    )
    shape.axvline(0, color=MUTED, linewidth=1, zorder=4)
    shape.set_xlim(-55, 55)
    shape.set_xlabel("ответ минус метка, пункты", fontsize=12, color=MUTED)
    shape.set_ylabel("доля фото, %", fontsize=12, color=MUTED)
    shape.annotate(
        f"{inside:.0f}% ответов\nв полосе ±10",
        xy=(10, spread[0] / total * 100 * 0.55),
        xytext=(21, spread[0] / total * 100 * 0.82),
        fontsize=12,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.1},
    )
    dress(shape, f"Промах со знаком, смещение {st.mean(errors):+.2f}")
    shape.grid(axis="y", color=GRID, zorder=0)

    # Потеря по диапазонам загрузки.
    names, values, sizes = [], [], []
    for name, low, high, last in RANGES:
        chosen = [
            index
            for index, label in enumerate(labels)
            if (label >= low if last else low <= label < high)
        ]
        names.append(name)
        values.append(st.mean(abs(errors[i]) for i in chosen))
        sizes.append(len(chosen))
    colors = [WARM if v == max(values) else MAIN for v in values]
    bands.bar(names, values, width=0.68, color=colors, zorder=3)
    for index, (value, size) in enumerate(zip(values, sizes, strict=True)):
        bands.text(index, value + 0.22, f"{value:.2f}", ha="center", fontsize=14, color=INK)
        bands.text(index, 0.34, f"{size}", ha="center", fontsize=12, color="white")
    bands.set_ylim(0, max(values) * 1.28)
    bands.set_ylabel("MAE, пункты; в столбце - сколько фото", fontsize=11, color=MUTED)
    dress(bands, "Где теряем")
    bands.grid(axis="y", color=GRID, zorder=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"{OUT}: {OUT.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
