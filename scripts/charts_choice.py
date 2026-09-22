"""Почему в поставке линейная голова: куда дотягивается ответ каждой из двух моделей."""

import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

V4 = Path("runs/finetune/run_20260919_183257_877c1e/oof.csv")
V5 = Path("runs/finetune/run_20260921_111551_c01b9b/oof.csv")
OUT = Path("runs/charts/choice.png")

BAR = "#d3dce2"
LINEAR = "#3d7ea6"
BINS = "#c98b3a"
WALL = "#b03a2e"
INK = "#1b2a33"
MUTED = "#5b6a73"

STEP = 5.0


def read(path: Path) -> tuple[list[float], list[float]]:
    """Метки и сырые, ещё не калиброванные ответы прогона."""
    rows = list(csv.DictReader(path.open()))
    return ([float(r["label"]) for r in rows], [float(r["raw"]) for r in rows])


def counts(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Доли выборки по корзинам шириной в один шаг сетки эксперта."""
    edges = np.arange(-STEP / 2, 100 + STEP, STEP)
    hist, _ = np.histogram(values, bins=edges)
    return edges[:-1] + STEP / 2, hist / len(values) * 100


def main() -> None:
    labels, raw_linear = read(V4)
    _, raw_bins = read(V5)
    lo, hi = min(raw_bins), max(raw_bins)
    extreme = sum(v in (0.0, 100.0) for v in labels)

    fig, (ax, notes) = plt.subplots(
        2, 1, figsize=(14, 7.2), height_ratios=[4, 1], gridspec_kw={"hspace": 0.34}
    )

    centres, share = counts(labels)
    ax.bar(centres, share, width=4.2, color=BAR, zorder=2, label="метки эксперта")
    for values, color, name in ((raw_linear, LINEAR, "линейная"), (raw_bins, BINS, "bins")):
        _, line = counts(values)
        ax.plot(centres, line, color=color, linewidth=2.6, zorder=4, label=name)
        ax.scatter([min(values), max(values)], [0, 0], s=46, color=color, zorder=6, clip_on=False)

    for span in ((-2.5, lo), (hi, 102.5)):
        ax.axvspan(*span, color=WALL, alpha=0.1, zorder=1)
    ax.axvline(lo, color=WALL, linewidth=1.6, linestyle=(0, (5, 3)), zorder=5)
    ax.axvline(hi, color=WALL, linewidth=1.6, linestyle=(0, (5, 3)), zorder=5)

    top = max(share)
    ax.annotate(
        f"ответ bins - среднее распределения\nпо 21 корзине, а softmax не бывает\n"
        f"единичным: ниже {lo:.1f} и выше {hi:.1f}\nэта голова не отвечает никогда",
        xy=(hi, top * 0.30),
        xytext=(41, top * 0.62),
        fontsize=12.5,
        color=WALL,
        ha="left",
        va="top",
        arrowprops={
            "arrowstyle": "->",
            "color": WALL,
            "linewidth": 1.2,
            "connectionstyle": "arc3,rad=-0.25",
        },
    )
    ax.annotate(
        f"{extreme} фото из {len(labels)} эксперт оценил\nровно в 0 или 100 - линейная голова\n"
        f"попадает туда сама, bins только\nпосле калибровки",
        xy=(100, share[-1]),
        xytext=(45, top * 1.30),
        fontsize=12.5,
        color=INK,
        ha="left",
        va="top",
        arrowprops={
            "arrowstyle": "->",
            "color": MUTED,
            "linewidth": 1.2,
            "connectionstyle": "arc3,rad=-0.12",
        },
    )

    ax.set_xlim(-2.5, 102.5)
    ax.set_ylim(0, top * 1.34)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("загрузка кузова, %", fontsize=13, color=MUTED, labelpad=8)
    ax.set_ylabel("доля выборки, %", fontsize=13, color=MUTED, labelpad=8)
    ax.set_title(
        "Сырой ответ до калибровки: bins упирается в стены 5.8 и 95.7 и копит массу у них",
        fontsize=15,
        color=INK,
        pad=16,
        loc="left",
    )
    ax.tick_params(colors=MUTED, labelsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c6d0d6")
    ax.grid(axis="y", color="#eef2f4", zorder=0)
    ax.legend(
        handles=[
            Patch(facecolor=BAR, label=f"метки эксперта, {len(labels)} фото"),
            Line2D(
                [],
                [],
                color=LINEAR,
                linewidth=2.6,
                label=f"линейная 960 → 1: размах {min(raw_linear):.1f} - {max(raw_linear):.1f}",
            ),
            Line2D(
                [],
                [],
                color=BINS,
                linewidth=2.6,
                label=f"bins, 21 корзина: размах {lo:.1f} - {hi:.1f}",
            ),
            Patch(facecolor=WALL, alpha=0.1, label="зона, недостижимая для bins"),
        ],
        loc="upper left",
        fontsize=12,
        frameon=False,
        labelcolor=INK,
    )

    def median_at(raw: list[float]) -> float:
        """Медиана сырого ответа на фото, которые эксперт оценил ровно в 100."""
        pairs = zip(raw, labels, strict=True)
        return st.median([answer for answer, label in pairs if label == 100])

    def middle(path: Path) -> float:
        """MAE в середине диапазона, как её считает прогон."""
        import json

        return json.loads((path.parent / "metrics.json").read_text())["mae_20-80"]

    plates = [
        (
            "размах сырого ответа",
            f"{min(raw_linear):.1f} - {max(raw_linear):.1f}",
            f"{lo:.1f} - {hi:.1f}",
        ),
        (
            f"медиана на метке 100, {labels.count(100)} фото",
            f"{median_at(raw_linear):.1f}",
            f"{median_at(raw_bins):.1f}",
        ),
        ("MAE в середине 20-80", f"{middle(V4):.2f}", f"{middle(V5):.2f}"),
    ]
    notes.set_axis_off()
    for index, (title, left, right) in enumerate(plates):
        x = index / 3 + 0.005
        notes.add_patch(
            plt.Rectangle(
                (x, 0.0),
                1 / 3 - 0.015,
                1.0,
                transform=notes.transAxes,
                facecolor="#f4f7f9",
                edgecolor="#dde5ea",
                zorder=0,
            )
        )
        notes.text(x + 0.018, 0.78, title, transform=notes.transAxes, fontsize=11.5, color=MUTED)
        notes.text(
            x + 0.018,
            0.42,
            f"линейная  {left}",
            transform=notes.transAxes,
            fontsize=14,
            color=LINEAR,
        )
        notes.text(
            x + 0.018, 0.11, f"bins  {right}", transform=notes.transAxes, fontsize=14, color=BINS
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"{OUT}: {OUT.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
