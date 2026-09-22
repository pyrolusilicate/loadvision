"""Зачем эта задача вообще: недогруз, которого не видно в цифрах."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

SOURCE = Path("data/train/train.csv")
OUT = Path("runs/charts/problem.png")

EMPTY = "#b03a2e"
LOW = "#d98c5f"
MID = "#a8b5bd"
FULL = "#3d7ea6"
INK = "#1b2a33"
MUTED = "#5b6a73"

BANDS = [
    (0, 20, EMPTY, "20% и меньше"),
    (20, 50, LOW, "20-50%"),
    (50, 90, MID, "50-90%"),
    (90, 101, FULL, "90% и больше"),
]


def main() -> None:
    values = sorted(float(row["load_pct"]) for row in csv.DictReader(SOURCE.open()))
    total = len(values)
    columns = 44
    rows = math.ceil(total / columns)

    fig, (ax, notes) = plt.subplots(
        2, 1, figsize=(14, 8.6), height_ratios=[4, 1.15], gridspec_kw={"hspace": 0.12}
    )

    for index, value in enumerate(values):
        colour = next(c for low, high, c, _ in BANDS if low <= value < high)
        ax.add_patch(
            Rectangle(
                (index % columns, rows - 1 - index // columns),
                0.82,
                0.82,
                facecolor=colour,
                edgecolor="none",
            )
        )

    ax.set_xlim(-0.5, columns)
    ax.set_ylim(-0.5, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Каждый квадрат - один рейс из {total}, снятых на сортировочных центрах",
        fontsize=17,
        color=INK,
        loc="left",
        pad=30,
    )

    counts = {
        label: sum(1 for value in values if low <= value < high) for low, high, _, label in BANDS
    }
    ax.legend(
        handles=[
            Patch(color=colour, label=f"{label}: {counts[label]} рейсов")
            for _, _, colour, label in BANDS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=11,
    )

    nearly_empty = counts["20% и меньше"]
    mean = sum(values) / total
    notes.axis("off")
    lines = [
        (
            f"Каждый третий кузов уходит заполненным на 20% и меньше: {nearly_empty} рейсов из "
            f"{total}. Рейс оплачен целиком, а везёт пятую часть.",
            INK,
        ),
        (
            f"В среднем по набору кузов заполнен на {mean:.0f}%. Почти половина оплаченного "
            f"объёма едет пустой.",
            MUTED,
        ),
        (
            "Сейчас это видит только приёмщик, который смотрит в кузов и называет число вслух. "
            "Нигде не записано, не сверено с нормативом, не посчитано за месяц.",
            MUTED,
        ),
        (
            "Недогруз, который не попал в цифры, не существует для того, кто принимает решения "
            "о транспорте.",
            EMPTY,
        ),
    ]
    for index, (line, colour) in enumerate(lines):
        notes.text(0.0, 0.92 - index * 0.28, line, fontsize=12, color=colour, va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor="white", bbox_inches="tight")
    print(f"{OUT}: {OUT.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
