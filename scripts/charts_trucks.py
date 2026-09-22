"""Четыре рейса, один тариф: сколько кузова едет пустым."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

SOURCE = Path("data/train/train.csv")
OUT = Path("runs/charts/trucks.png")

BODY = "#dfe5e9"
OUTLINE = "#8a99a3"
CABIN = "#c3ced5"
WHEEL = "#37474f"
INK = "#1b2a33"
MUTED = "#5b6a73"
SHOUT = "#b03a2e"

BANDS = [
    (0, 20, 0.12, "20% и меньше", "#b03a2e"),
    (20, 50, 0.36, "20-50%", "#d98c5f"),
    (50, 90, 0.70, "50-90%", "#a8b5bd"),
    (90, 101, 0.96, "90% и больше", "#3d7ea6"),
]


def truck(ax, share: float, cargo: str) -> None:
    """Рисует фуру сбоку: кузов, кабина, колёса и груз, сложенный к кабине."""
    left, right, floor, roof = 0.0, 10.0, 1.3, 5.2

    filled = (right - left) * share
    ax.add_patch(
        Rectangle((right - filled, floor), filled, roof - floor, facecolor=cargo, edgecolor="none")
    )
    ax.add_patch(
        Rectangle(
            (left, floor),
            right - left,
            roof - floor,
            facecolor="none",
            edgecolor=OUTLINE,
            linewidth=2.2,
        )
    )
    # Кабина и капот справа.
    ax.add_patch(
        Rectangle(
            (right + 0.15, floor), 2.4, 2.5, facecolor=CABIN, edgecolor=OUTLINE, linewidth=2.2
        )
    )
    ax.add_patch(
        Polygon(
            [
                (right + 2.55, floor),
                (right + 3.5, floor),
                (right + 3.5, floor + 1.1),
                (right + 2.55, floor + 1.6),
            ],
            closed=True,
            facecolor=CABIN,
            edgecolor=OUTLINE,
            linewidth=2.2,
        )
    )
    ax.add_patch(
        Rectangle(
            (right + 1.45, floor + 1.5),
            1.0,
            0.85,
            facecolor="#f4f7f9",
            edgecolor=OUTLINE,
            linewidth=1.4,
        )
    )
    for x in (1.9, 7.6, 11.9):
        ax.add_patch(Circle((x, floor - 0.45), 0.82, facecolor=WHEEL, edgecolor="none"))
        ax.add_patch(Circle((x, floor - 0.45), 0.34, facecolor="#90a4ae", edgecolor="none"))
    ax.plot([-0.6, 14.2], [floor - 1.3, floor - 1.3], color=OUTLINE, linewidth=2)

    ax.set_xlim(-1.2, 14.6)
    ax.set_ylim(-0.6, 7.4)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    values = [float(row["load_pct"]) for row in csv.DictReader(SOURCE.open())]
    total = len(values)

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[3, 1.25],
        hspace=0.05,
        wspace=0.05,
        left=0.03,
        right=0.97,
        top=0.86,
        bottom=0.04,
    )

    for column, (low, high, share, label, colour) in enumerate(BANDS):
        ax = fig.add_subplot(grid[0, column])
        truck(ax, share, colour)
        count = sum(1 for value in values if low <= value < high)
        ax.text(6.5, 6.6, label, ha="center", fontsize=16, color=INK, weight="bold")
        ax.text(6.5, 5.9, f"{count} рейсов из {total}", ha="center", fontsize=12, color=MUTED)

    fig.suptitle("Один тариф на все четыре", fontsize=30, color=INK, y=0.965, weight="bold")

    shout = fig.add_subplot(grid[1, :])
    shout.axis("off")
    shout.text(
        0.5,
        0.92,
        "Рейс стоит одинаково: водитель, топливо, время в пути. Что внутри - на цену не влияет.",
        fontsize=15,
        color=INK,
        ha="center",
        va="top",
    )
    shout.text(
        0.5,
        0.52,
        "Между первым и четвёртым - 80 процентов кузова, "
        "которые проехали через всю страну впустую.",
        fontsize=15,
        color=SHOUT,
        ha="center",
        va="top",
    )
    shout.text(
        0.5,
        0.12,
        "И никто не знает, какой из них какой: загрузку называют на глаз и нигде не записывают.",
        fontsize=13.5,
        color=MUTED,
        ha="center",
        va="top",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor="white", bbox_inches="tight")
    print(f"{OUT}: {OUT.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
