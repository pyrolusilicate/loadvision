"""Почти одинаковые кадры с разными оценками: сколько стоит разметка на глаз."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = Path("runs/charts/twins/pairs.json")
IMAGES = Path("runs/charts/twins")
OUT = Path("runs/charts/twins.png")

INK = "#1b2a33"
MUTED = "#5b6a73"
ACCENT = "#b03a2e"


def main() -> None:
    pairs = json.loads(PAIRS.read_text())
    fig, axes = plt.subplots(
        2,
        len(pairs) * 2,
        figsize=(15, 9),
        height_ratios=[10, 1],
        gridspec_kw={"hspace": 0.05, "wspace": 0.06},
    )

    for index, pair in enumerate(pairs):
        for offset, side in enumerate(("a", "b")):
            column = index * 2 + offset
            ax = axes[0, column]
            ax.imshow(plt.imread(IMAGES / f"pair{index + 1}_{side}.jpg"))
            ax.axis("off")
            ax.set_title(f"{pair['l' + side]:.0f}%", fontsize=26, color=INK, pad=10)

        note = axes[1, index * 2]
        note.axis("off")
        note.text(
            1.03,
            0.75,
            f"оценки расходятся на {pair['gap']:.0f} пунктов\n"
            f"похожесть кадров {pair['cos']:.2f} из 1.00",
            fontsize=12,
            color=ACCENT,
            ha="center",
            va="top",
        )
        axes[1, index * 2 + 1].axis("off")

    fig.suptitle(
        "Три пары кадров, которые модель видит как почти одинаковые, а эксперт оценил по-разному",
        fontsize=17,
        color=INK,
        y=0.97,
    )
    fig.text(
        0.5,
        0.055,
        "Из 95 таких пар в наборе у 38 оценки расходятся на 10 пунктов и больше, "
        "у 14 - на 20 и больше.\n"
        "Это не ошибка разметчика, а предел способа: загрузку оценивают на глаз, "
        "и двух одинаковых оценок не бывает.",
        fontsize=12.5,
        color=MUTED,
        ha="center",
        va="top",
        linespacing=1.6,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor="white", bbox_inches="tight")
    print(f"{OUT}: {OUT.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
