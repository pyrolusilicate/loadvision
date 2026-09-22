"""Распределение оценок эксперта в обучающем наборе: форма и что она означает."""

import collections
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SOURCE = Path("data/train/train.csv")
OUT = Path("runs/charts/labels.png")

BAR = "#8fa3ae"
PEAK = "#3d7ea6"
BAND = "#e8eef2"
MEDIAN = "#b03a2e"
MEAN = "#c98b3a"
INK = "#1b2a33"
MUTED = "#5b6a73"


def shape(values: list[float]) -> dict[str, float]:
    """Асимметрия, эксцесс и коэффициент бимодальности выборки."""
    count = len(values)
    mean = st.mean(values)
    deviation = st.pstdev(values)
    skew = sum((v - mean) ** 3 for v in values) / count / deviation**3
    kurtosis = sum((v - mean) ** 4 for v in values) / count / deviation**4 - 3
    return {
        "mean": mean,
        "median": st.median(values),
        "deviation": deviation,
        "skew": skew,
        "kurtosis": kurtosis,
        # Коэффициент Пирсона: выше 0.555 - распределение скорее двугорбое, чем одногорбое.
        "bimodality": (skew**2 + 1) / (kurtosis + 3),
    }


def main() -> None:
    values = [float(row["load_pct"]) for row in csv.DictReader(SOURCE.open())]
    counts = sorted(collections.Counter(int(value) for value in values).items())
    total = len(values)
    facts = shape(values)
    low, high = st.quantiles(values, n=4)[0], st.quantiles(values, n=4)[2]
    modes = [value for value, _ in sorted(counts, key=lambda pair: -pair[1])[:2]]
    middle_third = 100 * sum(33 <= v <= 67 for v in values) / total

    fig, (ax, notes) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[4, 1], gridspec_kw={"hspace": 0.32}
    )

    ax.axvspan(low, high, color=BAND, zorder=0)
    for value, count in counts:
        ax.bar(value, count, width=3.4, color=PEAK if value in modes else BAR, zorder=3)
        if count >= 20:
            ax.text(value, count + 2, str(count), ha="center", fontsize=9, color=MUTED)

    ax.axvline(facts["median"], color=MEDIAN, linewidth=2, zorder=4)
    ax.axvline(facts["mean"], color=MEAN, linewidth=2, linestyle="--", zorder=4)

    ax.set_title(
        f"Оценки эксперта в обучающем наборе: {total} фотографий, {len(counts)} разных значений",
        fontsize=16,
        loc="left",
        pad=36,
    )
    ax.set_xlabel("загрузка кузова, проценты")
    ax.set_ylabel("сколько фотографий")
    ax.set_xlim(-4, 104)
    ax.set_xticks(range(0, 101, 10))
    ax.grid(axis="y", color="#e2e2e2", linewidth=0.8, zorder=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        handles=[
            Patch(color=PEAK, label=f"две моды: {modes[0]}% и {modes[1]}%"),
            Patch(color=BAR, label="остальные значения"),
            Patch(color=BAND, label=f"половина данных: от {low:.0f} до {high:.0f}%"),
            Line2D([], [], color=MEDIAN, linewidth=2, label=f"медиана {facts['median']:.0f}%"),
            Line2D(
                [],
                [],
                color=MEAN,
                linewidth=2,
                linestyle="--",
                label=f"среднее {facts['mean']:.1f}%",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        frameon=False,
        fontsize=9.5,
    )

    notes.axis("off")
    lines = [
        f"Распределение двугорбое, а не колоколом: коэффициент бимодальности "
        f"{facts['bimodality']:.2f} при пороге 0.555, эксцесс {facts['kurtosis']:+.2f}. "
        f"Асимметрии почти нет ({facts['skew']:+.2f}) - горбы примерно равны.",
        f"Поэтому среднее и медиана здесь ничего не описывают: в средней трети шкалы, "
        f"от 33 до 67%, лежит всего {middle_third:.0f}% снимков.",
        f"Половина набора растянута от {low:.0f} до {high:.0f}% - межквартильный размах "
        f"{high - low:.0f} пунктов при шкале в сто. Снимать кузов возят либо почти пустым, "
        f"либо под завязку.",
    ]
    for index, line in enumerate(lines):
        notes.text(
            0.0,
            0.86 - index * 0.34,
            line,
            fontsize=11,
            color=INK if index == 0 else MUTED,
            va="top",
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"{OUT}: {OUT.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
