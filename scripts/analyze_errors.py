"""Explains where the model is wrong, as numbers, charts and the photos themselves.

Fits the ensemble out of fold, writes one row per photo to `runs/oof/predictions.csv` and
renders `runs/eda/errors.html`. The point is not the average error but its shape: which loads,
which scenes and which shooting conditions carry it.

Usage:
    uv run python scripts/analyze_errors.py
    uv run python scripts/analyze_errors.py --member dinov3_vitb16:200 --member siglip2_vitb16:300
"""

import argparse
import csv
import html
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import stats

from loadvision.ml.data.manifest import Sample, load_split
from loadvision.ml.data.splits import read_folds
from loadvision.ml.evaluation.cross_validation import out_of_fold
from loadvision.ml.evaluation.metrics import summary
from loadvision.ml.models.backbones import load_backbone
from loadvision.ml.training.embedding_cache import embed_samples
from loadvision.ml.training.head_trainer import FittedHead

CACHE_DIR: Final = Path("runs/cache/embeddings")
FOLDS: Final = Path("configs/splits/folds.json")
PREDICTIONS: Final = Path("runs/oof/predictions.csv")
REPORT: Final = Path("runs/eda/errors.html")
DEFAULT_MEMBERS: Final = (("dinov3_vitb16", 200.0), ("pe_spatial_vitb16", 50.0))

WORST_SHOWN: Final = 24
DARK_THRESHOLD: Final = 60.0

logger = logging.getLogger("errors")


# --------------------------------------------------------------------------- charts


def bars(
    items: list[tuple[str, float]],
    *,
    unit: str = "",
    decimals: int = 2,
    zero_line: bool = False,
) -> str:
    """One-series bar chart as inline SVG, with a native tooltip on every bar."""
    width, height, pad_left, pad_bottom, pad_top = 720, 230, 48, 36, 12
    values = [value for _, value in items]
    low = min(0.0, min(values)) if zero_line else 0.0
    high = max(values) or 1.0
    span = (high - low) or 1.0
    plot_w, plot_h = width - pad_left - 8, height - pad_bottom - pad_top
    slot = plot_w / len(items)
    bar_w = max(2.0, slot * 0.72)

    def y_of(value: float) -> float:
        return pad_top + plot_h * (1 - (value - low) / span)

    out = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for fraction in (0.0, 0.5, 1.0):
        value = low + span * fraction
        y = y_of(value)
        out.append(
            f'<line class="grid" x1="{pad_left}" y1="{y:.1f}" x2="{width - 8}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{pad_left - 6}" y="{y + 4:.1f}" text-anchor="end">'
            f"{value:.{decimals}f}</text>"
        )
    base = y_of(0.0) if zero_line else y_of(low)
    for index, (label, value) in enumerate(items):
        x = pad_left + slot * index + (slot - bar_w) / 2
        top = min(y_of(value), base)
        bar_h = max(abs(y_of(value) - base), 1.0)
        out.append(
            f'<rect class="bar" x="{x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{bar_h:.1f}" rx="3"><title>{html.escape(label)}: '
            f"{value:.{decimals}f}{unit}</title></rect>"
        )
        if len(items) <= 24 or index % max(1, len(items) // 12) == 0:
            out.append(
                f'<text class="tick" x="{x + bar_w / 2:.1f}" y="{height - pad_bottom + 16}" '
                f'text-anchor="middle">{html.escape(label)}</text>'
            )
    out.append("</svg>")
    return "\n".join(out)


def card(title: str, subtitle: str, body: str, note: str = "") -> str:
    tail = f'<p class="note">{note}</p>' if note else ""
    return (
        f'<section class="card"><h3>{title}</h3><p class="sub">{subtitle}</p>{body}{tail}</section>'
    )


def stat(value: str, label: str, hint: str = "") -> str:
    return (
        f'<div class="stat"><div class="v">{value}</div><div class="l">{label}</div>'
        f'<div class="h">{hint}</div></div>'
    )


# --------------------------------------------------------------------------- data


def brightness_of(samples: list[Sample]) -> NDArray[np.float64]:
    values: list[float] = []
    for item in samples:
        with Image.open(item.path) as image:
            values.append(float(np.asarray(image.convert("L").resize((32, 32))).mean()))
    return np.array(values)


def ensemble_oof(
    samples: list[Sample],
    labels: NDArray[np.float64],
    numbers: list[int],
    members: list[tuple[str, float]],
    device: str,
) -> NDArray[np.float64]:
    parts: list[NDArray[np.float64]] = []
    for encoder, alpha in members:
        backbone = load_backbone(encoder, device)
        features = embed_samples(samples, backbone, cache_dir=CACHE_DIR, device=device).astype(
            np.float64
        )
        result = out_of_fold(
            features,
            labels,
            numbers,
            lambda a, b, penalty=alpha: FittedHead.fit(a, b, alpha=penalty).predict,
        )
        logger.info("%s alpha=%g: oof mae %.3f", encoder, alpha, result.metrics["mae"])
        parts.append(result.predictions)
    return np.mean(parts, axis=0)


def bucket_mae(
    keys: list[str], errors: NDArray[np.float64], order: list[str]
) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for key, error in zip(keys, errors, strict=True):
        grouped[key].append(float(error))
    return [(key, float(np.mean(grouped[key]))) for key in order if grouped[key]]


# --------------------------------------------------------------------------- report


def build_report(
    samples: list[Sample],
    labels: NDArray[np.float64],
    predicted: NDArray[np.float64],
    numbers: list[int],
    light: NDArray[np.float64],
    members: list[tuple[str, float]],
) -> str:
    signed = predicted - labels
    errors = np.abs(signed)
    metrics = summary(labels, predicted, numbers)
    skew = float(stats.skew(signed))
    kurtosis = float(stats.kurtosis(signed))

    group_sizes = Counter(item.group_id for item in samples)
    bin_of = lambda value: min(int(value // 10) * 10, 90)  # noqa: E731
    bin_order = [f"{start}-{start + 9}" for start in range(0, 100, 10)] + ["100"]

    def label_bin(value: float) -> str:
        return "100" if value >= 100 else f"{bin_of(value)}-{bin_of(value) + 9}"

    label_keys = [label_bin(float(value)) for value in labels]
    mae_by_label = bucket_mae(label_keys, errors, bin_order)
    bias_by_label = [
        (name, float(np.mean(signed[[i for i, k in enumerate(label_keys) if k == name]])))
        for name, _ in mae_by_label
    ]
    mean_pred_by_label = [
        (name, float(np.mean(predicted[[i for i, k in enumerate(label_keys) if k == name]])))
        for name, _ in mae_by_label
    ]

    light_order = ["0-40", "40-60", "60-80", "80-100", "100+"]

    def light_bin(value: float) -> str:
        for name, top in zip(light_order, (40, 60, 80, 100), strict=False):
            if value < top:
                return name
        return "100+"

    mae_by_light = bucket_mae([light_bin(v) for v in light], errors, light_order)
    size_order = ["1", "2-3", "4+"]

    def size_bin(item: Sample) -> str:
        count = group_sizes[item.group_id]
        return "1" if count == 1 else ("2-3" if count <= 3 else "4+")

    mae_by_size = bucket_mae([size_bin(item) for item in samples], errors, size_order)
    mae_by_fold = bucket_mae(
        [str(n) for n in numbers], errors, [str(n) for n in sorted(set(numbers))]
    )

    edges = np.arange(-60, 65, 5)
    signed_hist, _ = np.histogram(signed, bins=edges)
    abs_hist, abs_edges = np.histogram(errors, bins=np.arange(0, 65, 5))

    worst = np.argsort(-errors)[:WORST_SHOWN]
    gallery = "".join(
        f'<figure><img loading=lazy src="/data/train/images/{samples[i].image_id}.jpg">'
        f"<figcaption>правда <b>{labels[i]:g}</b> · модель <b>{predicted[i]:.0f}</b><br>"
        f'<span class="err">ошибка {errors[i]:.0f}</span><br>'
        f"{html.escape(samples[i].image_id[4:12])} · яркость {light[i]:.0f}</figcaption></figure>"
        for i in worst
    )

    group_errors = defaultdict(list)
    for item, error in zip(samples, errors, strict=True):
        group_errors[item.group_id].append(float(error))
    multi = sorted(
        ((g, float(np.mean(v)), len(v)) for g, v in group_errors.items() if len(v) > 1),
        key=lambda row: -row[1],
    )[:10]
    group_rows = "".join(
        f"<tr><td>{html.escape(str(g)[4:12])}</td><td>{n}</td><td>{m:.1f}</td></tr>"
        for g, m, n in multi
    )

    members_text = ", ".join(f"{name} alpha={alpha:g}" for name, alpha in members)

    return f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Loadvision - разбор ошибок</title>
<style>
:root{{--bg:#0f1115;--surface:#171a21;--ink:#e8eaed;--ink2:#a8b0bd;--ink3:#6f7885;
--grid:#262b35;--accent:#4c9aff;--warn:#ff9f45;--bad:#ff6b6b}}
*{{box-sizing:border-box}}
body{{margin:0;padding:24px 16px 64px;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,system-ui,sans-serif}}
main{{max-width:780px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:38px 0 12px}}
h3{{font-size:16px;margin:0 0 2px}}
.lead{{color:var(--ink2);margin:0 0 20px}}
.card{{background:var(--surface);border:1px solid var(--grid);border-radius:12px;
padding:16px;margin:14px 0}}
.sub{{color:var(--ink3);font-size:13px;margin:0 0 10px}}
.note{{color:var(--ink2);font-size:13px;margin:10px 0 0;border-left:2px solid var(--accent);
padding-left:10px}}
svg{{width:100%;height:auto;display:block}}
.bar{{fill:var(--accent)}} .bar:hover{{fill:#7fb8ff}}
.grid{{stroke:var(--grid);stroke-width:1}}
.tick{{fill:var(--ink3);font-size:11px;font-family:ui-monospace,monospace}}
.stats{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}}
.stat{{flex:1 1 150px;background:var(--surface);border:1px solid var(--grid);
border-radius:12px;padding:12px 14px}}
.stat .v{{font-size:25px;font-weight:650;letter-spacing:-.02em}}
.stat .l{{font-size:13px;color:var(--ink2)}} .stat .h{{font-size:12px;color:var(--ink3)}}
.gal{{display:flex;flex-wrap:wrap;gap:10px}}
.gal figure{{margin:0;width:180px}}
.gal img{{width:180px;height:135px;object-fit:cover;border-radius:6px}}
figcaption{{font-size:12px;color:var(--ink2);line-height:1.35}}
.err{{color:var(--bad)}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}}
th{{color:var(--ink3);font-weight:500;font-size:12px}}
code{{font-family:ui-monospace,monospace;font-size:13px;color:var(--ink2)}}
</style>
<main>
<h1>Разбор ошибок</h1>
<p class="lead">Ансамбль: {html.escape(members_text)}. Предсказания out-of-fold по
<code>configs/splits/folds.json</code>, {len(samples)} фото.</p>

<div class="stats">
{stat(f"{metrics['mae']:.2f}", "MAE", "пунктов")}
{stat(f"{metrics['within_10']:.1%}", "в пределах 10", "доля фото")}
{stat(f"{metrics['p95_error']:.0f}", "p95 ошибки", "у 5% хуже")}
{stat(f"{skew:+.2f}", "skewness", "перекос распределения")}
{stat(f"{kurtosis:+.2f}", "kurtosis", "тяжесть хвостов")}
{stat(f"{metrics['bias']:+.2f}", "bias", "средний сдвиг")}
</div>

<h2>1. Форма ошибки</h2>
{
        card(
            "Распределение ошибки со знаком",
            "минус - модель занизила, плюс - завысила; ось X в пунктах",
            bars(
                [(f"{int(e)}", float(c)) for e, c in zip(edges[:-1], signed_hist, strict=False)],
                decimals=0,
            ),
            f"skewness {skew:+.2f}, kurtosis {kurtosis:+.2f}. "
            + (
                "Перекос влево: крупные ошибки чаще в занижение."
                if skew < -0.2
                else "Перекос вправо: крупные ошибки чаще в завышение."
                if skew > 0.2
                else "Заметного перекоса нет, промахи в обе стороны симметричны."
            )
            + (
                " Хвосты тяжелее нормальных: редкие очень большие промахи есть."
                if kurtosis > 1
                else " Хвосты не тяжелее нормальных."
            ),
        )
    }
{
        card(
            "Распределение модуля ошибки",
            "сколько фото попало в каждый диапазон ошибки",
            bars(
                [(f"{int(e)}", float(c)) for e, c in zip(abs_edges[:-1], abs_hist, strict=False)],
                decimals=0,
            ),
            f"Медиана {np.median(errors):.1f}, 75-й процентиль {np.percentile(errors, 75):.1f}, "
            f"максимум {errors.max():.0f}. Ошибок больше 30 пунктов: "
            f"{int((errors > 30).sum())} из {len(errors)}.",
        )
    }

<h2>2. Где ошибаемся по загрузке</h2>
{card("MAE по истинной метке", "ось X - настоящая загрузка", bars(mae_by_label))}
{
        card(
            "Сдвиг по истинной метке",
            "средняя разница модель минус правда; выше нуля - завышаем",
            bars(bias_by_label, zero_line=True),
            "Классическая картина регрессии: на пустых кузовах модель завышает, на полных "
            "занижает, потому что осторожничает и тянет ответ к середине.",
        )
    }
{
        card(
            "Средний прогноз по истинной метке",
            "идеал - когда столбик совпадает с подписью на оси",
            bars(mean_pred_by_label),
            "Если крайние столбики заметно ниже 100 и выше 0, модель не дотягивает "
            "до краёв диапазона.",
        )
    }

<h2>3. Где ошибаемся по условиям</h2>
{card("MAE по яркости кадра", "0-40 - почти чёрные кадры", bars(mae_by_light))}
{card("MAE по размеру группы", "сколько фото в сцене", bars(mae_by_size))}
{
        card(
            "MAE по фолдам",
            "разброс показывает, насколько метрика шумная",
            bars(mae_by_fold),
            f"Разброс {metrics['mae_fold_spread']:.2f} пункта. Улучшения меньше этого значения "
            "нельзя считать настоящими без проверки на нескольких разбиениях.",
        )
    }

<h2>4. Худшие группы</h2>
<section class="card">
<p class="sub">Сцены из нескольких фото, отсортированные по средней ошибке</p>
<table><tr><th>группа</th><th>фото</th><th>средняя ошибка</th></tr>{group_rows}</table>
</section>

<h2>5. Худшие {WORST_SHOWN} фото</h2>
<section class="card">
<p class="sub">Самые большие промахи. Сюда стоит смотреть глазами: если у них есть общее,
это и есть следующая задача.</p>
<div class="gal">{gallery}</div>
</section>
</main>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", action="append", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    members = (
        [(m.split(":")[0], float(m.split(":")[1])) for m in args.member]
        if args.member
        else list(DEFAULT_MEMBERS)
    )
    samples = load_split(args.data, "train")
    folds = read_folds(FOLDS)
    numbers = folds.fold_indices(samples)
    labels = np.array([s.label for s in samples], dtype=np.float64)

    predicted = ensemble_oof(samples, labels, numbers, members, args.device)
    light = brightness_of(samples)

    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink)
        writer.writerow(
            ["image_id", "group_id", "fold", "label", "predicted", "error", "brightness"]
        )
        for item, fold, true, pred, value in zip(
            samples, numbers, labels, predicted, light, strict=True
        ):
            writer.writerow(
                [
                    item.image_id,
                    item.group_id,
                    fold,
                    f"{true:g}",
                    f"{pred:.2f}",
                    f"{pred - true:.2f}",
                    f"{value:.1f}",
                ]
            )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        build_report(samples, labels, predicted, numbers, light, members), encoding="utf-8"
    )
    metrics: dict[str, Any] = summary(labels, predicted, numbers)
    logger.info("")
    logger.info("ensemble oof mae %.3f", metrics["mae"])
    logger.info("written %s and %s", PREDICTIONS, REPORT)


if __name__ == "__main__":
    main()
