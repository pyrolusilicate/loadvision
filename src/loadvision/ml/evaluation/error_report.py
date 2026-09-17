"""Where the model is wrong, as one HTML page: numbers, charts and the photos themselves.

The point is not the average error but its shape: which loads, which scenes and which shooting
conditions carry it.
"""

import html
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from loadvision.ml.data.manifest import Sample
from loadvision.ml.evaluation.charts import bars, card, stat
from loadvision.ml.evaluation.ensemble import Member
from loadvision.ml.evaluation.metrics import summary

WORST_SHOWN: Final = 24
_THUMBNAIL: Final = (32, 32)
_LIGHT_ORDER: Final = ("0-40", "40-60", "60-80", "80-100", "100+")
_LIGHT_EDGES: Final = (40, 60, 80, 100)
_SIZE_ORDER: Final = ("1", "2-3", "4+")
_WORST_GROUPS: Final = 10
_BIN_WIDTH: Final = 10


def brightness_of(samples: Sequence[Sample]) -> NDArray[np.float64]:
    """Mean brightness of every photo, 0 to 255, measured on a thumbnail.

    Rotation does not change it, so the photos are read without the EXIF handling the model
    input needs.

    Args:
        samples: Photos to measure.

    Raises:
        OSError: If a photo cannot be read.
    """
    values: list[float] = []
    for item in samples:
        with Image.open(item.path) as image:
            values.append(float(np.asarray(image.convert("L").resize(_THUMBNAIL)).mean()))
    return np.array(values)


def skewness(values: NDArray[np.float64]) -> float:
    """Returns how lopsided the distribution is; below zero means a longer left tail."""
    centered = values - values.mean()
    variance = float((centered**2).mean())
    return float((centered**3).mean() / variance**1.5) if variance else 0.0


def kurtosis(values: NDArray[np.float64]) -> float:
    """Returns how heavy the tails are compared with a normal distribution, which gives zero."""
    centered = values - values.mean()
    variance = float((centered**2).mean())
    return float((centered**4).mean() / variance**2 - 3.0) if variance else 0.0


def bucket_mae(
    keys: Sequence[str], errors: NDArray[np.float64], order: Sequence[str]
) -> list[tuple[str, float]]:
    """Mean error inside every bucket, in the given order; empty buckets are left out."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for key, error in zip(keys, errors, strict=True):
        grouped[key].append(float(error))
    return [(key, float(np.mean(grouped[key]))) for key in order if grouped[key]]


def label_bin(value: float) -> str:
    """Returns the name of the ten-point bucket a load falls into, with 100 on its own."""
    if value >= 100:
        return "100"
    start = min(int(value // _BIN_WIDTH) * _BIN_WIDTH, 90)
    return f"{start}-{start + _BIN_WIDTH - 1}"


def light_bin(value: float) -> str:
    """Returns the name of the brightness bucket a photo falls into."""
    for name, top in zip(_LIGHT_ORDER, _LIGHT_EDGES, strict=False):
        if value < top:
            return name
    return _LIGHT_ORDER[-1]


def build_report(
    samples: Sequence[Sample],
    labels: NDArray[np.float64],
    predicted: NDArray[np.float64],
    folds: Sequence[int],
    light: NDArray[np.float64],
    members: Sequence[Member],
) -> str:
    """Renders the report of one ensemble.

    Args:
        samples: Labelled photos, in the order of the predictions.
        labels: Reference loads.
        predicted: Out-of-fold predictions.
        folds: Fold number of every photo.
        light: Brightness of every photo.
        members: Members of the ensemble, for the header.

    Returns:
        The page as text.
    """
    signed = predicted - labels
    errors = np.abs(signed)
    metrics = summary(labels, predicted, folds)
    skew = skewness(signed)
    tails = kurtosis(signed)

    group_sizes = Counter(item.group_id for item in samples)
    bin_order = [f"{start}-{start + 9}" for start in range(0, 100, _BIN_WIDTH)] + ["100"]
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
    mae_by_light = bucket_mae([light_bin(v) for v in light], errors, _LIGHT_ORDER)

    def size_bin(item: Sample) -> str:
        count = group_sizes[item.group_id]
        return "1" if count == 1 else ("2-3" if count <= 3 else "4+")

    mae_by_size = bucket_mae([size_bin(item) for item in samples], errors, _SIZE_ORDER)
    mae_by_fold = bucket_mae([str(n) for n in folds], errors, [str(n) for n in sorted(set(folds))])

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

    group_errors: dict[str | None, list[float]] = defaultdict(list)
    for item, error in zip(samples, errors, strict=True):
        group_errors[item.group_id].append(float(error))
    multi = sorted(
        ((g, float(np.mean(v)), len(v)) for g, v in group_errors.items() if len(v) > 1),
        key=lambda row: -row[1],
    )[:_WORST_GROUPS]
    group_rows = "".join(
        f"<tr><td>{html.escape(str(g)[4:12])}</td><td>{n}</td><td>{m:.1f}</td></tr>"
        for g, m, n in multi
    )
    members_text = ", ".join(f"{m.encoder} alpha={m.alpha:g}" for m in members)

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
{stat(f"{tails:+.2f}", "kurtosis", "тяжесть хвостов")}
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
            f"skewness {skew:+.2f}, kurtosis {tails:+.2f}. "
            + (
                "Перекос влево: крупные ошибки чаще в занижение."
                if skew < -0.2
                else "Перекос вправо: крупные ошибки чаще в завышение."
                if skew > 0.2
                else "Заметного перекоса нет, промахи в обе стороны симметричны."
            )
            + (
                " Хвосты тяжелее нормальных: редкие очень большие промахи есть."
                if tails > 1
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
