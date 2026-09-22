"""Раскладывает замеренное время ответа на постоянную часть, глубину и один проход энкодера.

Ответ пакета складывается из трёх независимых слагаемых:

    ответ = постоянное + глубина * есть_ли_она + моделей * энкодер

Постоянное - это декодирование JPEG и letterbox, они считаются на процессоре при любом
устройстве. Три замера на одном устройстве задают три уравнения, дальше время любого другого
набора моделей считается без замера. Проверка разложения - постоянная часть должна сойтись с
отдельно замеренным декодированием, около 7 мс.
"""

import argparse
import json
from pathlib import Path
from typing import Final, NamedTuple

BENCH: Final = Path("runs/bench")


class Point(NamedTuple):
    """Один замер: сколько моделей внутри, есть ли глубина, и что показал p50."""

    name: str
    members: int
    depth: bool
    p50_ms: float


class Terms(NamedTuple):
    """Три слагаемых ответа, в миллисекундах."""

    fixed: float
    depth: float
    encoder: float

    def predict(self, members: int, depth: bool) -> float:
        """Время ответа набора из `members` моделей, с глубиной или без."""
        return self.fixed + (self.depth if depth else 0.0) + members * self.encoder


def fit(single: Point, single_depth: Point, many_depth: Point) -> Terms:
    """Решает три уравнения по трём замерам одного и того же устройства.

    Args:
        single: Одна модель без глубины.
        single_depth: Одна модель с глубиной.
        many_depth: Несколько моделей с глубиной.

    Returns:
        Постоянная часть, цена глубины и цена одного прохода энкодера.
    """
    depth = single_depth.p50_ms - single.p50_ms
    encoder = (many_depth.p50_ms - single_depth.p50_ms) / (many_depth.members - 1)
    return Terms(single.p50_ms - encoder, depth, encoder)


def read(run: Path) -> tuple[str, float, int, bool]:
    """Достаёт из отчёта `bench` имя пакета, p50 и состав ансамбля."""
    report = json.loads(run.read_text())
    manifest = json.loads((Path(report["package"]) / "manifest.json").read_text())
    members = manifest["members"]
    depth = any("depth" in str(member.get("inputs", [])) for member in members)
    return report["version"], report["one_client_ms"]["p50"], len(members), depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p50",
        nargs=3,
        type=float,
        metavar=("ONE", "ONE_DEPTH", "MANY_DEPTH"),
        required=True,
        help="p50 трёх замеров на одном устройстве, мс",
    )
    parser.add_argument("--many", type=int, default=5, help="моделей в третьем замере")
    parser.add_argument("--device", default="устройство", help="как подписать вывод")
    args = parser.parse_args()

    one, one_depth, many_depth = args.p50
    terms = fit(
        Point("одна", 1, False, one),
        Point("одна + глубина", 1, True, one_depth),
        Point(f"{args.many} + глубина", args.many, True, many_depth),
    )
    print(f"{args.device}:")
    print(f"  постоянное, декод и letterbox : {terms.fixed:8.1f} мс")
    print(f"  глубина, один проход          : {terms.depth:8.1f} мс")
    print(f"  энкодер, один проход          : {terms.encoder:8.1f} мс")
    print()
    for members, depth, label in (
        (1, False, "одна модель"),
        (1, True, "одна + глубина"),
        (5, True, "пять фолдов + глубина"),
        (5, False, "пять фолдов без глубины"),
        (25, True, "смесь 25 моделей"),
        (30, True, "смесь 30 моделей"),
    ):
        print(f"  {label:26s} {terms.predict(members, depth):8.0f} мс")


def demo() -> None:
    """Самопроверка на замерах CPU: постоянная часть обязана сойтись с декодированием."""
    terms = fit(
        Point("v2", 1, False, 250.0),
        Point("v3", 1, True, 423.0),
        Point("v4", 5, True, 1394.0),
    )
    assert 5.0 < terms.fixed < 10.0, terms  # отдельный замер декодирования дал 6.7 мс
    assert abs(terms.predict(1, False) - 250.0) < 0.5
    assert abs(terms.predict(5, True) - 1394.0) < 0.5
    assert abs(terms.predict(30, True) - 7463.0) < 1.0
    print("demo ok:", terms)


if __name__ == "__main__":
    main()
