# Loadvision

Сервис оценки загрузки грузового отсека по фотографии. Он возвращает число от 0 до 100
процентов, сохраняет его вместе с номером перевозки, отдаёт по XML API и умеет дообучаться на
разметке, которую подтвердил оператор.

**Качество.** MAE 5.76 на кросс-валидации по 716 фотографиям и 4.78 на публичном рейтинге
платформы. Базовое решение даёт 19.1. В 87 процентах случаев прогноз расходится с экспертной
меткой не больше чем на 10 пунктов.

**Скорость.** 222 мс на фотографию на видеокарте L4 в bfloat16 и около 1.4 секунды на
процессоре ноутбука. Это время самой модели, а разбор XML, декодирование и работа с базой
добавляют на той же карте около 23 мс.

**Лицензии.** Все веса распространяются под Apache-2.0. В пакет входят дообученный
DINOv2 ViT-B/14 и сеть глубины Depth Anything V2 Small, тексты лицензий и NOTICE хранятся
внутри самого пакета.

## Требования

- [uv](https://docs.astral.sh/uv/) 0.12 и новее
- Python 3.12, его установит uv
- Docker для локального Postgres
- [just](https://github.com/casey/just) для коротких команд, необязательно

## Запуск сервиса

Веса не хранятся в репозитории, поскольку пакет модели занимает два гигабайта. Он опубликован
на Hugging Face.

```bash
uv sync
cp .env.example .env            # пароль базы в POSTGRES_PASSWORD и в DATABASE_URL
just up                         # Postgres, миграции и API на http://localhost:8000

uv run --with huggingface_hub hf download Roman3427/loadvision-truck-load-v4 \
    --local-dir storage/handoff/v4
uv run --env-file .env loadvision model register storage/handoff/v4
uv run --env-file .env loadvision model activate v4

uv run --env-file .env loadvision operators add <логин>   # учётная запись для страниц
```

Готовность проверяется запросом `curl -s localhost:8000/health`, он должен ответить
`modelVersion="v4"`. Без трёх команд с моделью сервис поднимется на пакете `baseline-v0`,
у которого MAE 19.1.

Страницы оператора открываются по адресу `http://localhost:8000/ui`, XML API доступен по
`http://localhost:8000/api/v1`. Полный список методов и примеры запросов собраны в
[docs/running.md](docs/running.md).

На macOS модель считается заметно быстрее вне контейнера, поэтому вместо `just up` удобнее
поднять базу через `just db-up` и запустить сервис командой `just serve`.

## Проверки

```bash
scripts/check.sh                       # ruff, pyright, import-linter, быстрые тесты
git config core.hooksPath .githooks    # хук, запускающий их перед каждым push
just integration                       # тесты на реальном Postgres
```

## Команды

### Сервис и модель

```bash
uv run --env-file .env loadvision model register <папка>   # добавить пакет модели
uv run --env-file .env loadvision model activate <версия>  # включить версию в сервисе
uv run --env-file .env loadvision model build-baseline     # собрать базовый пакет
uv run --env-file .env loadvision operators add <логин>    # завести оператора
uv run --env-file .env loadvision-worker                   # воркер дообучения
uv run loadvision bench --package storage/models/v4        # задержка и пропускная способность
```

### Обучение

```bash
uv run loadvision finetune configs/finetune/dinov2_b14_depth.json --device cuda
uv run loadvision model build-finetuned runs/finetune/<прогон> --version v4 --device cuda
uv run loadvision train configs/experiments/<имя>.json --device mps
uv run loadvision predict submission --package storage/models/v4
```

Первая пара команд собирает поставляемую модель и требует видеокарты. Команда `train` проходит
весь пайплайн на замороженных энкодерах и считается минутами на процессоре, её же использует
воркер дообучения. Устройство задаётся флагом `--device` со значениями `cpu`, `mps` или `cuda`.

### Данные и замеры

```bash
uv run loadvision data folds                   # собрать configs/splits/folds.json
uv run loadvision data folds --check           # проверить существующий файл
uv run loadvision features probe               # kNN-зонд по реестру энкодеров
uv run loadvision eval heads                   # перебор энкодеров и голов по OOF
uv run loadvision eval postprocess             # перебор цепочек постобработки
uv run loadvision eval errors                  # отчёт об ошибках с графиками в runs/eda/
uv run python scripts/ood_scale.py <папка>     # где фотографии лежат на шкале модели
uv run python scripts/ood_line.py --help       # подобрать черту допустимости фотографий
```

Обучение и замеры читают датасет из `data/train` и `data/test` и требуют группы зависимостей
`ml`. Сервису достаточно пакета модели в `storage/models/`.

## Структура

```
configs/            настройки сервиса и конфиги экспериментов
src/loadvision/
  domain/           сущности и правила, без фреймворков
  application/      сценарии и порты
  infrastructure/   Postgres, файлы
  ml/               данные, обучение, инференс, пайплайн
  api/              FastAPI, XML и страницы оператора
  imaging/          проверка и подготовка фотографий
  observability/    логи и записи запусков
  cli/              команды
tests/              unit, contract, api, ml
baseline/           стартовое решение
```

Порядок слоёв проверяет import-linter, зависимости направлены только внутрь.
`api | cli` -> `infrastructure | ml` -> `application` -> `domain`.

## Документация

| Файл | Что внутри |
|---|---|
| [docs/running.md](docs/running.md) | запуск, все команды, примеры запросов к API, частые проблемы |
| [docs/architecture.md](docs/architecture.md) | устройство сервиса, сущности, каталог ошибок, принятые решения |
| [docs/features.md](docs/features.md) | отказ отвечать на чужом кадре, сверка с отчётом, оценка веса, карты внимания |
| [docs/deploy-vm.md](docs/deploy-vm.md) | развёртывание на машине с видеокартой |
| [docs/ml/model.md](docs/ml/model.md) | что внутри поставляемого пакета и почему выбрано именно это |
| [docs/ml/pipeline.md](docs/ml/pipeline.md) | ML-цепочка от фотографии до ответа, файл за файлом |
| [docs/ml/methods.md](docs/ml/methods.md) | каталог взаимозаменяемых методов с замерами |
| [docs/ml/experiments.md](docs/ml/experiments.md) | журнал замеров |
| [docs/ml/labels/](docs/ml/labels/) | 128 фотографий, размеченных вручную по виду груза |
| [CONTRIBUTING.md](CONTRIBUTING.md) | правила репозитория, слои и общие контракты |
