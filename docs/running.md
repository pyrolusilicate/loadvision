# Запуск и команды

Версия от 17.09.2026. Здесь только то, что работает на этой версии кода. Почему сделано именно
так - в `architecture.md`.

## Что нужно

| Что | Зачем |
|---|---|
| [uv](https://docs.astral.sh/uv/) 0.12+ | зависимости и запуск, Python 3.12 поставит сам |
| Docker | Postgres и образ сервиса |
| [just](https://github.com/casey/just) | короткие команды, необязательно |
| `data/train`, `data/test` | данные хакатона, в git не лежат |
| `baseline/model.npz` | решение организаторов, тоже не в git |

## Установка

```bash
uv sync                                # все группы зависимостей, включая ml
cp .env.example .env                   # логин, пароль и база Postgres
git config core.hooksPath .githooks    # проверки перед каждым git push
```

## Сервис за три команды

```bash
uv run loadvision model build-baseline      # модель, с которой сервис отвечает
just up                                     # Postgres, миграции и API на :8000
curl -s http://localhost:8000/health
```

`just up` поднимает три контейнера по очереди: `postgres`, затем `migrate` (`alembic upgrade
head`, завершается сам), затем `api`. Модель грузится в фоне уже после старта: пока она не
готова, оценка фото и `/health` отвечают 503 `MODEL_NOT_READY`, а чтение перевозок работает.

Готовый сервис отвечает так:

```xml
<Health xmlns="urn:loadvision:api:v1" modelVersion="baseline-v0"/>
```

### Оценить фото

```bash
photo=$(base64 < data/test/images/img_00b135ede40393ac4459.jpg | tr -d '\n')
printf '<AssessmentRequest xmlns="urn:loadvision:api:v1"><ShipmentId>RP-1</ShipmentId><Image>%s</Image></AssessmentRequest>' "$photo" > /tmp/request.xml
curl -s -X POST http://localhost:8000/api/v1/assessments \
  -H 'Content-Type: application/xml; charset=utf-8' --data-binary @/tmp/request.xml
```

```xml
<Assessment xmlns="urn:loadvision:api:v1" id="5b690640-9c24-4c3c-98b1-d3b311525373"
            shipmentId="RP-1" reused="false">
  <LoadPct>55.0</LoadPct>
  <Image sha256="ac690626..." mediaType="image/jpeg" width="300" height="400"/>
  <Model version="baseline-v0"/>
  <ProcessingMs>21</ProcessingMs>
  <CreatedAt>2026-09-17T15:48:57.231829Z</CreatedAt>
</Assessment>
```

Повтор того же фото в той же перевозке вернёт эту же оценку со статусом 200 и
`reused="true"`: модель второй раз не запускается.

### Подтвердить загрузку, прочитать перевозку

```bash
id=5b690640-9c24-4c3c-98b1-d3b311525373

curl -s -X PUT "http://localhost:8000/api/v1/assessments/$id/label" \
  -H 'Content-Type: application/xml; charset=utf-8' \
  --data '<LabelRequest xmlns="urn:loadvision:api:v1"><LoadPct>60</LoadPct><Author>operator-17</Author></LabelRequest>'

curl -s "http://localhost:8000/api/v1/assessments/$id"     # оценка вместе с меткой
curl -s http://localhost:8000/api/v1/shipments/RP-1        # текущая оценка и история
curl -s http://localhost:8000/api/v1/schema.xsd            # схема, по которой всё проверяется
```

Непригодное фото отмечается так же, но вместо загрузки идёт пустой элемент:
`<LabelRequest xmlns="urn:loadvision:api:v1"><Unusable/></LabelRequest>`.

### Методы API

| Метод | Адрес | Ответ |
|---|---|---|
| POST | `/api/v1/assessments` | 201 новая оценка, 200 повтор с `reused="true"` |
| GET | `/api/v1/assessments/{id}` | 200 оценка с меткой фото, 404 если нет |
| PUT | `/api/v1/assessments/{id}/label` | 200, повтор заменяет метку |
| GET | `/api/v1/shipments/{номер}` | 200 текущая оценка и история, 404 если нет |
| GET | `/api/v1/models` | 200 со списком версий, новые первыми |
| POST | `/api/v1/models/{версия}/activate` | 200 с активной версией, 404 и 409 при отказе |
| POST | `/api/v1/training-jobs` | 201 с задачей, 409 если дообучение уже идёт |
| GET | `/api/v1/training-jobs` | 200 с последними задачами |
| GET | `/api/v1/training-jobs/{id}` | 200 с задачей, 404 если нет |
| GET | `/health` | 200 с версией модели, 503 пока модель грузится |
| GET | `/api/v1/schema.xsd` | XSD API |

Тело запроса и ответа - всегда `application/xml; charset=utf-8`. Любая ошибка приходит
элементом `Error` с кодом из каталога и `requestId`; список кодов - в `architecture.md`.

## Команды `loadvision`

| Команда | Что делает | Что пишет | Что нужно |
|---|---|---|---|
| `model build-baseline` | пакет модели из решения организаторов | `storage/models/<версия>/` | `baseline/`, `data/`, git |
| `model register <пакет>` | проверить пакет, положить в папку моделей, записать версию | `storage/models/<версия>/`, база | база |
| `model activate <версия>` | сделать версию активной, прежнюю отправить в отставку | база | база |
| `model rollback` | вернуть версию, которая работала до текущей | база | база |
| `model list` | версии и их статусы | - | база |
| `jobs start` | поставить дообучение в очередь | база | база |
| `jobs list` | последние задачи и чем они кончились | - | база |
| `train <config>` | весь пайплайн обучения одного эксперимента | `runs/run_<время>_<хвост>/` | `data/`, группа `ml` |
| `predict submission` | прогноз пакетом модели по части датасета | `runs/submissions/<версия>-<часть>.csv` | пакет модели, `data/` |
| `data folds [--check]` | собрать или проверить фолды | `configs/splits/folds.json` | `data/` |
| `features probe [энкодеры]` | kNN-зонд по эмбеддингам, без обучения | `runs/probes/encoders.jsonl` | `data/`, `ml`, веса энкодеров |
| `eval heads [энкодеры]` | перебор энкодеров и голов по OOF | `runs/oof/history.jsonl` | то же |
| `eval postprocess` | сравнение цепочек постобработки | `runs/oof/postprocess.jsonl` | то же |
| `eval errors` | отчёт об ошибках с графиками | `runs/oof/predictions.csv`, `runs/eda/errors.html` | то же |

Общие опции: `--data` (корень датасета), `--device` (`cpu`, `mps`, `cuda`). У `train` есть ещё
`--runs-dir`, `--cache-dir` и `--seed`, у `data folds` - `--out` и `--seed`, у `model
build-baseline` - `--baseline-dir` и `--version`.

Коды возврата: 0 - успех, 1 - понятный отказ (одна строка с причиной, без трейсбека), 2 -
неверные аргументы.

Примеры:

```bash
uv run loadvision train configs/experiments/ens4_median16_round5.json --device mps
uv run loadvision data folds --check
uv run loadvision eval postprocess --member siglip2_vitb16:300 --member dinov2_vitb14:200
uv run loadvision model build-baseline --version baseline-v0
uv run loadvision predict submission                      # тестовая часть пакетом из настроек
uv run loadvision predict submission --package runs/run_20260918_101500_ab12/package \
  --device mps --out submission.csv
```

`predict submission` берёт всё из манифеста пакета: энкодеры, головы и постобработку. Проверка
на пакете baseline: 307 фото за 2,4 секунды на CPU, 8 мс на фото, числа совпали с
`baseline/submission.csv` организаторов с точностью до округления.

Записи `runs/*.jsonl` дописываются, а не перезаписываются: замеры копятся по дням, и старую
строку всегда можно сравнить с новой.

## Жизненный цикл версии модели

Прогон обучения оставляет пакет в папке запуска. Дальше два шага: сначала версия регистрируется,
потом кто-то решает её включить.

```bash
uv run loadvision model register runs/run_20260918_101500_ab12/package
uv run loadvision model activate 2026-09-18.1
docker compose restart api        # сервис берёт активную версию при старте
uv run loadvision model list
```

```
version                 status      registered            activated
2026-09-18.1            active      2026-09-18T10:20:31Z  2026-09-18T10:21:02Z
baseline-v0             retired     2026-09-17T20:45:42Z  2026-09-17T20:45:44Z
```

Если новая версия ведёт себя плохо, возврат - одна команда:

```bash
uv run loadvision model rollback      # включит baseline-v0 из примера выше
docker compose restart api
```

Что проверяет `register`: хэши файлов пакета, манифест и то, что предиктор действительно
собирается (`--skip-load-check` отключает последнюю проверку на машине без ML-зависимостей).
Пакет кладётся в папку моделей до записи в базу, поэтому у зарегистрированной версии всегда
есть файлы. Вторую версию с тем же именем команда не примет.

## Дообучение на подтверждённых метках

Цепочка целиком: оператор подтверждает загрузку через API, задача встаёт в очередь, воркер
дообучает модель, гейт решает, лучше ли она, и при успехе версия включается.

```bash
uv run loadvision jobs start           # поставить задачу в очередь (то же делает API)
uv run loadvision-worker               # воркер: берёт задачи по одной, пока его не остановят
uv run loadvision-worker --once        # один проход и выход, для cron или проверки
uv run loadvision jobs list
```

```
job                                   status      queued                result                  gate
145ee47a-0861-442e-a662-4459a94abfd9  succeeded   2026-09-18T09:21:23Z  2026-09-18.1            passed
```

То же самое через API:

```bash
curl -s -X POST http://localhost:8000/api/v1/training-jobs \
  -H 'Content-Type: application/xml; charset=utf-8' \
  --data '<TrainingJobRequest xmlns="urn:loadvision:api:v1"><Mode>fast</Mode></TrainingJobRequest>'

curl -s http://localhost:8000/api/v1/training-jobs        # что делала очередь
curl -s http://localhost:8000/api/v1/models               # версии и их статусы
curl -s -X POST http://localhost:8000/api/v1/models/2026-09-18.1/activate
```

Одновременно выполняется одна задача: вторая `jobs start` вернёт `TRAINING_ALREADY_RUNNING`.
Пакет, который собрал прогон, воркер сам кладёт в папку моделей и регистрирует; если гейт
пропустил кандидата, версия включается, и её подхватит перезапуск сервиса.

Что настраивается в `configs/app.yaml`, секция `training`: конфиг эксперимента, папка базового
датасета, устройство (`cpu`, `mps`, `cuda`) и пауза между опросами очереди.

Воркеру нужны ML-зависимости (torch), поэтому он запускается из репозитория, а не из образа
api: тот собран без них. Отдельный образ воркера - задача 10.1.

## База данных

```bash
just db-up          # Postgres из compose.yaml
just db-migrate     # alembic upgrade head в базу из .env
just db-down
```

Новая миграция после правки `infrastructure/db/tables.py`:

```bash
uv run --env-file .env alembic revision --autogenerate --rev-id 0004 -m "что добавили"
```

Автогенерацию нужно прочитать глазами: имена ограничений мы задаём сами, а `CHECK` Alembic не
всегда видит.

## Проверки и тесты

```bash
scripts/check.sh          # ruff, формат, pyright, import-linter, быстрые тесты; то же в хуке
just test                 # только быстрые тесты
just test -k label        # выбранные тесты
just integration          # тесты на настоящем Postgres, каждый в своей временной базе
uv run --env-file .env pytest -m data    # проверки на реальном датасете, нужен data/
```

Маркеры: `unit`, `contract` (ответы по XSD), `api`, `ml`, `integration`, `e2e`, `data`. Без
`DATABASE_URL` интеграционные тесты пропускаются, а не падают.

## Docker

```bash
docker compose build api
docker compose up -d api      # то же самое делает just up
docker compose logs -f api
```

Образ собирается из `docker/api.Dockerfile`: сборка в образе uv, итоговый образ - `python-slim`
плюс виртуальное окружение без группы `ml`, поэтому в нём нет torch и scikit-learn. Пакет модели
на энкодерах в таком образе не поднимется: для него нужен образ с группой `ml`.

Подключение к базе внутри compose идёт через переменные `PG*` при пустом `DATABASE_URL`: так
пароль со знаком `@` или `/` не нужно экранировать внутри URL. Папка `./storage` подключена в
контейнер, поэтому пакеты моделей и фото остаются на хосте.

## Если что-то не работает

| Что видно | Почему | Что делать |
|---|---|---|
| `/health` отвечает 503 `MODEL_NOT_READY` | пакета модели нет или он не загрузился | `uv run loadvision model build-baseline`, затем `docker compose logs api` |
| `DATABASE_URL is not set` | нет `.env` или переменной в окружении | `cp .env.example .env` |
| `images folder not found: data/train/images` | нет данных хакатона | положить `data/train` и `data/test` |
| `cannot read the git commit` | команда запущена вне git-репозитория | запускать из клона; для записей замеров это не мешает, для пакета модели обязательно |
| `storage/models/baseline-v0 already exists` | версия уже собрана | другая `--version` или удалить папку осознанно |
| 415 `INVALID_CONTENT_TYPE` | не тот заголовок | `Content-Type: application/xml; charset=utf-8` |
| 400 `INVALID_ENCODING` | base64 с префиксом `data:image/...` | передавать только байты файла в base64 |
| порт 8000 занят | что-то уже слушает | остановить процесс или поменять порт в `compose.yaml` |
