# Спецификации плана A «Loadvision» v1

Источник: план A «Loadvision» v1 от 15.09.2026, текст вставлен в чат 15.09. Архитектурный артефакт A (ссылка в `CONTRIBUTING.md`) Claude прочитать не может, поэтому этот файл – единственная копия деталей плана на стороне B.

Наши поправки помечены **[поправка, дата]**. Если поправка противоречит тексту плана A, действует поправка. Решения и контракты – в `notes.md`.

> **Это проектный документ, написанный до реализации.** Он описывает замысел, а не готовый
> сервис: часть модулей и команд отсюда мы так и не сделали, часть сделали иначе. Что в сервисе
> есть на самом деле – в корневом `README.md`, в `docs/architecture.md` и в `docs/running.md`.

## 1. Принципы

- **Одна технология, два входа:** исследование на GPU и дообучение в сервисе запускают один и тот же пайплайн.
- **Модель поставляется пакетом с манифестом.** Сервис не знает, что внутри пакета: он запрашивает активную версию и получает предиктор.
- **Слои и зависимости:**
  - `domain` – сущности и правила: перевозка, оценка, метка, версия модели, задача обучения, политика ручной проверки. Не импортирует FastAPI, SQLAlchemy, torch;
  - `application` – сценарии (оценить фото, получить перевозку, принять метку, запустить обучение, активировать версию); порты описаны через Protocol;
  - внешние слои (`api`, `infrastructure`, `ml`, `worker`, `cli`) реализуют порты и собирают зависимости, друг о друге знают только через `application`.
- Правило слоёв проверяет import-linter. **[поправка 16.09]** Добавлены три forbidden-контракта, см. `notes.md`.
- **Мера ООП:**
  - Protocol только там, где есть две реализации: боевая и фейковая или две боевые. Это репозитории, `ImageStorage`, `JobQueue`, `LoadPredictor`, рантайм модели (ONNX и torch). **[поправка 16.09]** Добавлен `UnitOfWork`;
  - остальное – обычные классы и dataclass;
  - точки расширения – реестр-словарь в `ml/registry.py`, не плагины;
  - зависимости собираются в одном месте на процесс: `api/app.py` и `worker/main.py`, без DI-фреймворка;
  - размер файла – ориентир 50–300 строк; больше значит, что у файла две ответственности.
- **Сознательно не делаем:**
  - микросервисы – один API-процесс и воркер;
  - Celery, Redis, Kafka – очередь задач в Postgres на `SKIP LOCKED`;
  - Kubernetes – Docker Compose, масштаб репликами API;
  - React SPA – Jinja2 и HTMX;
  - SOAP – XML поверх REST с XSD;
  - MLflow Model Registry – активная версия в Postgres;
  - DVC – хватает манифеста, хэшей и `uv.lock`;
  - плагины через entry points.

## 2. Целевая структура репозитория

```text
loadvision/
├─ CONTRIBUTING.md, justfile, pyproject.toml, compose.yaml, .gitlab-ci.yml
├─ configs/app.yaml, configs/experiments/          # один YAML = один эксперимент
├─ src/loadvision/
│  ├─ settings.py                                   # pydantic-settings: yaml + env
│  ├─ errors.py                                     # каталог ErrorCode: код, сообщение, подсказка, HTTP
│  ├─ observability/ logging.py, events.py, run_recorder.py
│  ├─ domain/ shipment.py, assessment.py, label.py, model_version.py, training_job.py, review_policy.py
│  ├─ application/ ports.py, assess_photo.py, get_shipment.py, submit_label.py, start_training.py,
│  │               activate_model.py, active_model.py, underload_report.py
│  ├─ imaging/ decoders.py, validator.py, quality.py, preprocessing.py
│  ├─ infrastructure/ db/(session, tables, migrations/, repositories/), storage/local_storage.py,
│  │                  jobs/postgres_queue.py
│  ├─ ml/ registry.py, data/, models/, training/, evaluation/, ensemble/, packaging/, inference/, pipeline/
│  ├─ api/ app.py, middleware.py, errors.py, xml/, xsd/loadvision-v1.xsd, routers/, ui/
│  ├─ worker/main.py
│  └─ cli/ main.py, commands: data, features, train, eval, model, predict, bench, runs
├─ tests/ unit/, contract/, integration/, ml/, api/, e2e/, quality/, fixtures/
├─ runs/                                            # в .gitignore
└─ docker/ api.Dockerfile, worker.Dockerfile
```

Статусы по схеме B на 16.09:
- в каркасе есть: `settings`, `errors`, `observability` (logging, run_recorder), `domain` (shipment, assessment, model_version), `application` (ports, assess_photo, get_shipment), `imaging` (validator, preprocessing), `infrastructure/db` (session, tables, репозитории shipments, assessments, model_versions), `ml` (registry, models, training, evaluation, packaging, inference/predictor, pipeline), `api` (app, middleware, errors, xml, xsd, routers assessments, shipments, health), `cli` с командами, тесты unit/contract/api/ml;
- Should: label, training_job, review_policy, submit_label, start_training, activate_model, active_model, quality, репозитории labels и jobs, postgres_queue, error_analysis, compare, ood, blender, promotion_gate, finetune_trainer, augmentations, onnx_export, batcher, routers labels, training_jobs и models, ui, worker;
- Could: `models/depth.py`, тесты integration, e2e, quality;
- Must: `db/migrations/` (появится с Alembic).

В схеме отмечены как «есть», но в репозитории на 16.09 отсутствуют: `infrastructure/storage/local_storage.py`, `ml/data/manifest.py`, `ml/data/splits.py`. Нет и `observability/events.py`.

## 3. XML API

Все тела запросов и ответов – `application/xml; charset=utf-8`, пространство имён `urn:loadvision:api:v1`. Схема отдаётся по `GET /api/v1/schema.xsd`.

| Метод | Путь | Что делает | Ответ |
|---|---|---|---|
| POST | `/api/v1/assessments` | оценить фото и привязать к перевозке | 201, при повторе 200 |
| GET | `/api/v1/shipments/{shipment_id}` | текущая оценка и история | 200, 404 |
| GET | `/api/v1/assessments/{assessment_id}` | одна оценка | 200, 404 |
| PUT | `/api/v1/assessments/{assessment_id}/label` | подтверждённая метка, повтор заменяет | 200 |
| POST | `/api/v1/training-jobs` | запустить дообучение в быстром режиме | 202; 409, если уже идёт |
| GET | `/api/v1/training-jobs/{job_id}` | статус и итог задачи | 200 |
| GET | `/api/v1/models` | версии, статусы, метрики, активная | 200 |
| POST | `/api/v1/models/{version}/activate` | ручная активация или откат | 200, 409 |
| GET | `/api/v1/reports/underloaded` | недогруз за период: from, to, threshold | 200 |
| GET | `/health`, `/metrics` | готовность и метрики Prometheus | 200; 503 без модели |

**[поправка 15.09]** Must/Should по этим методам – в `notes.md`; расхождение приоритетов – в `open-questions.md`.

### Передача фото и лимиты

- Фото передаётся в base64 внутри XML: один документ и одна XSD, привычно для корпоративных шин. Прирост размера на 33% при медианном файле ~240 КБ несущественен.
- Тело запроса до 20 МБ, декодированное фото до 15 МБ.
- Не больше 50 Мп, короткая сторона не меньше 256 px.
- Форматы JPEG, PNG, WebP. Формат определяется по магическим байтам, а не по атрибуту `mediaType`.
- HEIC получает 415 с подсказкой конвертировать в JPEG.

### Примеры (формат, значения условные)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<AssessmentRequest xmlns="urn:loadvision:api:v1">
  <ShipmentId>RP-2026-000123</ShipmentId>
  <Image mediaType="image/jpeg" encoding="base64">/9j/4AAQSkZJRgABAQ...</Image>
</AssessmentRequest>
```

```xml
<!-- 201 Created -->
<Assessment xmlns="urn:loadvision:api:v1" id="3f6c2a4e-7b1d-4c55-9a0e-5d2f8e41c7aa"
            shipmentId="RP-2026-000123" reused="false">
  <LoadPct>85.0</LoadPct>
  <Review needed="true">
    <Reason>low_confidence</Reason>
  </Review>
  <Signals uncertainty="7.4" oodScore="0.12" brightness="0.31" sharpness="0.78"/>
  <Image sha256="9b1d4c...e0" width="1440" height="1920"/>
  <Model version="2026-09-20.1"/>
  <ProcessingMs>184</ProcessingMs>
  <CreatedAt>2026-09-20T10:15:03Z</CreatedAt>
</Assessment>
```

```xml
<!-- GET /api/v1/shipments/RP-2026-000123 -->
<Shipment xmlns="urn:loadvision:api:v1" id="RP-2026-000123">
  <Current assessmentId="3f6c2a4e-..." loadPct="85.0" createdAt="2026-09-20T10:15:03Z"/>
  <History count="2">
    <Assessment id="3f6c2a4e-..." loadPct="85.0" modelVersion="2026-09-20.1" createdAt="2026-09-20T10:15:03Z"/>
    <Assessment id="a91b07d2-..." loadPct="40.0" modelVersion="2026-09-20.1" createdAt="2026-09-20T09:02:44Z"/>
  </History>
</Shipment>
```

```xml
<Error xmlns="urn:loadvision:api:v1" code="CORRUPTED_IMAGE" requestId="req_7Hc2kQ">
  <Message>Изображение не декодируется: файл обрезан или повреждён.</Message>
  <Hint>Отправьте исходный JPEG, PNG или WebP без изменений.</Hint>
</Error>
```

**[поправка 16.09]**
- `LoadPct` – `xs:decimal`, округление до одного знака только при сборке XML.
- `Review` и `Signals` необязательные.
- `DeclaredLoadPct` – предложено, не согласовано.

### Номер перевозки

- Это идентификатор рейса во внутренних системах, по нему потом ищут оценку. Формат в условии не задан, выбран свой.
- Всё, что не подходит под формат, получает `INVALID_SHIPMENT_ID`.
- Правило живёт в value object `ShipmentId` и повторяется в XSD. **[поправка 16.09]** Реализовано: `SHIPMENT_ID_PATTERN = "[A-Za-z0-9_-]{1,64}"` без якорей, проверка через `fullmatch`; в конфиге формата нет.

### Повторное обращение к перевозке

| Ситуация | Поведение | Статус |
|---|---|---|
| Новое фото, новая перевозка | создаём перевозку и оценку | 201 |
| Новое фото, известная перевозка | добавляем в историю, текущей становится новая оценка: пересняли после догрузки, значит актуальна последняя | 201 |
| То же фото (sha256), та же перевозка | ничего не пересчитываем, возвращаем сохранённую оценку с `reused="true"`; защита от сетевых ретраев | 200 |
| То же фото, другая перевозка | новая оценка, прогноз из кэша по паре (sha256, версия модели) | 201 |
| Модель сменилась после оценки | сохранённая оценка не меняется, в ней записана версия модели, которая её сделала | 200 |

### Ошибки

Все ошибки отдаются одним форматом. Коды живут в каталоге `errors.py`, им же пользуются пайплайн и логи.

| Код | HTTP | Когда |
|---|---|---|
| INVALID_CONTENT_TYPE | 415 | тело не `application/xml` |
| INVALID_XML | 400 | не парсится, не проходит XSD, содержит DTD |
| INVALID_SHIPMENT_ID | 400 | номер не соответствует формату |
| INVALID_ENCODING | 400 | битый base64 |
| PAYLOAD_TOO_LARGE | 413 | тело или фото больше лимита |
| UNSUPPORTED_FORMAT | 415 | магические байты не JPEG, PNG, WebP |
| CORRUPTED_IMAGE | 422 | не декодируется, обрезан, слишком маленький, слишком много пикселей |
| SHIPMENT_NOT_FOUND, ASSESSMENT_NOT_FOUND | 404 | нет такой записи |
| TRAINING_ALREADY_RUNNING | 409 | уже есть задача в статусе running |
| MODEL_NOT_READY | 503 | активная модель ещё не загружена |
| INTERNAL_ERROR | 500 | всё остальное; без трейсбека в ответе, с `requestId` для поиска в логах |

**[поправка 16.09]** Добавлен `MODEL_INFERENCE_FAILED` (500): падение модели, NaN или прогноз вне [0, 100].

### Безопасность XML и файлов

- **Парсер** создаётся явно: `lxml.etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)`. Документ с DOCTYPE отклоняется, `iterparse` не используется.
- **Порядок проверки:** сначала XSD, потом маппинг в модели pydantic-xml. `from_xml` пробрасывает аргументы в `etree.fromstring`, поэтому туда передаётся наш парсер.
- **Pillow:** лимит `MAX_IMAGE_PIXELS`; декодирование полное, а не только `verify()`, чтобы ловить обрезанные файлы.
- **Размер тела:** лимит в middleware до чтения. Base64 и сырые байты никогда не попадают в логи.
- **Ключ API** (опционально): заголовок `X-API-Key` из переменной окружения.

Что проверил A 15.09:
- в lxml с версии 5.0 обычный парсер по умолчанию не раскрывает внешние сущности, но `iterparse` и `ETCompatXMLParser` оставались уязвимыми до 6.1 – поэтому парсер задаём явно;
- `from_xml` в pydantic-xml передаёт дополнительные аргументы в `etree.fromstring`.

### Поток запросов и смена модели

- Один процесс uvicorn на контейнер, масштабирование репликами: иначе модель грузится в память N раз.
- Батчер `inference/batcher.py` собирает до 8 фото или ждёт до 20 мс. Инференс идёт в отдельном потоке и не блокирует event loop. Параметры в конфиге, значения подбираются нагрузочным тестом.
- `ActiveModel` раз в 10 секунд сверяет активную версию в БД, новый пакет грузит в фоне и атомарно подменяет ссылку. Запросы в полёте досчитываются на старой версии.

## 4. Хранение

| Таблица | Ключевые поля | Ограничения |
|---|---|---|
| images | sha256, media_type, width, height, size_bytes, path, created_at | PK по sha256 |
| shipments | id, created_at | id по формату ShipmentId |
| assessments | id, shipment_id, image_sha256, load_pct, uncertainty, ood_score, needs_review, review_reasons, model_version, processing_ms, request_id, created_at | UNIQUE (shipment_id, image_sha256); CHECK load_pct в [0, 100] |
| confirmed_labels | id, image_sha256, load_pct, source, author, created_at, updated_at | UNIQUE image_sha256; CHECK диапазона |
| model_versions | version, status, package_uri, manifest (jsonb), cv_mae, within_10, gate_report (jsonb), created_at, activated_at | частичный UNIQUE: одна строка со status = active |
| training_jobs | id, mode, status, run_id, params, result_version, error_code, created_at, started_at, finished_at | частичный UNIQUE: одна задача со status = running |

Инварианты «одна активная модель» и «одно обучение за раз» держит база, а не код: два воркера или гонка активаций физически не могут их нарушить.

**[поправка 15.09]** SQLAlchemy 2 **sync** и psycopg 3 вместо async и asyncpg. **[поправка 16.09]** `load_pct` – `double precision`; проверить тестом, что NaN не проходит CHECK.

## 5. Дообучение

Порядок:
1. Оператор подтверждает метку через `PUT .../label` или в UI.
2. `POST /training-jobs` создаёт задачу, воркер забирает её через `SKIP LOCKED`.
3. Пайплайн собирает датасет из исходного train и подтверждённых меток. Новые фото получают эмбеддинги, старые берутся из кэша.
4. Кросс-валидация по тем же правилам, голова переобучается, собирается пакет-кандидат.
5. PromotionGate сравнивает кандидата с активной моделью и пишет отчёт в `gate_report`.
6. При успехе версия активируется, API подхватывает её за 10 секунд; при провале статус `rejected` с причиной.
7. Откат – `POST /models/{version}/activate` на предыдущую версию.

Условия PromotionGate:
- **Старые данные не хуже:** OOF MAE кандидата на строках исходного train не хуже, чем у активной модели, с допуском из конфига.
- **Новые данные лучше:** OOF MAE кандидата на новых подтверждённых фото ниже, чем MAE активной модели на них же. Прогноз активной модели ничего не стоит: он уже сохранён в `assessments` в момент оценки.
- **Техническая проверка:** прогнозы конечны и в диапазоне, пакет загружается, p95 задержки выросла не больше чем на допуск.
- **[поправка 16.09]** Фолды старых строк не меняются. На новых фото нужен минимум N меток и допуск (значения – вопрос В4).

| Режим | Где | Что обучается | Для чего |
|---|---|---|---|
| fast | воркер, CPU | голова и ансамбль поверх замороженного backbone | регулярное дообучение, показ на демо |
| full | CLI, арендованный GPU | весь пайплайн с fine-tune | крупное обновление; итог регистрируется через `model register` и проходит тот же gate |

**[поправка 16.09, `CONTRIBUTING.md`]** Режим fast работает только на замороженных энкодерах. Эмбеддинги train от fine-tuned энкодера завышают OOF, и gate пропускал бы почти любого кандидата.

Воспроизводимость:
- манифест пакета содержит git-коммит, хэш данных, полный разрешённый конфиг, сиды, версии библиотек и железо обучения;
- окружение фиксируют `uv.lock` и digest Docker-образа;
- веса публикуются в GitLab Package Registry и скачиваются командой `loadvision model pull <version>`; вместе с весами DINOv3 кладётся текст лицензии;
- проверка перед сдачей: чистый клон, команды из README, тот же `submission.csv` по хэшу.

## 6. ML-решение по плану A

| Подход | Плюсы | Минусы | Роль |
|---|---|---|---|
| A. Замороженный backbone и голова | эксперимент за минуты, трудно переобучиться на 716 фото, эмбеддинги кэшируются, дообучение на CPU | потолок качества ниже fine-tune | фундамент |
| B. Fine-tune с LoRA | обычно лучший MAE на специфичном домене | нужен GPU, аккуратные аугментации и ансамбль по фолдам | усиление |
| C. Глубина и сегментация | физически ближе к смыслу метки, объяснимо на демо | нет разметки масок, нестабильно в тёмном кузове | доп. признаки |
| D. VLM zero-shot | не требует обучения | медленно, числа неточные | не берём |

Решения, которые двигают MAE:
- **Пропорции кадра:** не резать по центру в квадрат, потому что нижняя часть кадра с полом у двери решает «80 против 100». Ресайз по длинной стороне с паддингом или прямоугольный вход.
- **Аугментации:**
  - можно: горизонтальное отражение, яркость и гамма, размытие, JPEG-артефакты, поворот до нескольких градусов, имитация масок и рамок скриншотов;
  - нельзя: вертикальное отражение и кроп, отрезающий дверь.
- **Голова под MAE:** на OOF сравниваются регрессия с L1 и распределение по 21 бину (0, 5, ..., 100) с мягкими метками и медианой на выходе.
- **kNN-признак:** средняя метка ближайших train-соседей по эмбеддингу. **[поправка 16.09]** Соседей брать только из обучающих фолдов.
- **Постобработка:** клип в [0, 100]; округление к кратным 5 и притягивание к 100 – только если OOF подтверждает.
- **TTA:** горизонтальное отражение.

Валидация:
- StratifiedGroupKFold: 5 фолдов по `group_id` со стратификацией по бинам метки, 3 сида.
- Пары почти-дубликатов по эмбеддингам, которые не поймали группы, принудительно кладутся в один фолд.
- Метрики: MAE, доля ошибок ≤ 10 п.п., MAE по диапазонам 0–20, 20–80, 80–100, разброс MAE между фолдами.
- Срезы устойчивости: тип кузова в данных не размечен, поэтому срезы – скриншот или фото, тёмный или светлый кадр, кластеры сцен по эмбеддингам.
- Каждый запуск сохраняет `worst_oof.png`: контактный лист 30 худших фото с эталоном и прогнозом.

Сигналы доверия:

| Сигнал | Как считаем | Причина в ответе |
|---|---|---|
| uncertainty | разброс прогнозов членов ансамбля, п.п. | low_confidence |
| ood_score | расстояние до k ближайших train-эмбеддингов, порог – 99-й перцентиль train | unusual_scene |
| brightness | средняя яркость после EXIF-поворота | too_dark |
| sharpness | дисперсия лапласиана на уменьшенном кадре | blurry |

Решение о ручной проверке принимает `domain/review_policy.py`. Пороги лежат в конфиге и подбираются по OOF, чтобы флаг ловил большие ошибки, а не каждое второе фото.

Что уходит в сервис:
- модель в сервисе та же, что сформировала `submission.csv`;
- для голов на эмбеддингах усредняются 5 фолдовых голов;
- для fine-tune выбор по MAE и p95 задержки: обучить заново на всех данных с числом эпох из CV (один проход backbone) или усреднить 5 фолдовых моделей (пять проходов).

## 7. Пайплайн

```python
class Stage(Protocol):
    name: str
    version: str  # change resets the cache

    def inputs(self, ctx: RunContext) -> dict[str, str]: ...  # for the cache hash
    def run(self, ctx: RunContext) -> StageResult: ...
    def check(
        self, ctx: RunContext, result: StageResult
    ) -> None: ...  # postconditions, StageCheckError
```

- Каждый этап – отдельный класс в своём файле в `ml/pipeline/stages/`, отдельная CLI-команда и постусловия.
- Постусловия проверяются при каждом запуске, а не только в тестах.
- Кэш – по хэшу входов, конфига и версии этапа. Если упал пятый этап, перезапуск идёт с пятого.
- Runner умеет `--from-stage`, `--only`, `--no-cache`. Каждый запуск получает `run_id` и папку `runs/<run_id>/`.

| Этап | Команда | Вход -> выход | Постусловие |
|---|---|---|---|
| build_manifest | `data manifest` | фото и csv -> manifest.parquet | у каждого id есть файл, sha256 уникальны, метки конечны и в [0, 100] |
| find_duplicates | `data dedup` | эмбеддинги -> dedup_report.json | все пары выше порога перечислены, train–test отдельно |
| make_splits | `data split` | manifest -> folds.json | группа ровно в одном фолде, почти-дубли вместе, доли бинов по фолдам близки |
| extract_embeddings | `features embed` | manifest и backbone -> кэш | размерность совпадает, значения конечны, покрыты все sha256 |
| train_folds | `train folds` | фичи или фото и folds -> модели и oof.parquet | OOF покрывает каждую строку ровно один раз, прогнозы конечны |
| evaluate | `eval report` | oof -> metrics.json, worst_oof.png | метрики есть по всем диапазонам и фолдам |
| fit_ensemble | `ensemble fit` | несколько oof -> weights.json | веса ≥ 0, сумма 1, MAE ансамбля не хуже лучшего члена |
| package_model | `model package` | модели -> пакет | пакет грузится, прогноз на 8 фото совпадает с исходными моделями |
| predict | `predict submission` | пакет и test -> submission.csv | 307 строк, id совпадают с test.csv без повторов, значения конечны и в [0, 100], заголовок ровно `image_id,load_pct`, файл меньше 2 МБ |
| benchmark | `bench run` | пакет -> bench.json | записаны CPU, GPU, RAM, батч, p50, p95, RPS |

## 8. Логи и запуски

Формат событий:
- **Строка:** один JSON на строку через structlog; локально тот же поток рендерится в читаемую консоль.
- **Обязательные поля:** `ts`, `level`, `event`, `module`. Когда применимо, ещё `run_id`, `stage`, `request_id`, `job_id`, `duration_ms`.
- **Имена событий** – стабильные строки из каталога `observability/events.py`: `stage.started`, `stage.finished`, `stage.failed`, `request.finished`, `model.activated`, `gate.rejected`. Тест проверяет, что код не пишет событий вне каталога.
- **Ошибки** несут `error_code` из того же каталога, что и API, плюс тип исключения и сокращённый трейсбек.
- **Уровни:** INFO – границы этапов и запросов; DEBUG – детали; WARNING – деградация (torch вместо ONNX, тёмный кадр); ERROR – провал с кодом.
- **Фильтр** вырезает base64, сырые байты, ключи и строки подключения.

```json
{"ts":"2026-09-18T14:02:11.482Z","level":"error","event":"stage.failed","module":"ml.pipeline.runner",
 "run_id":"run_20260918_140155_a3f","stage":"extract_embeddings","error_code":"NONFINITE_EMBEDDINGS",
 "rows_bad":3,"duration_ms":41230}
```

Папка запуска `runs/<run_id>/`:
- `summary.json` – первое, что читает агент: статусы этапов, метрики, ошибка;
- `run.jsonl` – все события запуска;
- `config.resolved.yaml`;
- `env.json` – git-коммит, хэш `uv.lock`, версии библиотек, CPU, GPU, RAM;
- `metrics.json`, `worst_oof.png`, `artifacts/`.

`summary.json` при сбое содержит `status`, `failed_stage`, `error_code`, `hint`, список `stages` (name, status, cached, счётчики, duration_ms) и `rerun` – готовую команду перезапуска. Подсказку и команду перезапуска берут из каталога ошибок и runner.

Цикл правки:
- `just run exp=...` печатает одну строку итога и путь к `summary.json`; флаг `--json` даёт машинный вывод;
- при сбое читается `summary.json`, `run.jsonl` фильтруется по этапу, перезапуск идёт с `--from-stage`;
- `loadvision eval compare <run_a> <run_b>` печатает разницу по MAE в целом и по диапазонам, по within_10 и p95;
- ошибка в API: клиент получает `requestId` в XML, по нему в логах находится весь путь запроса.

## 9. Тестирование

Проверки на двух уровнях: постусловия встроены в этапы и выполняются при каждом запуске, в том числе у воркера в проде; тесты проверяют поведение так, как им пользуется реальный вызывающий код.

| Что | Какие проверки | Маркер |
|---|---|---|
| domain | границы LoadPercent, формат ShipmentId, правила ReviewPolicy на таблице случаев, переходы статусов версии и задачи | unit |
| imaging | фикстуры: обрезанный JPEG, PNG с расширением .jpg, HEIC, 0 байт, 1×1, pixel bomb, CMYK, EXIF-поворот, скриншот с рамкой; фаззинг hypothesis – случайные байты дают только известные коды, никогда 500 | unit |
| preprocessing | путь обучения и путь сервиса дают одинаковый тензор на одном фото; эталонный хэш тензора | unit |
| XML-контракт | каждый ответ в тестах валидируется XSD; XXE, billion laughs, внешний DTD и чужой namespace дают 400 и ни одного обращения к файлам | contract |
| API | все строки таблицы повторов, все коды ошибок, X-Request-ID в ответе и логе | api |
| БД и очередь | реальный Postgres в testcontainers; миграции вверх и вниз; два воркера не берут одну задачу; вторая активная версия невозможна | integration |
| Этапы пайплайна | каждый этап на синтетическом мини-датасете с энкодером `tiny_test` из реестра; сломанный вход ловит постусловие | ml |
| Утечки | ни одна группа и ни одна пара почти-дублей не разнесена по фолдам | ml |
| Паритеты модели | ONNX и torch совпадают в пределах 0,01 п.п.; CLI predict и API дают одинаковые числа на 20 фото | ml |
| Детерминизм | одинаковый сид в режиме fast даёт идентичные OOF | ml |
| PromotionGate | хуже -> отказ, лучше -> принято, NaN -> отказ, рост задержки сверх допуска -> отказ | unit |
| Качество | OOF MAE не хуже `tests/quality/expected.json` с допуском; перед тегом и сменой модели | data |
| E2E | compose: фото, чтение по перевозке, метка, задача обучения, новая активная версия, откат | e2e |
| Нагрузка | k6: p95 и RPS при заданной конкурентности | вручную |

CI в плане A: стадии static (ruff, pyright, import-linter; 1 мин), unit (unit, contract, api на фейковых портах; 1 мин), integration (3 мин), ml-smoke (3 мин), build (5 мин), release вручную (e2e, quality, публикация весов). Локально: `just check`, `just smoke`, `just e2e`. **[16.09]** Раннеров в GitLab хакатона нет: проверки идут в pre-push хуке через `scripts/check.sh`.

## 10. Расширяемость

| Что добавить | Что меняем | Какой тест |
|---|---|---|
| Новый backbone | запись в `ml/registry.py` и YAML эксперимента | ml-smoke подхватывает все записи реестра |
| Новая голова или loss | `models/heads.py` или `training/losses.py`, запись в реестре | этап train_folds на синтетике |
| Новый этап пайплайна | файл в `pipeline/stages/` по протоколу Stage, имя в конфиге пайплайна | smoke этапа и проверка постусловия на сломанном входе |
| Новый runtime, например TensorRT | класс runtime, поле runtime в манифесте | паритет с torch |
| Новый формат фото, например HEIC | декодер в `imaging/decoders.py` и фикстура | фикстура и фаззинг |
| Новый сигнал доверия | функция в `imaging/quality.py` или `inference/`, правило в `review_policy.py` | таблица случаев ReviewPolicy |
| Новый отчёт | сценарий в `application/`, роутер, элемент XSD | контракт XSD и api-тест |
| Хранилище S3 | реализация порта ImageStorage, выбор в settings | общий набор тестов порта на обе реализации |
| Несовместимое изменение API | роутеры v2 и XSD с namespace v2, v1 не трогаем | контракт обеих версий |

## 11. Стек плана A и что изменилось

| Слой | Выбор A | Изменения |
|---|---|---|
| Основа | Python 3.12, uv, ruff, pyright, import-linter, just | pyright как `pyright[nodejs]`, `default-groups = "all"` (16.09) |
| Backbone | DINOv3 ViT-B/16 из timm | DINOv3-B и DINOv2-B параллельно; PE-Spatial-B и SigLIP 2-B – предложено |
| Второй backbone | SigLIP2 | лицензия Apache-2.0 проверена 15.09 |
| Глубина | Depth Anything 3: DA3MONO-LARGE или DA3METRIC-LARGE (Apache-2.0) | Could |
| Обучение | PyTorch 2, Lightning, peft | – |
| Эксперименты | MLflow 3, backend `sqlite:///mlflow.db` | источник правды – `runs/`; MLflow – Could |
| Инференс | ONNX Runtime, запасной путь torch | – |
| API | FastAPI, pydantic-xml с lxml | – |
| БД | PostgreSQL, SQLAlchemy 2 async, asyncpg, Alembic | sync SQLAlchemy 2 и psycopg 3 |
| UI | Jinja2, HTMX | htmx 2.x локальным файлом |
| Наблюдаемость | structlog, prometheus-client | `/metrics` – Could |
| Тесты | pytest, hypothesis, httpx, testcontainers, k6 | hypothesis и testcontainers в CI – Could |
| Поставка | Docker Compose, GitLab CI, GitLab Package Registry | раннеров нет; лимиты Registry – вопрос В3 |

Ещё проверил A 15.09:
- официальные веса DINOv3 от Meta на HF закрыты гейтом, одобрение может занять дни; зеркало в timm без гейта;
- файловый backend MLflow объявлен устаревшим в 3.6, с 3.7 по умолчанию SQLite;
- экспорт DINOv3 в ONNX делают, но RoPE – зона риска, поэтому есть запасной путь на torch.

## 12. Контракты первого дня в плане A

```python
class LoadPredictor(Protocol):
    @property
    def version(self) -> str: ...
    def predict(self, images: Sequence[PreparedImage]) -> list[Prediction]: ...


@dataclass(frozen=True, slots=True)
class Prediction:
    load_pct: float  # [0, 100]
    uncertainty: float  # п.п.
    ood_score: float
```

**[поправка 16.09]** `load_pct: LoadPercent`; `uncertainty` и `ood_score` типа `float | None`; `PreparedImage` живёт в `imaging/`.

Пример манифеста пакета (формат):

```json
{
  "version": "2026-09-20.1",
  "created_at": "2026-09-20T08:40:00Z",
  "git_commit": "...",
  "data_hash": "sha256:...",
  "preprocessing": {"long_side": 512, "pad": "constant", "crop_screenshot_borders": true},
  "members": [
    {"backbone": "dinov3_vitb16", "runtime": "onnx", "weights": "m1/backbone.onnx", "head": "m1/head.npz", "weight": 0.6},
    {"backbone": "siglip2_b16", "runtime": "onnx", "weights": "m2/backbone.onnx", "head": "m2/head.npz", "weight": 0.4}
  ],
  "cv": {"mae": null, "within_10": null, "mae_by_range": {}},
  "ood": {"k": 10, "threshold": null},
  "licenses": ["licenses/DINOv3.md"]
}
```

Остальные контракты дня 1 по плану A:
- `imaging/preprocessing.py` – единственный путь от байтов к тензору;
- `api/xsd/loadvision-v1.xsd` и примеры XML в `tests/fixtures`;
- первая миграция Alembic;
- каталоги `errors.py` и `observability/events.py`;
- ModelPackage v0 – обёрнутый Ridge из baseline, чтобы платформа с первого дня работала с настоящей моделью через настоящий порт.

## 13. Риски из плана A

| Риск | Как заметим | Что делаем |
|---|---|---|
| Путаница 80 и 100 | MAE на 80–100 заметно выше остальных диапазонов | голова по бинам, признаки глубины, аугментации без обрезки двери |
| Переобучение на 716 фото | большой разброс MAE между фолдами и сидами | замороженный backbone первым, LoRA вместо полной разморозки, ансамбль |
| Погоня за public | решение принято по одному сабмиту | решаем по OOF, public – только проверка на грубую ошибку |
| Экспорт DINOv3 в ONNX | паритет с torch не сходится | runtime torch, решение по бенчмарку на день 3 |
| Аренда GPU срывается | нет машины к дню 2 | бронь на день 1; подход A считается и на CPU |
| Лицензии весов | вопрос жюри или заказчика | лицензии в пакете, запасной ансамбль без DINOv3 |
| Разные версии в материалах | числа в слайдах и в `/models` расходятся | один тег, версия читается из манифеста везде |
| Кузова 5, 10, 20 т без разметки типа | MAE сильно различается по кластерам сцен | срезы в отчёте CV, аугментации масштаба, флаг unusual_scene |
| Интеграция в последний день | API впервые видит финальную модель на день 6 | пакет v0 с первого дня, v1 на день 3, дальше меняются только веса |
