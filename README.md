# Loadvision

Сервис оценки загрузки грузового отсека по фотографии: прогноз 0-100%, привязка к номеру перевозки, XML API и дообучение на подтверждённой разметке.

Статус: каркас проекта. Разделы про API, обучение, прогноз, метрики и ограничения появятся по мере реализации.

## Требования

- [uv](https://docs.astral.sh/uv/) 0.12+
- Python 3.12 (uv установит сам)
- [just](https://github.com/casey/just) для коротких команд, необязательно
- Docker для локального Postgres

## Установка

```bash
uv sync
cp .env.example .env
```

Данные хакатона положите в `data/train` и `data/test`, они не коммитятся.

## Проверки

Все проверки (ruff, pyright, import-linter, быстрые тесты) собраны в одном скрипте:

```bash
scripts/check.sh    # или just check
```

Один раз после клонирования включите хук, который запускает этот скрипт перед каждым `git push` и отменяет пуш, если проверки красные:

```bash
git config core.hooksPath .githooks    # или just hooks
```

## CI

На GitLab хакатона нет раннеров, поэтому пайплайн не создаётся и проверки идут локально в хуке. Когда раннер появится, в Settings -> CI/CD -> Variables добавьте переменную `RUN_CI` со значением `true`: CI запустит тот же `scripts/check.sh`.

## Структура

```
configs/            настройки сервиса и эксперименты
src/loadvision/
  domain/           сущности и правила, без фреймворков
  application/      сценарии и порты
  infrastructure/   Postgres, файлы
  ml/               данные, обучение, инференс, пайплайн
  api/              FastAPI и XML
  imaging/          проверка и подготовка фото
  observability/    логи и записи запусков
  cli/              команды
tests/              unit, contract, api, ml
baseline/           стартовое решение организаторов, лежит локально, не коммитится
```

Правило слоёв проверяет import-linter: `api | cli` -> `infrastructure | ml` -> `application` -> `domain`.
