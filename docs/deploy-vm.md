# Запуск на ВМ организаторов с GPU

Машина, которую выдаёт бот командой `/vm_start`: Yandex Cloud, 16 vCPU, 64 ГиБ, одна GPU T4i
(определяется как NVIDIA L4, 24 ГБ), Ubuntu 22.04, CUDA 12.2, драйвер NVIDIA 535, системный
Python 3.10, пользователь `yc-user`, публичный IPv4. Диск сохраняется между остановками.

## Чем эта машина отличается

- **torch.** Сборка torch 2.14 с PyPI сделана под CUDA 13 и требует драйвер 580 и новее. На
  драйвере 535 она ставится, но GPU не видит. Поэтому на ВМ ставится группа зависимостей `cuda`:
  torch 2.14 под CUDA 12.6, который работает на драйверах 525 и новее. Ноутбуки и образы Docker
  по-прежнему получают группу `cpu`, так устроены группы в `pyproject.toml`.
- **Python.** Код требует 3.12, на машине 3.10. Ставить ничего не нужно: uv сам скачает 3.12.
- **Где что работает.** Сервис и воркер запускаются без Docker, так torch видит GPU без NVIDIA
  Container Toolkit. Postgres работает в Docker из нашего `compose.yaml`. Если нужно всё в
  контейнерах - есть второй путь, ниже.
- **Устройство выбирается само.** В `configs/app.yaml` у `model.device` и `training.device`
  стоит `auto`: модель считается на GPU, когда torch его видит, иначе на CPU. Сервис пишет
  выбор в лог строкой `the model runs on cuda`.

## Один раз

```bash
ssh yc-user@<IP>
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install rust-just                      # команда just
git clone <адрес репозитория> loadvision && cd loadvision
cp .env.example .env                           # пароль: в POSTGRES_PASSWORD и в DATABASE_URL

just gpu-sync                                  # torch 2.14 под CUDA 12.6, около 3 ГБ
echo 'export UV_NO_SYNC=1' >> ~/.bashrc && export UV_NO_SYNC=1
just gpu-check
```

`gpu-check` падает, если torch не видит GPU. Если видит, он печатает имя видеокарты, версию
`torch 2.14.0+cu126` и `matmul ok: True`.

**`UV_NO_SYNC=1` обязателен.** Обычный `uv run` и `uv sync` приводят окружение к группам по
умолчанию, то есть молча возвращают CPU-сборку torch. С этой переменной `uv run` работает с тем,
что поставил `just gpu-sync`. Рецепты `gpu-*` сами передают нужные группы и без неё.

### Postgres

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker "$USER" && newgrp docker
docker compose up -d postgres                  # образ с mirror.gcr.io, Docker Hub не нужен
uv run --env-file .env alembic upgrade head
```

### Модель и аккаунт оператора

Пакет модели передаётся целиком, как папка с `manifest.json`: дообученных весов нет на Hugging
Face, и `model fetch` их не скачает.

```bash
# с ноутбука: пакет, который уже зарегистрирован у тебя локально
scp -r storage/models/v2 yc-user@<IP>:loadvision/storage/handoff/

# на ВМ
uv run --env-file .env loadvision model register storage/handoff/v2
uv run --env-file .env loadvision model activate v2
uv run --env-file .env loadvision operators add <логин>
```

## Каждый запуск

```bash
tmux new -d -s api 'just gpu-serve'            # сервис на 0.0.0.0:8000
tmux new -d -s worker 'just gpu-worker'        # воркер дообучения
curl -s localhost:8000/health                  # modelVersion="v2"
tmux attach -t api                             # логи; выйти, не останавливая: Ctrl-b d
```

`tmux` держит процессы после отключения SSH. Остановить: `tmux kill-session -t api`.

## В Docker, если сервис должен работать сам по себе

Способ выше запускает сервис прямо на машине. Если нужно показать, что всё поднимается одной
командой и переживает перезагрузку, есть второй путь: те же образы, собранные с CUDA-сборкой
torch.

```bash
# один раз: доступ к карте из контейнеров
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all mirror.gcr.io/library/ubuntu:22.04 nvidia-smi   # карта видна

just up-gpu          # Postgres, миграции и сервис; образ около 5 ГБ, сборка долгая
just worker-gpu      # воркер дообучения на карте
```

Что тут важно:

- Образ параметризован: `TORCH_BUILD=cpu` по умолчанию, `compose.gpu.yaml` ставит `cuda`. Тег
  образа тоже разный (`loadvision-api` против `loadvision-api:cuda`), так что CPU-образ на
  ноутбуке и GPU-образ на ВМ не затирают друг друга.
- **Без `nvidia-container-toolkit` контейнер не упадёт, а тихо посчитает на процессоре**:
  `model.device` стоит `auto`. Проверять по логу сервиса - там должна быть строка
  `the model runs on cuda`, а не `on cpu`.
- Первый старт дольше обычного: шкала пакета на карте ещё не мерена, и замер идёт заново.
  Проверка здоровья ждёт три минуты вместо одной.

## Доступ снаружи

Страницы открываются по `http://<IP>:8000/`. Если адрес не отвечает, откройте входящий TCP 8000 в
группе безопасности ВМ в консоли Yandex Cloud. XML API открыт без пароля, страницы требуют вход.

## Замер скорости для презентации

```bash
uv run loadvision predict submission --package storage/models/v2 --device cuda --out runs/bench-gpu.csv
uv run loadvision predict submission --package storage/models/v2 --device cpu --out runs/bench-cpu.csv
```

Команда печатает строку `307 photos in ... s, ... ms per photo`: это прогноз всех тестовых фото
пачками по 8 (`--batch-size`), вместе с чтением файлов, поделённый на число фото. На GPU в это
время входит и первый прогрев CUDA. Нужна папка `data/` с датасетом.

## Если что-то не работает

| Что видно | Почему | Что делать |
|---|---|---|
| `torch does not see the GPU` | окружение пересобрано под CPU: `uv sync` или `uv run` без `UV_NO_SYNC=1` | `just gpu-sync`, затем `just gpu-check` |
| `Groups cpu and cuda are incompatible` | обе группы запрошены сразу | ставить через `just gpu-sync`, он убирает `cpu` |
| uv не может скачать Python 3.12 | нет доступа к GitHub | `uv python install 3.12 --mirror <адрес зеркала>`, затем `just gpu-sync` |
| пакета `docker-compose-v2` нет | старый список пакетов | `sudo apt-get update`, затем повторить установку |
