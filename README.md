# diarization

Сервис диаризации речи («кто и когда говорил») + транскрипт с тайм-кодами.
Форк ASR-каркаса из [`transcriptions`](../transcriptions); диаризация добавлена сверху.

- **Стерео-звонки** (телефония, 1 спикер = 1 канал): диаризация по каналам, без ML-диаризатора.
- **Моно-совещания** (один поток, много спикеров): pyannote `speaker-diarization-community-1`.
- **ASR**: GigaAM v3 через onnx-asr с тайм-кодами (`.with_timestamps()`); слова выравниваются по спикерам.

> Исполнение — на сервере (GPU). С ноутбука сервис не запускается: код правится локально, разворачивается на `dedicated_server`.

## API

```
POST /v1/diarize        multipart, см. параметры ниже
POST /v1/test           multipart: первые N секунд файла через выбранный провайдер
POST /v1/models/unload  выгрузить локальные модели, вернуть видеопамять
GET  /v1/models         модели с разбивкой по провайдерам
GET  /health            + что сейчас загружено в память
```

Параметры `/v1/diarize`:

| Поле | Значения | По умолчанию | Назначение |
|---|---|---|---|
| `file` | аудио/видео | — | вход |
| `provider` | `local` / `mws` | из `ASR_PROVIDER_DEFAULT` | где считаем ASR |
| `speakers` | bool | `true` | `false` — просто расшифровка, pyannote не поднимается |
| `mode` | `auto` / `stereo` / `mono` | `auto` | способ диаризации |
| `model` | локально `gigaam`/`parakeet`, в MWS — из `MWS_MODELS` | по провайдеру | ASR |
| `response_format` | `json` / `text` | `json` | формат ответа |
| `merge_speakers` | bool | `true` | слить подряд идущие реплики одного спикера |
| `timestamps` | bool | `true` | показывать тайм-коды в тексте |
| `summary` | bool | `false` | саммари (LLM) |
| `follow_up` | bool | `false` | follow-up: открытые вопросы (LLM) |
| `todo` | bool | `false` | to-do: задачи/действия (LLM) |

`mode=auto`: 2+ независимых канала → стерео-режим; иначе моно + pyannote.
`summary`/`follow_up`/`todo` используют серверный LLM (`gpt-oss-120b`); в JSON приходят как поля `summary`/`follow_up`/`todo`, в `text` — секциями в конце.

Пример:
```bash
curl https://diarization.cloudsmasters.ru/v1/diarize \
  -H "Authorization: Bearer <DIARIZE_API_KEY>" \
  -F "file=@meeting.mp4" -F "mode=auto"
```
Ответ: `{"segments":[{"speaker","start","end","text"}], "text":"[mm:ss-mm:ss] SPEAKER: ..."}`.

## Где считаем: локально или в MWS

ASR-провайдер выбирается в интерфейсе и в API:

- **local** — модели onnx-asr на нашей карте. Отдают слова с тайм-кодами, поэтому
  спикеры выравниваются пословно — самый точный вариант.
- **mws** — облачный OpenAI-совместимый endpoint (`gigaam-v3`, `whisper-large-v3`).
  Тайм-коды по словам он не отдаёт, поэтому при `speakers=true` аудио режется по
  turn'ам pyannote и каждая реплика уходит отдельным запросом (пул из
  `REMOTE_CONCURRENCY` потоков). При `speakers=false` — один запрос на весь файл.

Диаризацию (`кто говорил`) умеет только pyannote, и он локальный: в MWS моделей
диаризации нет. Значит видеопамять полностью свободна лишь в режиме
**mws + speakers=false**.

**Локальные модели грузятся лениво.** При старте контейнера GPU не занимается
вообще; модель поднимается при первом запросе с `provider=local` и выгружается
после `LOCAL_IDLE_UNLOAD_S` секунд простоя либо по кнопке «Выгрузить модели»
(`POST /v1/models/unload`). Что сейчас в памяти — видно в `/health` и под формой
в интерфейсе.

**Кнопка «Тест»** в интерфейсе гоняет первые 30 секунд загруженного файла через
выбранную пару провайдер+модель и показывает время и распознанный текст. Это
проверяет весь путь целиком: ключ и сеть для MWS либо загрузку модели на GPU.

## Конфиг

См. `.env.example`. Обязательно: `HF_TOKEN` с принятой лицензией
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

## Деплой (на сервере)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.traefik.yml up -d --build
```
Требуется сеть `dedicated_server_default` (Traefik основного стека) и A-запись
`diarization.cloudsmasters.ru` на сервер.

## Структура

```
app/
  asr.py        # GigaAM/Parakeet + слова с тайм-кодами (форк transcribers.py)
  diarize.py    # pyannote: turn'ы + назначение спикера слову
  pipeline.py   # авто-режим (стерео по каналам / моно через pyannote), сборка реплик
  main.py       # FastAPI: /v1/diarize, /v1/models, /health, auth
```

## Дальше (план)
- База знаний: razdel-предложения из реплик + метаданные `{speaker,start,end,meeting_id}` →
  индексация в гибридный RAG ([`ai-search-regulations_v3`](../ai-search-regulations_v3), pgvector,
  отдельный индекс `conversations`) → поиск/QA с цитатами «спикер @ mm:ss».
