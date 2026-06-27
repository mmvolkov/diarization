# diarization

Сервис диаризации речи («кто и когда говорил») + транскрипт с тайм-кодами.
Форк ASR-каркаса из [`transcriptions`](../transcriptions); диаризация добавлена сверху.

- **Стерео-звонки** (телефония, 1 спикер = 1 канал): диаризация по каналам, без ML-диаризатора.
- **Моно-совещания** (один поток, много спикеров): pyannote `speaker-diarization-community-1`.
- **ASR**: GigaAM v3 через onnx-asr с тайм-кодами (`.with_timestamps()`); слова выравниваются по спикерам.

> Исполнение — на сервере (GPU). С ноутбука сервис не запускается: код правится локально, разворачивается на `dedicated_server`.

## API

```
POST /v1/diarize    multipart: file, model(gigaam|parakeet), mode(auto|stereo|mono), response_format(json|text)
GET  /v1/models
GET  /health
```

`mode=auto`: 2+ независимых канала → стерео-режим; иначе моно + pyannote.

Пример:
```bash
curl https://diarization.cloudsmasters.ru/v1/diarize \
  -H "Authorization: Bearer <DIARIZE_API_KEY>" \
  -F "file=@meeting.mp4" -F "mode=auto"
```
Ответ: `{"segments":[{"speaker","start","end","text"}], "text":"[mm:ss-mm:ss] SPEAKER: ..."}`.

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
