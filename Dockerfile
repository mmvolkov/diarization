# CPU-сборка (по умолчанию). GPU-вариант — Dockerfile.gpu (для сервера).
FROM python:3.12-slim

# ffmpeg/ffprobe — конвертация/нарезка аудио; libsndfile1 — для soundfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Тяжёлые слои отдельно и ДО requirements.txt (кеш не сбрасывается мелкими зависимостями).
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir onnxruntime
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
