# CPU-сборка (по умолчанию). GPU-вариант — Dockerfile.gpu (для сервера).
FROM python:3.12-slim

# ffmpeg/ffprobe — конвертация/нарезка аудио; libsndfile1 — для soundfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch CPU + onnxruntime CPU, затем общие зависимости (pyannote увидит torch установленным)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir onnxruntime \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
