"""ASR с тайм-кодами. Форк из transcriptions/app/transcribers.py.

Отличие от исходника: помимо текста умеем отдавать СЛОВА С ТАЙМ-КОДАМИ
(GigaAM/Parakeet через onnx-asr `.with_timestamps()`), что нужно для выравнивания
диаризации. Whisper тайм-коды через onnx-asr не отдаёт — для диаризации не годится.

Длинное аудио распознаём по окнам (модель целиком длинный файл не тянет),
смещая тайм-коды на начало окна.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

import soundfile as sf

SR = 16000
WINDOW_S = 30.0


@dataclass
class Word:
    start: float
    end: float
    text: str


def has_audio_stream(path: str) -> bool:
    """True, если в файле есть хотя бы одна аудиодорожка (видео тоже подходит)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def channel_count(path: str) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return int((r.stdout.strip() or "1"))


@contextmanager
def as_wav(path: str, channel: int | None = None):
    """Привести к 16 кГц моно WAV. channel=None — даунмикс; channel=i — выделить канал i."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    af = ["-ac", "1"] if channel is None else ["-af", f"pan=mono|c0=c{channel}"]
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-ar", str(SR), *af, "-y", tmp.name],
            check=True, capture_output=True,
        )
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def tokens_to_words(tokens, timestamps) -> list[Word]:
    """Склеить subword-токены в слова. Граница слова — токен с ведущим пробелом/▁."""
    words: list[Word] = []
    cur, start, end = "", None, None
    for tok, ts in zip(tokens, timestamps):
        boundary = tok.startswith(" ") or tok.startswith("▁")
        clean = tok.replace("▁", " ")
        if boundary or start is None:
            if cur.strip():
                words.append(Word(start, end, cur.strip()))
            cur, start, end = clean, ts, ts
        else:
            cur += clean
            end = ts
    if cur.strip():
        words.append(Word(start, end, cur.strip()))
    return words


class Asr:
    """GigaAM/Parakeet через onnx-asr; отдаёт слова с тайм-кодами."""

    def __init__(self, model_name: str = "gigaam-v3-e2e-rnnt"):
        import onnx_asr
        self._ts = onnx_asr.load_model(model_name).with_timestamps()

    def words(self, wav_path: str, window_s: float = WINDOW_S) -> list[Word]:
        """Распознать моно-16к WAV по окнам, вернуть слова с глобальными тайм-кодами."""
        samples, sr = sf.read(wav_path, dtype="float32")
        out: list[Word] = []
        step = int(window_s * sr)
        with tempfile.TemporaryDirectory() as d:
            for i in range(0, len(samples), step):
                off = i / sr
                cpath = os.path.join(d, f"c{i}.wav")
                sf.write(cpath, samples[i:i + step], sr)
                res = self._ts.recognize(cpath)
                if not getattr(res, "tokens", None) or not getattr(res, "timestamps", None):
                    continue
                for w in tokens_to_words(res.tokens, res.timestamps):
                    out.append(Word(w.start + off, w.end + off, w.text))
        return out
