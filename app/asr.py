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
# GigaAM обучалась на записях до 30 с, а .transcribe официально держит до 25 с:
# рабочий диапазон сегмента в апстриме — 15–22 с, 30 с там аварийный потолок.
# Поэтому режем короче и по паузам (см. speech_chunks), а не фиксированным окном.
WINDOW_S = float(os.getenv("ASR_WINDOW_S", "22.0"))
CHUNK_MIN_S = float(os.getenv("ASR_CHUNK_MIN_S", "15.0"))
CHUNK_HARD_S = float(os.getenv("ASR_CHUNK_HARD_S", "30.0"))
MIN_CHUNK_S = 0.3  # окна короче — не подаём в ONNX (падает STFT)
MWS_MAX_UPLOAD_BYTES = int(float(os.getenv("MWS_MAX_UPLOAD_MB", "24")) * 1048576)


@dataclass
class Word:
    start: float
    end: float
    text: str
    conf: float | None = None  # уверенность ASR: min по токенам слова, 0..1


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


@contextmanager
def as_mp3(path: str, bitrate: str = "32k"):
    """Сжать в моно-mp3: облачные ASR ограничивают размер загрузки."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
             "-c:a", "libmp3lame", "-b:a", bitrate, "-y", tmp.name],
            check=True, capture_output=True,
        )
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def tokens_to_words(tokens, timestamps, logprobs=None) -> list[Word]:
    """Склеить subword-токены в слова. Граница слова — токен с ведущим пробелом/▁.

    logprobs (если модель их отдаёт) сворачиваются в уверенность слова: берём
    минимум по его токенам — слово настолько надёжно, насколько его слабое место.
    """
    import math

    words: list[Word] = []
    cur, start, end = "", None, None
    cur_lp: list[float] = []
    lps = list(logprobs) if logprobs is not None else None

    def flush():
        if cur.strip():
            conf = math.exp(min(cur_lp)) if cur_lp else None
            words.append(Word(start, end, cur.strip(), conf))

    for i, (tok, ts) in enumerate(zip(tokens, timestamps)):
        lp = lps[i] if lps is not None and i < len(lps) else None
        boundary = tok.startswith(" ") or tok.startswith("▁")
        clean = tok.replace("▁", " ")
        if boundary or start is None:
            flush()
            cur, start, end = clean, ts, ts
            cur_lp = [lp] if lp is not None else []
        else:
            cur += clean
            end = ts
            if lp is not None:
                cur_lp.append(lp)
    flush()
    return words


def speech_chunks(samples, sr: int, window_s: float = WINDOW_S,
                  min_s: float = CHUNK_MIN_S, hard_s: float = CHUNK_HARD_S) -> list[tuple[int, int]]:
    """Границы кусков для ASR: режем по паузам в диапазоне [min_s, window_s].

    Фиксированное окно рвёт слово на стыке (отсюда «Крост» → «Крос»+«т»).
    Ищем самую тихую точку в хвосте окна и режем там; если тишины нет —
    режем принудительно на hard_s (апстримный strict-лимит модели).
    """
    import numpy as np

    n = len(samples)
    if n <= int(window_s * sr):
        return [(0, n)]

    mono = samples if samples.ndim == 1 else samples.mean(axis=1)
    win = max(1, int(0.02 * sr))  # окно энергии 20 мс
    out: list[tuple[int, int]] = []
    pos = 0
    while pos < n:
        target = pos + int(window_s * sr)
        if target >= n:
            out.append((pos, n))
            break
        lo = pos + int(min_s * sr)                       # раньше не режем
        hi = min(n, pos + int(hard_s * sr))              # позже — нельзя
        seg = mono[lo:hi]
        if len(seg) < win * 2:
            out.append((pos, hi))
            pos = hi
            continue
        # энергия по 20-мс окнам; самая тихая точка — стык фраз
        k = len(seg) // win
        energy = np.abs(seg[:k * win].reshape(k, win)).mean(axis=1)
        cut = lo + int(energy.argmin()) * win + win // 2
        cut = max(lo, min(cut, hi))
        out.append((pos, cut))
        pos = cut
    return out


class Asr:
    """GigaAM/Parakeet через onnx-asr; отдаёт слова с тайм-кодами."""

    supports_words = True

    def __init__(self, model_name: str = "gigaam-v3-e2e-rnnt"):
        import onnx_asr
        self._ts = onnx_asr.load_model(model_name).with_timestamps()

    def words(self, wav_path: str, window_s: float = WINDOW_S) -> list[Word]:
        """Распознать моно-16к WAV, вернуть слова с тайм-кодами и уверенностью.

        Режем не фиксированным окном, а по паузам (speech_chunks): модель обучена
        на сегментах до 30 с, и разрыв слова на стыке портит распознавание.
        """
        samples, sr = sf.read(wav_path, dtype="float32")
        out: list[Word] = []
        min_chunk = int(MIN_CHUNK_S * sr)
        with tempfile.TemporaryDirectory() as d:
            for i, j in speech_chunks(samples, sr, window_s):
                chunk = samples[i:j]
                # огрызок в пару миллисекунд — на таком ONNX-STFT падает
                if len(chunk) < min_chunk:
                    continue
                off = i / sr
                cpath = os.path.join(d, f"c{i}.wav")
                sf.write(cpath, chunk, sr)
                res = self._ts.recognize(cpath)
                if not getattr(res, "tokens", None) or not getattr(res, "timestamps", None):
                    continue
                for w in tokens_to_words(res.tokens, res.timestamps,
                                         getattr(res, "logprobs", None)):
                    out.append(Word(w.start + off, w.end + off, w.text, w.conf))
        return out

    def transcribe(self, wav_path: str) -> str:
        """Сплошной текст без разметки по спикерам (склейка слов)."""
        return " ".join(w.text for w in self.words(wav_path))


class MwsAsr:
    """Облачный ASR MWS (OpenAI-совместимый /audio/transcriptions).

    Тайм-коды по словам не отдаёт ни одна из его моделей (gigaam-v3 возвращает
    один сегмент на весь файл), поэтому выравнивание по спикерам делается не
    здесь, а в pipeline: аудио режется по turn'ам pyannote и каждый кусок
    распознаётся отдельным запросом.
    """

    supports_words = False

    def __init__(self, model_name: str, base_url: str, api_key: str, timeout: float = 600.0):
        from openai import OpenAI
        if not api_key:
            raise RuntimeError("MWS_API_KEY не задан")
        self.model_name = model_name
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def transcribe(self, wav_path: str) -> str:
        # 16-кГц WAV на час записи — это ~115 МБ, столько облако не принимает
        # («Payload Too Large»), поэтому перед отправкой всегда жмём в mp3.
        with as_mp3(wav_path) as path:
            size = os.path.getsize(path)
            if size > MWS_MAX_UPLOAD_BYTES:
                raise RuntimeError(
                    f"файл {size // 1048576} МБ больше лимита облака "
                    f"({MWS_MAX_UPLOAD_BYTES // 1048576} МБ) — используйте локальный провайдер "
                    f"или разбейте запись"
                )
            with open(path, "rb") as f:
                r = self._client.audio.transcriptions.create(model=self.model_name, file=f)
        return (getattr(r, "text", "") or "").strip()

    def words(self, wav_path: str, window_s: float = WINDOW_S) -> list[Word]:
        raise NotImplementedError("MWS не отдаёт тайм-коды по словам")
