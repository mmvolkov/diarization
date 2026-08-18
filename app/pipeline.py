"""Оркестрация диаризации.

Два режима:
- stereo (телефония): спикеры раздельно по каналам → ASR на каждом канале, без pyannote.
- mono (совещания): один поток → pyannote даёт «кто/когда», ASR даёт слова,
  каждое слово получает спикера по перекрытию с turn'ом.

auto: 2+ канала и они независимы (низкая корреляция L/R) → stereo; иначе → mono.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from app import asr, diarize
from app.asr import Word, as_wav

GAP = 1.0  # сек: пауза внутри реплики одного спикера
MIN_TURN_S = 0.4  # короче — не отправляем в облачный ASR (вздохи, поддакивания)
TURN_PAD_S = 0.25  # подушка по краям реплики при нарезке для облачного ASR
REMOTE_CONCURRENCY = int(os.getenv("REMOTE_CONCURRENCY", "8"))


def merge_consecutive(utts: list[dict]) -> list[dict]:
    """Слить подряд идущие реплики одного спикера в один блок (постобработка)."""
    out: list[dict] = []
    for u in utts:
        if out and out[-1]["speaker"] == u["speaker"]:
            out[-1]["text"] += " " + u["text"]
            out[-1]["end"] = u["end"]
        else:
            out.append(dict(u))
    return out


def _group(words: list[Word], speakers) -> list[dict]:
    """Склеить слова в реплики. speakers — спикер для каждого слова, по порядку."""
    utts: list[dict] = []
    for w, spk in zip(words, speakers):
        if utts and utts[-1]["speaker"] == spk and w.start - utts[-1]["end"] <= GAP:
            utts[-1]["text"] += " " + w.text
            utts[-1]["end"] = w.end
        else:
            utts.append({"speaker": spk, "start": w.start, "end": w.end, "text": w.text})
    return utts


def _channels_independent(src: str, probe_s: int = 120) -> bool:
    """Стерео с раздельными спикерами? Низкая корреляция L/R → раздельные; ~1 → микс."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-t", str(probe_s), "-i", src,
             "-ar", "16000", "-ac", "2", "-y", tmp.name],
            check=True, capture_output=True,
        )
        data, _ = sf.read(tmp.name, dtype="float32")
        if data.ndim < 2 or data.shape[1] < 2:
            return False
        l, r = data[:, 0], data[:, 1]
        if l.std() < 1e-6 or r.std() < 1e-6:
            return False
        corr = float(np.corrcoef(l, r)[0, 1])
        return corr < 0.9
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def _turn_texts(wav: str, turns, asr_obj) -> list[dict]:
    """Реплики через провайдера без тайм-кодов (MWS): режем wav по turn'ам pyannote.

    Каждый кусок — отдельный запрос, поэтому шлём пулом потоков. Слишком короткие
    turn'ы (вздохи, поддакивания) пропускаем: запрос стоит дороже пользы.
    """
    from concurrent.futures import ThreadPoolExecutor

    samples, sr = sf.read(wav, dtype="float32")
    jobs = [(i, ts, te, spk) for i, (ts, te, spk) in enumerate(turns) if te - ts >= MIN_TURN_S]

    def run(job):
        i, ts, te, spk = job
        # подушка по краям: на точной границе turn'а срезается первый/последний
        # звук («прикрепил» → «крепил»), поэтому режем чуть шире, а тайм-коды
        # оставляем исходные
        a = max(0, int((ts - TURN_PAD_S) * sr))
        b = min(len(samples), int((te + TURN_PAD_S) * sr))
        chunk = samples[a:b]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, f"t{i}.wav")
            sf.write(p, chunk, sr)
            text = asr_obj.transcribe(p)
        return {"speaker": spk, "start": ts, "end": te, "text": text.strip()}

    with ThreadPoolExecutor(max_workers=REMOTE_CONCURRENCY) as pool:
        utts = list(pool.map(run, jobs))
    return [u for u in utts if u["text"]]


def _plain_text(path: str, asr_obj) -> list[dict]:
    """Без диаризации: один запрос на весь файл, одна «реплика» без спикера."""
    with as_wav(path) as wav:
        text = asr_obj.transcribe(wav)
        dur = float(sf.info(wav).duration)
    return [{"speaker": "", "start": 0.0, "end": dur, "text": text.strip()}]


def diarize_file(path: str, asr_obj, mode: str = "auto", speakers: bool = True,
                 num_speakers: int | None = None, min_speakers: int | None = None,
                 max_speakers: int | None = None, exclusive: bool = True) -> list[dict]:
    if not speakers:
        return _plain_text(path, asr_obj)

    n = asr.channel_count(path)
    if mode == "stereo":
        use_channels = n >= 2
    elif mode == "mono":
        use_channels = False
    else:  # auto
        use_channels = n >= 2 and _channels_independent(path)

    if use_channels:
        utts: list[dict] = []
        for ch in range(n):
            spk = f"Канал {ch + 1}"
            with as_wav(path, channel=ch) as wav:
                if asr_obj.supports_words:
                    words = asr_obj.words(wav)
                    utts += _group(words, [spk] * len(words))
                else:
                    # провайдер без тайм-кодов: канал целиком = одна реплика
                    text = asr_obj.transcribe(wav).strip()
                    if text:
                        utts.append({"speaker": spk, "start": 0.0,
                                     "end": float(sf.info(wav).duration), "text": text})
        utts.sort(key=lambda u: u["start"])
        return utts

    # mono: pyannote даёт «кто/когда».
    # Поднимаем его ЧЕРЕЗ models_state, иначе загруженная модель не попадёт под
    # учёт и сторож простоя никогда её не выгрузит (видеопамять останется занятой).
    from app import models_state
    with as_wav(path) as wav:
        models_state.get_diarizer()
        turns = diarize.turns_of(wav, num_speakers=num_speakers, min_speakers=min_speakers,
                                 max_speakers=max_speakers, exclusive=exclusive)
        if asr_obj.supports_words:
            words = asr_obj.words(wav)
            return _group(words, diarize.assign_speakers(words, turns))
        return _turn_texts(wav, turns, asr_obj)
