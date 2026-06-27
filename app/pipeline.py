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


def _group(words: list[Word], speaker_fn) -> list[dict]:
    utts: list[dict] = []
    for w in words:
        spk = speaker_fn(w)
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


def diarize_file(path: str, asr_obj: asr.Asr, mode: str = "auto") -> list[dict]:
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
            with as_wav(path, channel=ch) as wav:
                words = asr_obj.words(wav)
            spk = f"Канал {ch + 1}"
            utts += _group(words, lambda w, s=spk: s)
        utts.sort(key=lambda u: u["start"])
        return utts

    # mono: pyannote + выравнивание слов по turn'ам
    with as_wav(path) as wav:
        turns = diarize.turns_of(wav)
        words = asr_obj.words(wav)
    return _group(words, lambda w: diarize.speaker_of(w.start, w.end, turns))
