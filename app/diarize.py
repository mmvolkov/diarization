"""Диаризация моно-аудио через pyannote (источник «кто/когда»).

pyannote 4.x: pipeline возвращает DiarizeOutput(.speaker_diarization=Annotation).
Аудио подаём как waveform-словарь {'waveform','sample_rate'}, минуя torchcodec
(ему нужны shared FFmpeg-DLL, которых может не быть).
"""
from __future__ import annotations

import os

import soundfile as sf
import torch
from pyannote.audio import Pipeline

DIAR_MODEL = os.getenv("DIAR_MODEL", "pyannote/speaker-diarization-community-1")

_pipe = None


def _pipeline():
    global _pipe
    if _pipe is None:
        token = os.environ.get("HF_TOKEN")
        try:
            _pipe = Pipeline.from_pretrained(DIAR_MODEL, token=token)
        except TypeError:  # старый API
            _pipe = Pipeline.from_pretrained(DIAR_MODEL, use_auth_token=token)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _pipe.to(torch.device(dev))
    return _pipe


def warmup() -> None:
    _pipeline()


def turns_of(wav_path: str):
    """pyannote -> [(start, end, speaker)] для моно-16к WAV."""
    samples, sr = sf.read(wav_path, dtype="float32")
    wav_t = torch.from_numpy(samples).unsqueeze(0)  # (1, N)
    diar = _pipeline()({"waveform": wav_t, "sample_rate": sr})
    ann = getattr(diar, "speaker_diarization", diar)
    turns = [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]
    turns.sort(key=lambda x: x[0])
    return turns


def speaker_of(ws: float, we: float, turns) -> str:
    """Спикер слова [ws, we]: max перекрытие, иначе ближайший turn по времени."""
    best, best_ov = None, 0.0
    for ts, te, spk in turns:
        ov = min(we, te) - max(ws, ts)
        if ov > best_ov:
            best_ov, best = ov, spk
    if best is not None:
        return best
    mid = (ws + we) / 2
    nearest, best_d = "SPEAKER_?", float("inf")
    for ts, te, spk in turns:
        d = 0.0 if ts <= mid <= te else min(abs(mid - ts), abs(mid - te))
        if d < best_d:
            best_d, nearest = d, spk
    return nearest
