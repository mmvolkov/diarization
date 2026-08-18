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


def unload() -> None:
    """Отпустить pipeline, чтобы освободить видеопамять (см. app/models_state.py)."""
    global _pipe
    _pipe = None


def turns_of(wav_path: str, num_speakers: int | None = None,
             min_speakers: int | None = None, max_speakers: int | None = None,
             exclusive: bool = True):
    """pyannote -> [(start, end, speaker)] для моно-16к WAV.

    num_speakers / min_speakers / max_speakers: подсказка о числе участников.
    Документация pyannote рекомендует её передавать, когда число известно
    (для звонка это почти всегда 2) — качество разметки заметно выше.

    exclusive=True берёт exclusive_speaker_diarization: в каждый момент ровно
    один говорящий. Обычная диаризация допускает перекрытия, и слово на стыке
    приходится «делить» между спикерами — отсюда разорванные реплики. Для
    склейки с ASR exclusive-вариант удобнее, и Community-1 отдаёт его сам.
    """
    samples, sr = sf.read(wav_path, dtype="float32")
    wav_t = torch.from_numpy(samples).unsqueeze(0)  # (1, N)
    hints = {k: v for k, v in (("num_speakers", num_speakers),
                               ("min_speakers", min_speakers),
                               ("max_speakers", max_speakers)) if v}
    diar = _pipeline()({"waveform": wav_t, "sample_rate": sr}, **hints)
    ann = None
    if exclusive:
        ann = getattr(diar, "exclusive_speaker_diarization", None)
    if ann is None:
        ann = getattr(diar, "speaker_diarization", diar)
    turns = [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]
    turns.sort(key=lambda x: x[0])
    return turns


MIN_WORD_S = 0.30   # короче — считаем слово точечным и расширяем до окна
TIE_S = 0.05        # разница перекрытий меньше этой — считаем ничьёй


def _widen(ws: float, we: float) -> tuple[float, float]:
    """Расширить вырожденный интервал слова до осмысленного окна.

    ASR иногда отдаёт слову одинаковые start и end. С нулевой длиной перекрытие
    с любым turn'ом равно нулю, выбор спикера становится случайным и слово
    уезжает к соседу — отсюда разорванные реплики.
    """
    if we - ws >= MIN_WORD_S:
        return ws, we
    mid = (ws + we) / 2
    return mid - MIN_WORD_S / 2, mid + MIN_WORD_S / 2


def speaker_of(ws: float, we: float, turns) -> str:
    """Спикер слова [ws, we]: max перекрытие, иначе ближайший turn по времени."""
    a, b = _widen(ws, we)
    best, best_ov = None, 0.0
    for ts, te, spk in turns:
        ov = min(b, te) - max(a, ts)
        if ov > best_ov:
            best_ov, best = ov, spk
    if best is not None:
        return best
    mid = (a + b) / 2
    nearest, best_d = "SPEAKER_?", float("inf")
    for ts, te, spk in turns:
        d = 0.0 if ts <= mid <= te else min(abs(mid - ts), abs(mid - te))
        if d < best_d:
            best_d, nearest = d, spk
    return nearest


def assign_speakers(words, turns) -> list[str]:
    """Спикер для каждого слова с учётом соседей.

    Слово, у которого два turn'а перекрываются почти поровну (типично на стыке
    реплик), само по себе неразрешимо. Такое слово наследует спикера следующего
    уверенно определённого слова: на практике обрывок принадлежит начинающейся
    реплике, а не заканчивающейся.
    """
    resolved: list[str | None] = []
    for w in words:
        a, b = _widen(w.start, w.end)
        best, best_ov, second_ov = None, 0.0, 0.0
        for ts, te, spk in turns:
            ov = min(b, te) - max(a, ts)
            if ov > best_ov:
                best_ov, second_ov, best = ov, best_ov, spk
            elif ov > second_ov:
                second_ov = ov
        resolved.append(None if best is None or best_ov - second_ov < TIE_S else best)

    # сначала протягиваем назад — от следующего уверенного слова
    nxt: str | None = None
    for i in range(len(resolved) - 1, -1, -1):
        if resolved[i] is None:
            resolved[i] = nxt
        else:
            nxt = resolved[i]
    # хвост в начале (уверенных слов дальше не нашлось) — от предыдущего
    prev = "SPEAKER_?"
    for i, v in enumerate(resolved):
        if v is None:
            resolved[i] = prev
        else:
            prev = v
    return resolved  # type: ignore[return-value]
