"""Ленивая загрузка и выгрузка локальных моделей (onnx-ASR + pyannote).

Локальные модели держат видеопамять (pyannote ~5 ГБ), поэтому при старте
контейнера они НЕ грузятся. Модель поднимается при первом запросе, которому она
действительно нужна, и выгружается либо по простою (LOCAL_IDLE_UNLOAD_S), либо
по кнопке в интерфейсе (POST /v1/models/unload).

Облачный провайдер (MWS) сюда не попадает: он ничего не держит в памяти.
"""
from __future__ import annotations

import os
import threading
import time

IDLE_UNLOAD_S = float(os.getenv("LOCAL_IDLE_UNLOAD_S", "900"))
SWEEP_EVERY_S = 60.0

_lock = threading.RLock()
_asr: dict[str, object] = {}
_diar_loaded = False
_last_used = 0.0


def _touch() -> None:
    global _last_used
    _last_used = time.time()


def get_asr(key: str, model_name: str):
    """Локальная ASR-модель по ключу (gigaam/parakeet); грузится при первом вызове."""
    from app import asr
    with _lock:
        if key not in _asr:
            _asr[key] = asr.Asr(model_name)
        _touch()
        return _asr[key]


def get_diarizer() -> None:
    """Поднять pyannote (идемпотентно). Сам объект живёт в app.diarize."""
    from app import diarize
    with _lock:
        diarize.warmup()
        global _diar_loaded
        _diar_loaded = True
        _touch()


def loaded() -> dict:
    with _lock:
        idle = (time.time() - _last_used) if _last_used else None
        return {
            "asr": sorted(_asr.keys()),
            "diarizer": _diar_loaded,
            "idle_seconds": round(idle) if idle is not None else None,
            "idle_unload_seconds": IDLE_UNLOAD_S,
        }


def unload_all() -> dict:
    """Выгрузить всё локальное и вернуть видеопамять. Возвращает, что было выгружено."""
    global _diar_loaded, _last_used
    with _lock:
        freed = {"asr": sorted(_asr.keys()), "diarizer": _diar_loaded}
        _asr.clear()
        if _diar_loaded:
            from app import diarize
            diarize.unload()
            _diar_loaded = False
        _last_used = 0.0

    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return freed


def _sweep_loop() -> None:
    while True:
        time.sleep(SWEEP_EVERY_S)
        try:
            with _lock:
                busy = bool(_asr) or _diar_loaded
                idle = (time.time() - _last_used) if _last_used else 0.0
            if busy and IDLE_UNLOAD_S > 0 and idle >= IDLE_UNLOAD_S:
                freed = unload_all()
                print(f"[models] выгружено по простою ({int(idle)} с): {freed}", flush=True)
        except Exception as e:  # фоновая задача не должна ронять сервис
            print(f"[models] sweep error: {e}", flush=True)


def start_sweeper() -> None:
    threading.Thread(target=_sweep_loop, name="models-sweeper", daemon=True).start()
