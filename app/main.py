"""Сервис диаризации (форк каркаса из transcriptions).

Эндпоинты:
  POST /v1/diarize   — multipart: file, model(gigaam|parakeet), mode(auto|stereo|mono),
                       response_format(json|text). Возвращает диаризованный транскрипт.
  GET  /v1/models    — список ASR-моделей.
  GET  /health       — liveness + устройство/провайдеры.

Диаризация: стерео-звонки — по каналам; моно-совещания — pyannote. ASR — GigaAM с тайм-кодами.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response

from app import asr, auth, llm, models_state, pipeline

_APP = Path(__file__).parent

VERSION = "0.1.0"

ASR_MODELS = {
    "gigaam": os.getenv("GIGAAM_MODEL", "gigaam-v3-e2e-rnnt").strip(),
    # raw-вариант без e2e-нормализации: e2e «причёсывает» редкие имена собственные
    # («Крост» → «Кросс»), raw их сохраняет — годится для сверки имён.
    "gigaam-raw": os.getenv("GIGAAM_RAW_MODEL", "gigaam-v3-rnnt").strip(),
    "parakeet": os.getenv("PARAKEET_MODEL", "nemo-parakeet-tdt-0.6b-v3").strip(),
}
DEFAULT_MODEL = os.getenv("DIARIZE_DEFAULT_MODEL", "gigaam").strip().lower()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))

# Облачный ASR MWS: список моделей и доступ задаются в .env.
MWS_BASE_URL = os.getenv("MWS_BASE_URL", "").strip()
MWS_API_KEY = os.getenv("MWS_API_KEY", "").strip()
MWS_MODELS = [m.strip() for m in os.getenv("MWS_MODELS", "").split(",") if m.strip()]
MWS_DEFAULT_MODEL = os.getenv("MWS_DEFAULT_MODEL", "").strip() or (MWS_MODELS[0] if MWS_MODELS else "")
MWS_TIMEOUT = float(os.getenv("MWS_TIMEOUT", "600"))
DEFAULT_PROVIDER = os.getenv("ASR_PROVIDER_DEFAULT", "local").strip().lower()

app = FastAPI(title="Diarization", version=VERSION)


def _mws_enabled() -> bool:
    return bool(MWS_BASE_URL and MWS_API_KEY and MWS_MODELS)


def _default_model() -> str:
    return DEFAULT_MODEL if DEFAULT_MODEL in ASR_MODELS else "gigaam"


def _default_provider() -> str:
    if DEFAULT_PROVIDER == "mws" and _mws_enabled():
        return "mws"
    return "local"


def _get_asr(name: str, provider: str = "local"):
    """ASR-провайдер. local — ленивая загрузка на GPU, mws — HTTP, память не держит."""
    if provider == "mws":
        if not _mws_enabled():
            raise HTTPException(status_code=400, detail="MWS не настроен: задайте MWS_BASE_URL/MWS_API_KEY/MWS_MODELS")
        model = name if name in MWS_MODELS else MWS_DEFAULT_MODEL
        return asr.MwsAsr(model, MWS_BASE_URL, MWS_API_KEY, MWS_TIMEOUT)
    key = name if name in ASR_MODELS else _default_model()
    return models_state.get_asr(key, ASR_MODELS[key])


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


def _device() -> str:
    return "gpu" if "CUDAExecutionProvider" in _onnx_providers() else "cpu"


def _load_api_keys() -> set[str] | None:
    raw = os.environ.get("DIARIZE_API_KEYS", "").strip()
    if raw:
        keys = {k.strip() for k in raw.replace(";", ",").split(",") if k.strip()}
        return keys or None
    return None


async def verify_auth(request: Request) -> None:
    # 1) сессия (cookie) — для браузера после логина
    if auth.enabled() and auth.valid_token(request.cookies.get(auth.COOKIE)):
        return
    # 2) API-ключ (Bearer / X-API-Key) — для curl/SDK
    keys = _load_api_keys()
    if keys is not None:
        token = None
        h = request.headers.get("Authorization", "")
        if h.startswith("Bearer "):
            token = h[7:]
        if token is None:
            token = request.headers.get("X-API-Key")
        if token and token in keys:
            return
    # 3) ничего не настроено — открытый доступ (как раньше)
    if keys is None and not auth.enabled():
        return
    raise HTTPException(status_code=401, detail="Invalid or missing credentials")


def _mmss(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def _render(utts: list[dict], timestamps: bool = True) -> str:
    lines = []
    for u in utts:
        prefix = f"[{_mmss(u['start'])}-{_mmss(u['end'])}] " if timestamps else ""
        # без диаризации спикера нет — не печатаем пустой «: »
        who = f"{u['speaker']}: " if u.get("speaker") else ""
        lines.append(f"{prefix}{who}{u['text']}")
    return "\n".join(lines)


async def _save_upload(file: UploadFile) -> str:
    """Сохранить загруженный файл во временный, с проверкой лимита размера."""
    ext = Path(file.filename or "audio").suffix or ".bin"
    size = 0
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        path = tmp.name
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp.close()
                os.unlink(path)
                raise HTTPException(status_code=413, detail="Uploaded file is too large")
            tmp.write(chunk)
    return path


@app.on_event("startup")
def _startup() -> None:
    # Локальные модели намеренно НЕ греем: они занимают видеопамять и поднимаются
    # только когда в интерфейсе выбран локальный провайдер (см. app/models_state.py).
    print(f"[startup] device={_device()} providers={_onnx_providers()} "
          f"default_provider={_default_provider()} mws={'on' if _mws_enabled() else 'off'} "
          f"idle_unload={models_state.IDLE_UNLOAD_S}s", flush=True)
    models_state.start_sweeper()


@app.get("/", include_in_schema=False)
async def index(request: Request):
    # Простой веб-интерфейс. Если включён логин — требуем сессию.
    if auth.enabled() and not auth.valid_token(request.cookies.get(auth.COOKIE)):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(_APP / "index.html")


@app.get("/login", include_in_schema=False)
async def login_page():
    return FileResponse(_APP / "login.html")


@app.post("/login", include_in_schema=False)
async def login_submit(username: str = Form(""), password: str = Form("")):
    if auth.check_credentials(username, password):
        resp = RedirectResponse("/", status_code=302)
        resp.set_cookie(auth.COOKIE, auth.make_token(), max_age=auth.SESSION_TTL,
                        httponly=True, secure=True, samesite="lax", path="/")
        return resp
    return RedirectResponse("/login?e=1", status_code=302)


@app.get("/logout", include_in_schema=False)
async def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(auth.COOKIE, path="/")
    return resp


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": VERSION,
        "default_model": _default_model(),
        "default_provider": _default_provider(),
        "mws": _mws_enabled(),
        "device": _device(),
        "providers": _onnx_providers(),
        "loaded": models_state.loaded(),
    }


@app.get("/v1/models", dependencies=[Depends(verify_auth)])
async def list_models():
    now = int(time.time())
    data = [{"id": n, "object": "model", "created": now, "owned_by": "local", "provider": "local"}
            for n in ASR_MODELS]
    data += [{"id": n, "object": "model", "created": now, "owned_by": "mws", "provider": "mws"}
             for n in (MWS_MODELS if _mws_enabled() else [])]
    return {
        "object": "list",
        "data": data,
        "defaults": {"provider": _default_provider(), "local": _default_model(), "mws": MWS_DEFAULT_MODEL},
    }


@app.post("/v1/models/unload", dependencies=[Depends(verify_auth)])
async def unload_models():
    """Немедленно выгрузить локальные модели и вернуть видеопамять."""
    freed = models_state.unload_all()
    return {"unloaded": freed, "loaded": models_state.loaded()}


@app.post("/v1/export/docx", dependencies=[Depends(verify_auth)])
async def export_docx(request: Request):
    from app import export
    data = await request.json()
    content = export.build_docx(data)
    return Response(
        content,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="diarization.docx"'},
    )


@app.post("/v1/test", dependencies=[Depends(verify_auth)])
async def test_endpoint(
    file: UploadFile = File(...),
    provider: str = Form("local"),
    model: str = Form(""),
    seconds: int = Form(30),
):
    """Пробная расшифровка первых N секунд загруженного файла выбранным провайдером.

    Проверяет весь путь целиком: доступ (ключ/сеть для MWS либо загрузку модели на
    GPU для local), саму модель и качество на реальном материале. Диаризацию не
    трогает — она проверяется полным прогоном.
    """
    pv = (provider or "local").strip().lower()
    if pv not in {"local", "mws"}:
        raise HTTPException(status_code=400, detail="provider must be one of: local, mws")
    seconds = max(5, min(int(seconds or 30), 120))

    src = await _save_upload(file)
    cut = None
    try:
        if not asr.has_audio_stream(src):
            raise HTTPException(status_code=400, detail="Uploaded file has no audio track")
        cut = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        cut.close()
        subprocess.run(
            ["ffmpeg", "-v", "error", "-t", str(seconds), "-i", src,
             "-ar", "16000", "-ac", "1", "-y", cut.name],
            check=True, capture_output=True,
        )
        t0 = time.time()
        try:
            asr_obj = _get_asr(model, pv)
            text = asr_obj.transcribe(cut.name)
        except HTTPException:
            raise
        except Exception as e:
            return {"ok": False, "provider": pv, "model": model,
                    "elapsed": round(time.time() - t0, 2), "error": str(e)}
        return {
            "ok": True,
            "provider": pv,
            "model": getattr(asr_obj, "model_name", model or _default_model()),
            "seconds": seconds,
            "elapsed": round(time.time() - t0, 2),
            "text": text,
            "loaded": models_state.loaded(),
        }
    finally:
        for p in (src, cut.name if cut else None):
            if p and os.path.exists(p):
                os.unlink(p)


@app.post("/v1/diarize", dependencies=[Depends(verify_auth)])
async def diarize_endpoint(
    file: UploadFile = File(...),
    model: str = Form("gigaam"),
    provider: str = Form("local"),     # local (наша карта) | mws (облако)
    speakers: bool = Form(True),       # false — просто расшифровка, pyannote не поднимается
    num_speakers: int = Form(0),       # точное число участников (0 — определить самому)
    min_speakers: int = Form(0),       # либо диапазон: от…
    max_speakers: int = Form(0),       # …до
    exclusive: bool = Form(True),      # exclusive-диаризация: один говорящий в момент времени
    mode: str = Form("auto"),
    response_format: str = Form("json"),
    merge_speakers: bool = Form(True),   # слить подряд идущие реплики одного спикера
    timestamps: bool = Form(True),       # показывать тайм-коды в text-выводе
    summary: bool = Form(False),         # LLM: саммари
    follow_up: bool = Form(False),       # LLM: follow-up (открытые вопросы)
    todo: bool = Form(False),            # LLM: to-do (задачи)
):
    rf = (response_format or "json").strip().lower()
    if rf not in {"json", "text"}:
        raise HTTPException(status_code=400, detail="response_format must be one of: json, text")
    md = (mode or "auto").strip().lower()
    if md not in {"auto", "stereo", "mono"}:
        raise HTTPException(status_code=400, detail="mode must be one of: auto, stereo, mono")

    pv = (provider or "local").strip().lower()
    if pv not in {"local", "mws"}:
        raise HTTPException(status_code=400, detail="provider must be one of: local, mws")

    tmp_path = None
    try:
        tmp_path = await _save_upload(file)

        if not asr.has_audio_stream(tmp_path):
            raise HTTPException(status_code=400, detail="Uploaded file has no audio track")

        try:
            asr_obj = _get_asr(model, pv)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load ASR model '{model}': {e}") from e

        try:
            utts = pipeline.diarize_file(
                tmp_path, asr_obj, mode=md, speakers=speakers,
                num_speakers=num_speakers or None,
                min_speakers=min_speakers or None,
                max_speakers=max_speakers or None,
                exclusive=exclusive,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Diarization failed: {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if merge_speakers:
        utts = pipeline.merge_consecutive(utts)

    text = _render(utts, timestamps)
    analysis = llm.analyze(utts, summary=summary, follow_up=follow_up, todo=todo) \
        if (summary or follow_up or todo) else {}

    if rf == "text":
        parts = []
        for key, title in (("summary", "САММАРИ"), ("follow_up", "FOLLOW-UP"), ("todo", "TO-DO")):
            if key in analysis:
                parts.append(f"===== {title} =====\n{analysis[key]}")
        parts.append(f"===== ТРАНСКРИПТ =====\n{text}")
        return PlainTextResponse("\n\n".join(parts))
    return {**analysis, "segments": utts, "text": text}
