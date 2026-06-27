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
import tempfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse

from app import asr, auth, llm, pipeline

_APP = Path(__file__).parent

VERSION = "0.1.0"

ASR_MODELS = {
    "gigaam": os.getenv("GIGAAM_MODEL", "gigaam-v3-e2e-rnnt").strip(),
    "parakeet": os.getenv("PARAKEET_MODEL", "nemo-parakeet-tdt-0.6b-v3").strip(),
}
DEFAULT_MODEL = os.getenv("DIARIZE_DEFAULT_MODEL", "gigaam").strip().lower()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))

app = FastAPI(title="Diarization", version=VERSION)
_ASR: dict[str, asr.Asr] = {}


def _default_model() -> str:
    return DEFAULT_MODEL if DEFAULT_MODEL in ASR_MODELS else "gigaam"


def _get_asr(name: str) -> asr.Asr:
    key = name if name in ASR_MODELS else _default_model()
    if key not in _ASR:
        _ASR[key] = asr.Asr(ASR_MODELS[key])
    return _ASR[key]


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
        lines.append(f"{prefix}{u['speaker']}: {u['text']}")
    return "\n".join(lines)


@app.on_event("startup")
def _startup() -> None:
    print(f"[startup] device={_device()} providers={_onnx_providers()}", flush=True)
    try:
        _get_asr(_default_model())
    except Exception as e:
        print(f"[startup] asr warmup failed: {e}", flush=True)
    try:
        from app import diarize
        diarize.warmup()
    except Exception as e:
        print(f"[startup] diarization warmup failed: {e}", flush=True)


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
        "device": _device(),
        "providers": _onnx_providers(),
    }


@app.get("/v1/models", dependencies=[Depends(verify_auth)])
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [{"id": n, "object": "model", "created": now, "owned_by": "local"} for n in ASR_MODELS],
    }


@app.post("/v1/diarize", dependencies=[Depends(verify_auth)])
async def diarize_endpoint(
    file: UploadFile = File(...),
    model: str = Form("gigaam"),
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

    ext = Path(file.filename or "audio").suffix or ".bin"
    tmp_path = None
    size = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded file is too large")
                tmp.write(chunk)

        if not asr.has_audio_stream(tmp_path):
            raise HTTPException(status_code=400, detail="Uploaded file has no audio track")

        try:
            asr_obj = _get_asr(model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load ASR model '{model}': {e}") from e

        try:
            utts = pipeline.diarize_file(tmp_path, asr_obj, mode=md)
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
        body = text
        for key, title in (("summary", "САММАРИ"), ("follow_up", "FOLLOW-UP"), ("todo", "TO-DO")):
            if key in analysis:
                body += f"\n\n===== {title} =====\n{analysis[key]}"
        return PlainTextResponse(body)
    return {"segments": utts, "text": text, **analysis}
