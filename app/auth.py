"""Простой логин/пароль с подписанной cookie-сессией (без внешних зависимостей).

Включается, когда заданы AUTH_USERNAME + AUTH_PASSWORD_HASH (sha256 hex) + AUTH_SECRET.
Если не заданы — auth выключен (доступ как раньше: по API-ключу или открыто).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

AUTH_USERNAME = os.getenv("AUTH_USERNAME", "")
AUTH_PASSWORD_HASH = os.getenv("AUTH_PASSWORD_HASH", "")  # sha256(password) в hex
AUTH_SECRET = os.getenv("AUTH_SECRET", "")
SESSION_TTL = int(os.getenv("AUTH_SESSION_TTL", str(7 * 24 * 3600)))  # 7 дней
COOKIE = "diar_session"


def enabled() -> bool:
    return bool(AUTH_USERNAME and AUTH_PASSWORD_HASH and AUTH_SECRET)


def check_credentials(username: str, password: str) -> bool:
    if not enabled():
        return False
    ph = hashlib.sha256((password or "").encode()).hexdigest()
    return hmac.compare_digest(username or "", AUTH_USERNAME) and hmac.compare_digest(ph, AUTH_PASSWORD_HASH)


def _sign(msg: str) -> str:
    return hmac.new(AUTH_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()


def make_token() -> str:
    payload = f"{AUTH_USERNAME}|{int(time.time()) + SESSION_TTL}"
    b = base64.urlsafe_b64encode(payload.encode()).decode()
    return f"{b}.{_sign(b)}"


def valid_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    b, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(sig, _sign(b)):
        return False
    try:
        user, exp = base64.urlsafe_b64decode(b.encode()).decode().split("|")
    except Exception:
        return False
    return user == AUTH_USERNAME and int(exp) > int(time.time())
