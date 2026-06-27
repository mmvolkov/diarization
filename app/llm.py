"""Постобработка транскрипта через LLM сервера (gpt-oss-120b, OpenAI-совместимый).

Доступ изнутри docker-сети: LLM_BASE_URL=http://gpt-oss-120b:8000/v1.
Каждая задача (summary/follow_up/todo) — отдельный запрос, plain-text ответ.
"""
from __future__ import annotations

import os

from openai import OpenAI

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://gpt-oss-120b:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy-key")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss-120b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

_SYS = (
    "Ты ассистент, который обрабатывает транскрипт делового совещания на русском языке. "
    "Отвечай по-русски, только запрошенным содержимым, без преамбул и пояснений."
)

_client = None


def _c() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT)
    return _client


def _transcript(utts: list[dict]) -> str:
    out = []
    for u in utts:
        s = int(u["start"])
        out.append(f"[{s // 60:02d}:{s % 60:02d}] {u['speaker']}: {u['text']}")
    return "\n".join(out)


def _ask(transcript: str, instruction: str) -> str:
    try:
        resp = _c().chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content": f"{instruction}\n\nТранскрипт:\n\n{transcript}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # LLM недоступен/ошибка — не валим весь ответ
        return f"(не удалось сформировать: {e})"


def analyze(utts: list[dict], summary: bool = False, follow_up: bool = False, todo: bool = False) -> dict:
    """Вернуть запрошенные секции постобработки. Ключи: summary / follow_up / todo."""
    t = _transcript(utts)
    out: dict[str, str] = {}
    if summary:
        out["summary"] = _ask(
            t,
            "Сделай САММАРИ встречи — что на ней произошло. Освети четыре аспекта, каждый отдельной "
            "группой: что обсуждали; к каким выводам пришли; какие решения приняли; какие вопросы "
            "остались открытыми. Формат — маркированный список: группа как жирный заголовок "
            "(**Обсуждали:**, **Выводы:**, **Решения:**, **Открытые вопросы:**), под ней пункты "
            "с отступом в 2 пробела. Верни только список, без вступления.",
        )
    if follow_up:
        out["follow_up"] = _ask(
            t,
            "Составь FOLLOW-UP — готовое деловое сообщение участникам по итогам встречи (как письмо). "
            "Включи: краткое резюме итогов; подтверждённые договорённости и решения; следующие шаги; "
            "запрос недостающих материалов (если есть); предложение по следующей встрече (если уместно). "
            "Пиши деловым тоном от первого лица. Блок «Следующие шаги» оформи маркированным списком "
            "(каждый пункт с «- »). Без служебных пояснений до и после.",
        )
    if todo:
        out["todo"] = _ask(
            t,
            "Составь TO-DO — конкретные задачи по итогам встречи в виде таблицы Markdown РОВНО с тремя "
            "колонками: «Задача», «Ответственный», «Срок». Задача — конкретное действие (при "
            "необходимости с ожидаемым результатом). Ответственный — имя или спикер из контекста; если "
            "не ясно — «—». Срок — если назывался на встрече, иначе «—». Верни ТОЛЬКО таблицу "
            "(с шапкой и строкой-разделителем), без текста до и после.",
        )
    return out
