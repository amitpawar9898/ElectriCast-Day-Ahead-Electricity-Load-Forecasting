"""Runtime configuration for the Ask ElectriCast assistant.

Every value can be overridden from the environment; no key is committed.
The assistant is optional: when no provider is reachable the API returns
"Assistant unavailable" and the rest of the dashboard is unaffected.
"""

import os


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


CHAT_PROVIDER = os.environ.get("CHAT_PROVIDER", "auto").strip().lower()
if CHAT_PROVIDER not in ("auto", "ollama", "gemini"):
    CHAT_PROVIDER = "auto"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b").strip()

GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or ""
).strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_API_BASE = os.environ.get(
    "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
).rstrip("/")

CHAT_TIMEOUT_SECONDS = _env_float("CHAT_TIMEOUT_SECONDS", 60.0)
CHAT_HEALTH_TIMEOUT_SECONDS = _env_float("CHAT_HEALTH_TIMEOUT_SECONDS", 1.0)
CHAT_MAX_QUESTION_CHARS = _env_int("CHAT_MAX_QUESTION_CHARS", 500)
CHAT_MAX_ANSWER_CHARS = _env_int("CHAT_MAX_ANSWER_CHARS", 4000)
CHAT_STATUS_TTL_SECONDS = _env_float("CHAT_STATUS_TTL_SECONDS", 15.0)
