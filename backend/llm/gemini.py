import time
import json
from urllib import request as urlrequest
from urllib import error as urlerror
import google.genai as genai

from config import (
    GEMINI_API_KEY,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GENERATION_TEMPERATURE,
    GENERATION_MAX_OUTPUT_TOKENS,
    MODEL_REQUEST_TIMEOUT_SECONDS,
    MODEL_REQUEST_MAX_RETRIES,
    MODEL_RETRY_BACKOFF_SECONDS,
    OLLAMA_MODEL_CHECK_TIMEOUT_SECONDS,
)

try:
    import ollama
except ImportError:  # pragma: no cover - environment dependent
    ollama = None


_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
_ollama_client = ollama.Client(host=OLLAMA_BASE_URL) if ollama else None
_OLLAMA_API_BASE = OLLAMA_BASE_URL.rstrip("/")
_last_ollama_error = ""


def _with_timeout_and_retries(func, *args, max_retries=None, timeout_sec=None, **kwargs):
    """
    Generic retry wrapper with exponential backoff.
    max_retries: total attempts (retries + 1 initial).
    timeout_sec: max seconds to wait for the full request sequence.
    Returns the result or raises the last exception.
    """
    if max_retries is None:
        max_retries = MODEL_REQUEST_MAX_RETRIES
    if timeout_sec is None:
        timeout_sec = MODEL_REQUEST_TIMEOUT_SECONDS

    start_time = time.time()
    last_exc = None
    backoff = MODEL_RETRY_BACKOFF_SECONDS

    for attempt in range(max_retries + 1):
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            raise TimeoutError(f"Request exceeded timeout of {timeout_sec}s") from last_exc

        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait_time = backoff * (2 ** attempt)
                remaining = timeout_sec - elapsed
                if wait_time > remaining:
                    wait_time = remaining / 2
                if wait_time > 0:
                    time.sleep(wait_time)

    raise last_exc


def _ollama_http_get(path: str):
    req = urlrequest.Request(f"{_OLLAMA_API_BASE}{path}", method="GET")
    with urlrequest.urlopen(req, timeout=OLLAMA_MODEL_CHECK_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_ollama_model_names() -> list[str]:
    tags = _ollama_http_get("/api/tags")
    models = tags.get("models", []) if isinstance(tags, dict) else []
    names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("name") or model.get("model") or ""
        else:
            name = getattr(model, "name", None) or getattr(model, "model", None) or ""
        if name:
            names.append(str(name))
    return names


def _resolve_ollama_model_name() -> str | None:
    """Resolve a usable local model name with sensible fallbacks."""
    try:
        model_names = _get_ollama_model_names()
    except Exception:
        return None

    if not model_names:
        return None

    # 1) Exact configured model or namespace variant.
    for name in model_names:
        if name == OLLAMA_MODEL or name.startswith(f"{OLLAMA_MODEL}:") or name.split(":")[0] == OLLAMA_MODEL:
            return name

    # 2) Any model containing configured token.
    for name in model_names:
        if OLLAMA_MODEL in name:
            return name

    # 3) Common llama fallbacks when users have llama locally but renamed/tagged.
    preferred_prefixes = ("llama3", "llama3.1", "llama3.2", "llama")
    for prefix in preferred_prefixes:
        for name in model_names:
            if name.startswith(prefix):
                return name

    # 4) Last resort: first available model.
    return model_names[0]


def _ollama_http_post(path: str, payload: dict, timeout_sec: float | None = None):
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{_OLLAMA_API_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout_sec or MODEL_REQUEST_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ollama_http_stream_post(path: str, payload: dict, timeout_sec: float | None = None):
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{_OLLAMA_API_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout_sec or MODEL_REQUEST_TIMEOUT_SECONDS) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            yield json.loads(line)


def _is_ollama_model_available() -> bool:
    global _last_ollama_error
    try:
        def check_list():
            return _resolve_ollama_model_name() is not None

        ok = _with_timeout_and_retries(check_list, max_retries=1, timeout_sec=OLLAMA_MODEL_CHECK_TIMEOUT_SECONDS)
        if not ok:
            _last_ollama_error = f"No usable model found from {OLLAMA_BASE_URL}. Configure OLLAMA_MODEL or pull a model."
        return ok
    except Exception as exc:
        _last_ollama_error = f"Failed to reach Ollama at {OLLAMA_BASE_URL}: {exc}"
        return False


def _generate_response_ollama(prompt: str) -> str:
    resolved_model = _resolve_ollama_model_name() or OLLAMA_MODEL

    def chat_request():
        if _ollama_client is not None:
            return _ollama_client.chat(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": GENERATION_TEMPERATURE},
            )
        return _ollama_http_post(
            "/api/chat",
            {
                "model": resolved_model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": GENERATION_TEMPERATURE},
                "stream": False,
            },
        )

    response = _with_timeout_and_retries(chat_request)
    message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", None)
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "") or ""


def _generate_response_stream_ollama(prompt: str):
    resolved_model = _resolve_ollama_model_name() or OLLAMA_MODEL

    def stream_request():
        if _ollama_client is not None:
            return _ollama_client.chat(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": GENERATION_TEMPERATURE},
            )
        return _ollama_http_stream_post(
            "/api/chat",
            {
                "model": resolved_model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": GENERATION_TEMPERATURE},
                "stream": True,
            },
        )

    stream = _with_timeout_and_retries(stream_request, max_retries=1)
    for chunk in stream:
        if isinstance(chunk, dict):
            message = chunk.get("message", {})
            if isinstance(message, dict):
                text = message.get("content")
            else:
                text = getattr(message, "content", None)
            if text:
                yield text
        else:
            message = getattr(chunk, "message", None)
            if isinstance(message, dict):
                text = message.get("content")
            else:
                text = getattr(message, "content", None)
            if text:
                yield text


def _generate_response_gemini(prompt: str) -> str:
    if _gemini_client is None:
        detail = _last_ollama_error or "Ollama is unavailable."
        raise RuntimeError(f"Gemini API key is not configured and local Ollama is unavailable. {detail}")

    def gemini_request():
        return _gemini_client.models.generate_content(
            config={
                "temperature": GENERATION_TEMPERATURE,
                "maxOutputTokens": GENERATION_MAX_OUTPUT_TOKENS,
                "system_instruction": prompt,
            },
            model=MODEL_NAME,
            contents=prompt,
        )

    response = _with_timeout_and_retries(gemini_request)
    return response.text


def _generate_response_stream_gemini(prompt: str):
    if _gemini_client is None:
        detail = _last_ollama_error or "Ollama is unavailable."
        raise RuntimeError(f"Gemini API key is not configured and local Ollama is unavailable. {detail}")

    def gemini_stream_request():
        return _gemini_client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": GENERATION_TEMPERATURE,
                "maxOutputTokens": GENERATION_MAX_OUTPUT_TOKENS,
                "system_instruction": prompt,
            },
        )

    stream = _with_timeout_and_retries(gemini_stream_request, max_retries=1)
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def generate_response(prompt):
    if _is_ollama_model_available():
        try:
            return _generate_response_ollama(prompt)
        except Exception:
            pass
    return _generate_response_gemini(prompt)


def generate_response_stream(prompt):
    """Yields text chunks using Ollama llama3 first, then Gemini fallback."""
    if _is_ollama_model_available():
        try:
            yield from _generate_response_stream_ollama(prompt)
            return
        except Exception:
            pass

    yield from _generate_response_stream_gemini(prompt)
