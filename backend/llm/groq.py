import time
import json
from urllib import request as urlrequest

try:
    from groq import Groq
except ImportError:  # pragma: no cover - environment dependent
    Groq = None

from config import (
    GROQ_API_KEY,
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


_groq_client = Groq(api_key=GROQ_API_KEY) if Groq and GROQ_API_KEY else None
_ollama_client = ollama.Client(host=OLLAMA_BASE_URL) if ollama else None
_OLLAMA_API_BASE = OLLAMA_BASE_URL.rstrip("/")
_last_groq_error = ""
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

    for name in model_names:
        if name == OLLAMA_MODEL or name.startswith(f"{OLLAMA_MODEL}:") or name.split(":")[0] == OLLAMA_MODEL:
            return name

    for name in model_names:
        if OLLAMA_MODEL in name:
            return name

    preferred_prefixes = ("llama3", "llama3.1", "llama3.2", "llama")
    for prefix in preferred_prefixes:
        for name in model_names:
            if name.startswith(prefix):
                return name

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


def _generate_response_groq(prompt: str) -> str:
    global _last_groq_error
    if _groq_client is None:
        _last_groq_error = "GROQ_API_KEY is not configured."
        raise RuntimeError(_last_groq_error)

    def groq_request():
        response = _groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=GENERATION_TEMPERATURE,
            max_tokens=GENERATION_MAX_OUTPUT_TOKENS,
        )
        choice = response.choices[0] if response.choices else None
        message = getattr(choice, "message", None) if choice else None
        return getattr(message, "content", "") or ""

    try:
        _last_groq_error = ""
        return _with_timeout_and_retries(groq_request)
    except Exception as exc:
        _last_groq_error = str(exc)
        raise


def _generate_response_stream_groq(prompt: str):
    global _last_groq_error
    if _groq_client is None:
        _last_groq_error = "GROQ_API_KEY is not configured."
        raise RuntimeError(_last_groq_error)

    def stream_request():
        return _groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=GENERATION_TEMPERATURE,
            max_tokens=GENERATION_MAX_OUTPUT_TOKENS,
            stream=True,
        )

    try:
        _last_groq_error = ""
        stream = _with_timeout_and_retries(stream_request, max_retries=1)
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if not delta:
                continue
            text = getattr(delta, "content", None)
            if text:
                yield text
    except Exception as exc:
        _last_groq_error = str(exc)
        raise


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


def generate_response(prompt):
    if _groq_client is not None:
        try:
            return _generate_response_groq(prompt)
        except Exception:
            pass

    if _is_ollama_model_available():
        try:
            return _generate_response_ollama(prompt)
        except Exception:
            pass

    detail = _last_groq_error or _last_ollama_error or "No available LLM provider."
    raise RuntimeError(f"Groq generation failed and Ollama is unavailable. {detail}")


def generate_response_stream(prompt):
    """Yields text chunks using Groq first, then Ollama fallback."""
    if _groq_client is not None:
        try:
            yield from _generate_response_stream_groq(prompt)
            return
        except Exception:
            pass

    if _is_ollama_model_available():
        try:
            yield from _generate_response_stream_ollama(prompt)
            return
        except Exception:
            pass

    detail = _last_groq_error or _last_ollama_error or "No available LLM provider."
    raise RuntimeError(f"Groq generation failed and Ollama is unavailable. {detail}")


def get_provider_info():
    if _groq_client is not None:
        return {
            "label": "GROQ",
            "detail": "Groq primary, Ollama fallback",
            "primary": "groq",
        }

    if _is_ollama_model_available():
        return {
            "label": "OLLAMA",
            "detail": "Groq key missing, using Ollama",
            "primary": "ollama",
        }

    return {
        "label": "UNAVAILABLE",
        "detail": "No Groq key and Ollama is unavailable",
        "primary": "none",
    }