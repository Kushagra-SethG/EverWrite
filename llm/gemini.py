import google.genai as genai

from config import GEMINI_API_KEY, MODEL_NAME, OLLAMA_BASE_URL, OLLAMA_MODEL

try:
    import ollama
except ImportError:  # pragma: no cover - environment dependent
    ollama = None


_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
_ollama_client = ollama.Client(host=OLLAMA_BASE_URL) if ollama else None


def _is_ollama_model_available() -> bool:
    if _ollama_client is None:
        return False

    try:
        tags = _ollama_client.list()
        models = tags.get("models", []) if isinstance(tags, dict) else getattr(tags, "models", [])
        model_names = []

        for model in models:
            if isinstance(model, dict):
                name = model.get("name") or model.get("model")
            else:
                name = getattr(model, "name", None) or getattr(model, "model", None)
            if name:
                model_names.append(name)

        return any(name == OLLAMA_MODEL or name.startswith(f"{OLLAMA_MODEL}:") for name in model_names)
    except Exception:
        return False


def _generate_response_ollama(prompt: str) -> str:
    response = _ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.7,
        },
    )
    message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", None)
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "") or ""


def _generate_response_stream_ollama(prompt: str):
    stream = _ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={
            "temperature": 0.7,
        },
    )
    for chunk in stream:
        message = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", None)
        if isinstance(message, dict):
            text = message.get("content")
        else:
            text = getattr(message, "content", None)
        if text:
            yield text


def _generate_response_gemini(prompt: str) -> str:
    if _gemini_client is None:
        raise RuntimeError("Gemini API key is not configured and Ollama llama3 is unavailable.")

    response = _gemini_client.models.generate_content(
        config={
            "temperature": 0.7,
            "maxOutputTokens": 1500,
            "system_instruction": prompt,
        },
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text


def _generate_response_stream_gemini(prompt: str):
    if _gemini_client is None:
        raise RuntimeError("Gemini API key is not configured and Ollama llama3 is unavailable.")

    for chunk in _gemini_client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": 0.7,
            "maxOutputTokens": 1500,
            "system_instruction": prompt,
        },
    ):
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