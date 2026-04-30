import os
from dotenv import load_dotenv

# Load .env from the project root (one level above backend/)
_ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(_ENV_PATH)


def _to_bool(value: str | None, default: bool) -> bool:
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
	try:
		return int(value) if value is not None else default
	except ValueError:
		return default


def _to_float(value: str | None, default: float) -> float:
	try:
		return float(value) if value is not None else default
	except ValueError:
		return default


def _to_list(value: str | None) -> list[str]:
	if not value:
		return []
	return [item.strip() for item in value.split(",") if item.strip()]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_TEMPERATURE = _to_float(os.getenv("GENERATION_TEMPERATURE"), 0.7)
GENERATION_MAX_OUTPUT_TOKENS = _to_int(os.getenv("GENERATION_MAX_OUTPUT_TOKENS"), 15000)
MODEL_REQUEST_TIMEOUT_SECONDS = _to_float(os.getenv("MODEL_REQUEST_TIMEOUT_SECONDS"), 60.0)
MODEL_REQUEST_MAX_RETRIES = _to_int(os.getenv("MODEL_REQUEST_MAX_RETRIES"), 2)
MODEL_RETRY_BACKOFF_SECONDS = _to_float(os.getenv("MODEL_RETRY_BACKOFF_SECONDS"), 1.0)
OLLAMA_MODEL_CHECK_TIMEOUT_SECONDS = _to_float(os.getenv("OLLAMA_MODEL_CHECK_TIMEOUT_SECONDS"), 5.0)

TOP_K = _to_int(os.getenv("TOP_K"), 5)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.path.abspath(
	os.getenv("CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
)

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "everwrite-dev-secret")
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = _to_int(os.getenv("FLASK_PORT"), 8000)
FLASK_DEBUG = _to_bool(os.getenv("FLASK_DEBUG"), True)
FRONTEND_API_BASE_URL = os.getenv("FRONTEND_API_BASE_URL", "")

_raw_cors_origins = _to_list(os.getenv("CORS_ORIGINS"))
CORS_ORIGINS: list[str] | str = _raw_cors_origins if _raw_cors_origins else "*"