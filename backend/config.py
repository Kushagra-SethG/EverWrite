import os
from dotenv import load_dotenv

# Load .env from the project root (one level above backend/)
_ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(_ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3-flash-preview"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TOP_K = 5