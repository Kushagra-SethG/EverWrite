import os
from dotenv import load_dotenv

# Load .env from the project root (one level above backend/)
_ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(_ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3-flash-preview"
TOP_K = 5