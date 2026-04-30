"""
EverWrite — root entry point.
Run from the project root:  python run.py
"""
import sys
import os

# Put backend/ on the path so flat imports (game, llm, memory, config) resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app import app          # noqa: E402  (app.py lives in backend/)
from config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT  # noqa: E402

if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT, threaded=True)
