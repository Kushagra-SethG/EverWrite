import os
import sys
import uuid
import json
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS

# Ensure backend package imports resolve whether run from root or backend/
sys.path.insert(0, os.path.dirname(__file__))

from game.state import GameState
from game.engine import process_turn_stream
from game.prompt import build_prompt
from llm.gemini import generate_response_stream
from memory.vector_store import get_memory, add_memory

_BASE   = os.path.dirname(__file__)           # .../backend
_FRONT  = os.path.join(_BASE, '..', 'frontend')

app = Flask(
    __name__,
    template_folder=os.path.join(_FRONT, 'templates'),
    static_folder=os.path.join(_FRONT, 'static'),
)
app.secret_key = "everwrite-retro-2026"
CORS(app)

# In-memory session store: { session_id: GameState }
sessions: dict[str, GameState] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_game():
    """
    Create a brand-new game session and stream the intro message back.
    Response is SSE: chunks of text, then a final event with session_id + state.
    """
    session_id = str(uuid.uuid4())
    state = GameState()
    sessions[session_id] = state

    # Build the intro prompt (empty user input, intro phase)
    memory = get_memory("[START]")
    prompt = build_prompt("[START]", memory, state)

    def generate():
        full_response = ""
        try:
            for chunk in generate_response_stream(prompt):
                full_response += chunk
                yield _sse({"chunk": chunk, "done": False})
        except Exception as exc:
            yield _sse({"error": str(exc), "done": True, "session_id": session_id,
                        "state": {"phase": state.phase, "faction": state.faction, "equipment": state.equipment}})
            return

        add_memory(f"AI: {full_response}")

        yield _sse({
            "done": True,
            "session_id": session_id,
            "state": {
                "phase": state.phase,
                "faction": state.faction,
                "equipment": state.equipment,
            },
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Accept a player message and stream the AI narrative response.
    Body JSON: { session_id, message }
    Response is SSE: text chunks, then final event with updated state.
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()
    user_input = data.get("message", "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or expired session. Start a new game."}), 400

    if not user_input:
        return jsonify({"error": "Empty message."}), 400

    state = sessions[session_id]

    if state.phase == "ended":
        return jsonify({"error": "The journey has ended. Start a new game."}), 400

    def generate():
        try:
            for chunk, is_done, state_dict in process_turn_stream(user_input, state):
                if not is_done:
                    yield _sse({"chunk": chunk, "done": False})
                else:
                    yield _sse({"done": True, "state": state_dict})
        except Exception as exc:
            yield _sse({"error": str(exc), "done": True,
                        "state": {"phase": state.phase, "faction": state.faction, "equipment": state.equipment}})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/state", methods=["GET"])
def get_state():
    session_id = request.args.get("session_id", "")
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400
    state = sessions[session_id]
    return jsonify({
        "phase": state.phase,
        "faction": state.faction,
        "equipment": state.equipment,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
