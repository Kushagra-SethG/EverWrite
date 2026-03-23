# ✨ EverWrite

EverWrite is an interactive AI narrative game set in the world of Aethel.
Players make choices, pick a faction, choose equipment, and shape a branching story in real time.

The app uses a Flask backend, a retro-themed frontend, semantic memory with ChromaDB, and an LLM routing layer that prefers local Ollama (llama3) when available.

> 🌌 Weave your own sci-fantasy journey with real-time AI narration.

## 🚀 Highlights

- ⚡ Streaming story responses with Server-Sent Events (SSE)
- 🧭 Stateful sessions (phase, faction, equipment)
- 🧠 Semantic memory retrieval with sentence-transformer embeddings
- 📴 Offline-first LLM routing via Ollama llama3
- ☁️ Gemini fallback when local model is unavailable

## 🛠️ Tech Stack

### 🔧 Backend

- Python 3
- Flask
- Flask-CORS
- python-dotenv

### 🤖 LLM + AI

- Ollama Python client (local model inference)
- Google GenAI SDK (Gemini fallback)
- Model defaults:
  - Ollama model: llama3
  - Gemini model: gemini-3-flash-preview

### 🧠 Memory / Retrieval

- ChromaDB
- sentence-transformers (all-MiniLM-L6-v2)

### 🎨 Frontend

- HTML
- CSS
- Vanilla JavaScript
- SSE stream rendering for token-like live output

## 🗂️ Project Structure

```text
EverWrite/
├─ run.py                    # Root entry point
├─ requirements.txt
├─ backend/
│  ├─ app.py                  # Flask app + HTTP/SSE routes
│  ├─ main.py                 # CLI runner entry
│  ├─ config.py               # Env loading + model configuration
│  ├─ requirements.txt
│  ├─ game/
│  │  ├─ engine.py            # Turn processing + state transitions
│  │  ├─ prompt.py            # Prompt template + phase instructions
│  │  └─ state.py             # GameState model
│  ├─ llm/
│  │  └─ gemini.py            # Ollama-first + Gemini fallback logic
│  └─ memory/
│     └─ vector_store.py      # ChromaDB + embedding memory retrieval
├─ frontend/
│  ├─ templates/
│  │  └─ index.html           # Main game UI
│  └─ static/
│     ├─ css/
│     │  └─ style.css
│     └─ js/
│        └─ game.js           # Client state + streaming parser
└─ .env                       # Environment configuration (not in repo)
```

## ⚙️ How It Works

1. Client starts a session via POST /api/start.
2. Backend creates a GameState and streams intro narration.
3. Player sends choices via POST /api/chat.
4. Backend builds prompt from:
   - current game phase
   - selected faction/equipment
   - retrieved semantic memory context
5. Response streams back chunk-by-chunk to the UI.
6. State updates based on player choices and model output.

## 📴 Offline Capability (Ollama + Llama3)

EverWrite is configured to run offline for inference when Ollama is installed and the configured model exists locally.

### 🔀 Routing Behavior

- 1️⃣ Check whether configured Ollama model is available locally.
- 2️⃣ If available, generate responses with Ollama (local).
- 3️⃣ If not available (or Ollama errors), fallback to Gemini.
- 4️⃣ If neither local model nor Gemini API key is available, request fails with a clear runtime error.

This means you can run gameplay generation without internet by using Ollama llama3.

### 📥 Install and Prepare Ollama

1. Install Ollama: https://ollama.com/download
2. Pull model:

```bash
ollama pull llama3
```

3. Start Ollama service (if not already running).

### 🔐 Environment Setup

Create a .env file in the project root:

```env
# Optional when using only offline Ollama mode
GEMINI_API_KEY=

# Local model settings
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
```

Notes:

- ✅ Keep GEMINI_API_KEY set if you want cloud fallback.
- ✅ Leave GEMINI_API_KEY empty if you want strict local-only behavior.

## ▶️ Setup and Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python run.py
```

4. Open:

```text
http://127.0.0.1:8000
```

## 🔌 API Endpoints

- GET / -> game UI
- POST /api/start -> starts session and streams intro
- POST /api/chat -> streams turn response for player input
- GET /api/state?session_id=... -> returns current state snapshot

## 📝 Notes

- 📡 Streaming is implemented via text/event-stream responses.
- 🧳 Session storage is currently in-memory (process local).
- 🗃️ Vector memory is in-process ChromaDB client storage by default.
