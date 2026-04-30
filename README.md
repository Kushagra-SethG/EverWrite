# EverWrite

EverWrite is an AI-powered interactive fantasy game set in Aethel. You play as a reincarnated character, choose a faction, pick equipment, and shape the narrative through free-form actions.

The project uses a Flask backend for game logic + streaming, a React/Vite frontend for UI, ChromaDB for semantic memory retrieval, and an LLM routing layer that prefers Groq and falls back to local Ollama.

## Features

- SSE streaming responses for live narrative output
- Stateful progression across phases (`name` -> `intro` -> `equipment` -> `story`)
- Faction-aware gameplay and consequence parsing
- Semantic memory retrieval with sentence-transformer embeddings + ChromaDB
- Groq-first inference with Ollama fallback

## Tech Stack

- Backend: Python, Flask, Flask-CORS, python-dotenv
- Frontend: React 18, Vite 5
- LLM: Groq SDK + Ollama client
- Retrieval: ChromaDB + sentence-transformers

## Project Structure

```text
EverWrite/
├─ run.py
├─ backend/
│  ├─ app.py                 # Flask app + API routes + SSE responses
│  ├─ config.py              # Environment config and defaults
│  ├─ main.py                # CLI game runner
│  ├─ requirements.txt
│  ├─ game/
│  │  ├─ engine.py           # Turn processing + consequence parsing
│  │  ├─ prompt.py           # Prompt construction + faction lore
│  │  └─ state.py            # Game state model
│  ├─ llm/
│  │  └─ groq.py             # Groq-first generation + Ollama fallback
│  └─ memory/
│     └─ vector_store.py     # ChromaDB memory read/write
├─ frontend/
│  ├─ src/                   # React application
│  ├─ index.html             # Vite entry HTML
│  ├─ vite.config.js         # Dev proxy (/api -> Flask)
│  ├─ templates/index.html   # Legacy template-based UI
│  └─ static/                # Legacy static assets
└─ chroma_db/                # Persistent vector store data
```

## Runtime Flow

1. Client starts a session with `POST /api/start`.
2. Backend creates `GameState`, builds prompt context (state + memory), and streams response chunks.
3. Client sends actions to `POST /api/chat`.
4. Backend streams narrative text, parses optional `[CONSEQUENCE]` metadata, and updates state.
5. Updated state is returned at stream completion.

## Environment Variables

Create a `.env` file at the project root.

```env
# Flask
FLASK_HOST=127.0.0.1
FLASK_PORT=8000
FLASK_DEBUG=true
FLASK_SECRET_KEY=replace-with-a-secret
CORS_ORIGINS=
FRONTEND_API_BASE_URL=

# LLM
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
GENERATION_TEMPERATURE=0.7
GENERATION_MAX_OUTPUT_TOKENS=15000
MODEL_REQUEST_TIMEOUT_SECONDS=60
MODEL_REQUEST_MAX_RETRIES=2
MODEL_RETRY_BACKOFF_SECONDS=1
OLLAMA_MODEL_CHECK_TIMEOUT_SECONDS=5

# Memory / embeddings
TOP_K=5
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=chroma_db
```

Notes:

- If `GROQ_API_KEY` is empty, the app falls back to Ollama when available.
- If both Groq and local Ollama are unavailable, generation requests fail.

## Setup

### 1) Backend setup

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
```

### 2) Frontend setup

```bash
cd frontend
npm install
```

## Run Modes

### Mode A: Full local development (recommended)

Run backend and frontend in separate terminals.

Terminal 1:

```bash
python run.py
```

Terminal 2:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`.

Vite proxies `/api` requests to Flask (`http://127.0.0.1:8000`).

### Mode B: Serve built frontend from Flask

Build the frontend first:

```bash
cd frontend
npm run build
```

Then run backend:

```bash
python run.py
```

Open `http://127.0.0.1:8000`.

If `frontend/dist` is missing, the root route returns a helpful message instead of UI.

## API Endpoints

- `GET /` -> Serves built frontend (`frontend/dist/index.html`) when available
- `GET /assets/<path>` -> Serves built frontend assets
- `POST /api/start` -> Creates a new session and streams intro response
- `POST /api/chat` -> Streams turn response for a session
- `GET /api/state?session_id=...` -> Returns current state snapshot
- `GET /api/factions` -> Returns faction metadata

## LLM Routing Behavior

Generation path:

1. Try Groq first using the configured API key and model.
2. Fall back to Ollama generation/streaming when Groq is unavailable or errors.
3. Use the local Ollama path for offline or self-hosted runs.

This enables offline-first gameplay when Ollama is set up locally.

## Current Limitations

- Session state is in-memory (process-local).
- No authentication/multi-tenant persistence layer.
- ChromaDB data is local to this project path unless configured otherwise.
