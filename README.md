---
title: EverWrite
emoji: 🎮
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Groq-first fantasy RPG with Ollama fallback
---

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

## Deploy on Hugging Face Spaces

This repository is ready for a Docker-based Hugging Face Space. The container builds the React frontend, installs the Python backend, and starts the Flask app on port `7860`.

### 1) Prepare the repository

Make sure these files exist at the repository root:

- `Dockerfile`
- `.dockerignore`

The app will start from `python run.py`, so the Docker image must keep `run.py`, `backend/`, and the built `frontend/dist/` folder.

### 2) Create the Space

1. Go to Hugging Face and create a new Space.
2. Choose `Docker` as the Space SDK.
3. Connect the GitHub repository or upload the source.
4. Keep the Space public or private depending on your preference.

### 3) Add secrets

Set the following secret variables in the Space settings:

- `GROQ_API_KEY` - required if you want Groq to be the primary model provider.
- `GROQ_MODEL` - optional, default is `llama-3.3-70b-versatile`.
- `OLLAMA_BASE_URL` - optional, only if you run Ollama somewhere reachable from the Space.
- `OLLAMA_MODEL` - optional, default is `llama3`.

Recommended setup for Spaces:

- Use `GROQ_API_KEY` so the app can run in the hosted environment.
- Treat Ollama as a fallback for local development, because Hugging Face Spaces does not include Ollama by default.

### 4) Build and run behavior

The Dockerfile does the following:

1. Installs frontend dependencies with `npm ci`.
2. Builds the React app with `npm run build`.
3. Installs Python backend dependencies.
4. Copies the built frontend into the runtime image.
5. Starts the app on `0.0.0.0:7860`.

### 5) First launch expectations

- The first build can take a while because Python and frontend dependencies must be installed.
- `sentence-transformers` may download its embedding model on the first request.
- If `GROQ_API_KEY` is missing and Ollama is not reachable, the app will fail to generate responses.

### 6) Persistent memory

If you want the ChromaDB memory to survive restarts, enable persistent storage for the Space and keep `chroma_db/` in the mounted storage path.

### 7) Local verification before pushing

Build the Docker image locally first if you want to verify the Space setup:

```bash
docker build -t everwrite-space .
docker run -p 7860:7860 --env GROQ_API_KEY=your_key everwrite-space
```

Then open `http://localhost:7860`.

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
