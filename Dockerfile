FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=7860 \
    FLASK_DEBUG=false \
    PORT=7860 \
    CHROMA_PERSIST_DIR=/app/chroma_db

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY run.py ./run.py
RUN mkdir -p /app/chroma_db
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 7860

CMD ["python", "run.py"]