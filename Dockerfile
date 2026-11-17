# Pharma RAG — one container serving the FastAPI backend + the built React
# frontend (FastAPI serves frontend/dist from /; API routes take precedence).
# Multi-arch safe: python:3.12-slim and node:22-alpine both ship ARM64 images,
# and every Python dependency has ARM64 wheels. No keys are baked in —
# configure the LLM at runtime with environment variables (see .env.example).
#
#   docker build -t pharma-rag .
#   docker run -p 8000:8000 -v pharma-data:/app/data --env-file .env pharma-rag
#   # pre-index the demo drugs (from the host, once the container is up):
#   curl -X POST localhost:8000/index -H 'Content-Type: application/json' \
#     -d '{"drugs": ["atorvastatin", "metformin", "lisinopril"]}'
#
# The /app/data volume holds the ChromaDB index (data/chroma) and the
# DailyMed SPL cache (data/spl_cache) — persist it or the index is rebuilt
# from scratch on restart. (Embeddings go through the NVIDIA API now, so
# there are no downloaded models to persist.)
# Vector store: local ChromaDB by default; set QDRANT_URL + QDRANT_API_KEY
# in the env file to use Qdrant Cloud instead (collection `pharma_rag`).
#
# UPGRADING from a version with local embeddings: the old ChromaDB index is
# incompatible (384-dim local vectors vs 4096-dim API vectors) — clear
# /app/data/chroma in the volume (or drop the pharma-data volume) before
# starting the new image, then re-index via POST /index (see below).

# ---- Stage 1: build the React frontend -------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime -----------------------------------------------
FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    CHROMA_DIR=/app/data/chroma \
    SPL_CACHE_DIR=/app/data/spl_cache
WORKDIR /app

# Runtime deps only (no streamlit/pytest — the container serves the React UI).
COPY requirements-docker.txt ./
RUN pip install -r requirements-docker.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY --from=frontend /build/dist ./frontend/dist

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

VOLUME ["/app/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health')"
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
