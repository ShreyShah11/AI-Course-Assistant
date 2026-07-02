# ─────────────────────────────────────────────────────────────────────────────
# Shared Monorepo Dockerfile  (used by BOTH the API and the Worker services)
#
#  • API service  → default CMD: alembic upgrade head && uvicorn ...
#  • Worker service → render.yaml overrides CMD via dockerCommand:
#                      python -m apps.worker.run_all_workers
#
# Only ONE requirements.txt is needed (apps/api/requirements.txt).
# It already contains rq, redis, torch, pinecone, etc. — everything the
# worker pipelines need — so the worker venv is the API venv.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies (shared by both API and worker pipelines) ─────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OCR  — pytesseract / unstructured
    tesseract-ocr \
    tesseract-ocr-eng \
    # PDF-to-image — pdf2image
    poppler-utils \
    # Audio transcription — openai-whisper
    ffmpeg \
    # OpenCV runtime
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Build tools (psycopg2, some C extensions)
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Monorepo root is the working directory ────────────────────────────────────
WORKDIR /app

# ── Install Python deps from the API venv only (covers worker too) ───────────
# Copy requirements first for Docker layer cache — image won't rebuild this
# layer unless requirements.txt changes.
COPY apps/api/requirements.txt /app/apps/api/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r apps/api/requirements.txt

# ── Copy the full monorepo source tree ───────────────────────────────────────
# `apps.worker.*` imports and `apps.api.*` pipeline imports both resolve from /app
COPY . .

# ── Python environment ────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# /app on PYTHONPATH lets both `apps.api.*` and `apps.worker.*` imports work
ENV PYTHONPATH=/app

# ── Default CMD: API service ──────────────────────────────────────────────────
# Worker service overrides this in render.yaml via `dockerCommand`.
WORKDIR /app/apps/api
CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port $PORT"]
