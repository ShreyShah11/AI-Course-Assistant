from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from apps.worker.core.config import load_environment

load_environment("api")
import os

print("GEMINI_API_KEY =", os.getenv("GEMINI_API_KEY"))
print("GOOGLE_API_KEY =", os.getenv("GOOGLE_API_KEY"))

from routes.agentic_retrieval import router as agentic_retrieval_router
from routes.audio_chunking_jobs import router as audio_chunking_jobs_router
from routes.document_chunking_jobs import router as document_chunking_jobs_router
from routes.handwritten_chunking_jobs import router as handwritten_chunking_jobs_router
from routes.image_chunking_jobs import router as image_chunking_jobs_router
from routes.qna_chunking_jobs import router as qna_chunking_jobs_router
from routes.temp_chunk_preview import router as temp_chunk_preview_router
from routes.youtube_chunking_jobs import router as youtube_chunking_jobs_router
from routes.evals import router as evals_router
from app.routers.auth import router as coursegpt_auth_router
from app.routers.chat import router as coursegpt_chat_router
from app.routers.courses import router as coursegpt_courses_router
from app.routers.materials import router as coursegpt_materials_router
from app.routers.quiz import router as coursegpt_quiz_router

try:
    from routes.image_processing import router as image_processing_router
except ImportError:
    image_processing_router = None

app = FastAPI(title="CourseGPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agentic_retrieval_router)
app.include_router(audio_chunking_jobs_router)
app.include_router(document_chunking_jobs_router)
app.include_router(handwritten_chunking_jobs_router)
app.include_router(image_chunking_jobs_router)
app.include_router(qna_chunking_jobs_router)
app.include_router(youtube_chunking_jobs_router)
app.include_router(coursegpt_auth_router)
app.include_router(coursegpt_courses_router)
app.include_router(coursegpt_materials_router)
app.include_router(coursegpt_chat_router)
app.include_router(coursegpt_quiz_router)
# TEMPORARY: delete this include and the import above when chunk preview is removed.
app.include_router(temp_chunk_preview_router)
if image_processing_router is not None:
    app.include_router(image_processing_router)


# LOCAL EVAL — hidden from OpenAPI docs, for local use only.
# Hit POST /internal/evals/run from Postman or curl to run the evaluation suite.
app.include_router(evals_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
