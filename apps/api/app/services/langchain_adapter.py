from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mpeg", ".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac", ".mpga"}


def enqueue_ingestion(*, file_path: str, course_id: str, file_type: str) -> str | None:
    suffix = Path(file_path).suffix.lower()
    try:
        if file_type == "youtube":
            from apps.worker.services.youtube_chunking.worker import enqueue_job

            job = enqueue_job(url=file_path, course_id=course_id)
        elif suffix in DOCUMENT_EXTENSIONS:
            from apps.worker.services.document_chunking.worker import enqueue_job

            job = enqueue_job(file_paths=[file_path], course_id=course_id)
        elif suffix in VIDEO_EXTENSIONS:
            from apps.worker.services.audio_chunking.workers import enqueue_job

            job = enqueue_job(file_path=file_path, course_id=course_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported material type: {suffix or file_type}")
        return job.id
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not start ingestion pipeline: {exc}") from exc


def extract_sources(retrieval_payload: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = retrieval_payload.get("chunks") or []
    sources: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") if isinstance(chunk, dict) else getattr(chunk, "metadata", {})
        text = chunk.get("text") if isinstance(chunk, dict) else getattr(chunk, "text", "")
        sources.append(
            {
                "citation": index,
                "document": metadata.get("source") or metadata.get("file_name") or metadata.get("title") or "Course material",
                "page": metadata.get("page") or metadata.get("page_number"),
                "preview": (text or "")[:320],
                "metadata": metadata,
            }
        )
    return sources
