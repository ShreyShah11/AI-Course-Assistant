from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from apps.worker.core.config import load_environment
from apps.worker.core.pinecone import get_course_index_name


PROJECT_ROOT = Path(__file__).resolve().parents[4]
YOUTUBE_PIPELINE_DIR = (
    PROJECT_ROOT / "apps" / "api" / "pipelines" / "chunking pipeline" / "youtube pipeline"
)
PIPELINE_FILE = YOUTUBE_PIPELINE_DIR / "ingestion_pipeline.py"


def load_ingestion_pipeline_module():
    load_environment("worker")
    if str(YOUTUBE_PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(YOUTUBE_PIPELINE_DIR))
    spec = importlib.util.spec_from_file_location("youtube_ingestion_pipeline", PIPELINE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load YouTube pipeline from {PIPELINE_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_pipeline(
    url: str,
    course_id: str,
    course_name: str = "",
    instructor: str = "",
    semester: str = "",
    subject: str = "",
    tags: list[str] | None = None,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    load_environment("worker")
    pipeline = load_ingestion_pipeline_module()
    index_name = get_course_index_name(course_id)
    namespace = os.getenv("YOUTUBE_CHUNKING_NAMESPACE", "youtube-chunks")
    tags = [tag for tag in (tags or []) if tag and tag != "string"]
    languages = [lang for lang in (languages or ["en"]) if lang and lang != "string"] or ["en"]
    chunks = pipeline.ingest_youtube_video(
        url=url,
        course_id=course_id,
        course_name=course_name,
        instructor=instructor,
        semester=semester,
        subject=subject,
        tags=tags,
        languages=languages,
        index_name=index_name,
        namespace=namespace,
    )
    return {
        "status": "completed" if chunks else "completed_no_chunks",
        "url": url,
        "course_id": course_id,
        "chunk_count": len(chunks),
        "namespace": namespace,
        "pinecone_index": index_name,
        "chunks": [chunk.__dict__ for chunk in chunks],
    }
