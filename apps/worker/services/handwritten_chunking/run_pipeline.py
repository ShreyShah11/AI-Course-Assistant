from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from apps.worker.core.config import load_environment
from apps.worker.core.pinecone import get_course_index_name


PROJECT_ROOT = Path(__file__).resolve().parents[4]
HANDWRITTEN_PIPELINE_DIR = (
    PROJECT_ROOT / "apps" / "api" / "pipelines" / "chunking pipeline" / "handwritten pipeline"
)
PIPELINE_FILE = HANDWRITTEN_PIPELINE_DIR / "ingestion_pipeline.py"
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def validate_file_path(file_path: str) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
    return path


def load_ingestion_pipeline_module():
    load_environment("worker")
    if str(HANDWRITTEN_PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(HANDWRITTEN_PIPELINE_DIR))
    spec = importlib.util.spec_from_file_location("handwritten_ingestion_pipeline", PIPELINE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load handwritten pipeline from {PIPELINE_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_pipeline(
    file_path: str,
    course_id: str,
    course_name: str = "",
    subject: str = "",
    instructor: str = "",
    semester: str = "",
    university: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    load_environment("worker")
    path = validate_file_path(file_path)
    pipeline = load_ingestion_pipeline_module()
    index_name = get_course_index_name(course_id)
    namespace = os.getenv("HANDWRITTEN_CHUNKING_NAMESPACE", "handwritten-chunks")
    tags = [tag for tag in (tags or []) if tag and tag != "string"]
    pipeline.cfg.pinecone_index_name = index_name
    pipeline.cfg.pinecone_namespace = namespace
    pipeline.cfg.pinecone_dimension = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    pipeline.cfg.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    result = pipeline.run_pipeline(
        filepath=str(path),
        course_id=course_id,
        course_name=course_name,
        subject=subject,
        instructor=instructor,
        semester=semester,
        university=university,
        tags=tags,
    )
    if result.errors:
        raise RuntimeError("; ".join(result.errors))
    return {
        "status": "completed",
        "file_path": str(path),
        "course_id": course_id,
        "chunk_count": len(result.chunks),
        "upserted_count": result.upserted_count,
        "namespace": namespace,
        "pinecone_index": index_name,
        "chunks": [chunk.metadata() | {"text": chunk.text} for chunk in result.chunks],
    }
