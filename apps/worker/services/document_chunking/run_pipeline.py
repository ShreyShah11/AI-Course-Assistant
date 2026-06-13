from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from apps.worker.core.config import load_environment
from apps.worker.core.pinecone import get_course_index_name


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOCUMENT_PIPELINE_DIR = (
    PROJECT_ROOT
    / "apps"
    / "api"
    / "pipelines"
    / "chunking pipeline"
    / "document pipeline"
)
PIPELINE_FILE = DOCUMENT_PIPELINE_DIR / "ingestion_pipeline_pinecone.py"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".ppt",
    ".docx",
    ".doc",
    ".txt",
    ".md",
}


def validate_file_paths(file_paths: list[str]) -> list[Path]:
    paths = []
    for file_path in file_paths:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                "Unsupported file type. Supported: "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS))
            )

        paths.append(path)

    return paths


def load_ingestion_pipeline_module():
    print("LOADER 1")

    load_environment("worker")

    print("LOADER 2")

    if str(DOCUMENT_PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(DOCUMENT_PIPELINE_DIR))

    print("LOADER 3")

    spec = importlib.util.spec_from_file_location(
        "ingestion_pipeline_pinecone",
        PIPELINE_FILE,
    )

    print("LOADER 4")

    module = importlib.util.module_from_spec(spec)

    print("LOADER 5")

    spec.loader.exec_module(module)

    print("LOADER 6")

    return module


def run_pipeline(
    file_paths: list[str],
    course_id: str,
) -> dict[str, Any]:
    paths = validate_file_paths(file_paths)
    target_index = get_course_index_name(course_id)

    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")
    print("GEMINI_API_KEY:", bool(os.getenv("GEMINI_API_KEY")))
    print("GOOGLE_API_KEY:", bool(os.getenv("GOOGLE_API_KEY")))
    print("PINECONE_API_KEY:", bool(os.getenv("PINECONE_API_KEY")))
    pipeline = load_ingestion_pipeline_module()
    target_namespace = os.getenv("DOCUMENT_CHUNKING_NAMESPACE", "document-chunks")
    print("DEBUG: About to run ingestion pipeline")
    pipeline_result = pipeline.run_pipeline(
        file_paths=[str(path) for path in paths],
        namespace=target_namespace,
        index_name=target_index,
        course_id=course_id,
    ) or {}
    print("DEBUG: Pipeline completed")
    return {
        "status": "completed",
        "file_count": len(paths),
        "files": [str(path) for path in paths],
        "namespace": target_namespace,
        "pinecone_index": pipeline_result.get("pinecone_index", target_index),
        "pipeline_result": pipeline_result,
    }
