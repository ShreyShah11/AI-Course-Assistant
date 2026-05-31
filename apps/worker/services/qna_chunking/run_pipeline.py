from __future__ import annotations

import importlib.util
import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from apps.worker.core.config import load_environment
from apps.worker.core.pinecone import ensure_index, get_course_index_name, get_pinecone_client


PROJECT_ROOT = Path(__file__).resolve().parents[4]
QNA_PIPELINE_DIR = (
    PROJECT_ROOT
    / "apps"
    / "api"
    / "pipelines"
    / "chunking pipeline"
    / "QnA pipeline"
)
PIPELINE_FILE = QNA_PIPELINE_DIR / "ingestion_pipeline"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".ppt"}


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
    load_environment("worker")
    loader = SourceFileLoader("qna_ingestion_pipeline", str(PIPELINE_FILE))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f"Cannot load QnA pipeline from {PIPELINE_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(loader.name, module)
    loader.exec_module(module)
    return module


def _embed_chunks_with_gemini(texts: list[str]) -> list[list[float]]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    dimension = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    client = genai.Client(api_key=api_key)
    vectors = []
    for text in texts:
        result = client.models.embed_content(
            model=model,
            contents=text if text.strip() else " ",
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=dimension,
            ),
        )
        vectors.append(result.embeddings[0].values)
    return vectors


def _pinecone_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(chunk)
    metadata["page_range"] = [str(page) for page in metadata.get("page_range", [])]
    metadata["raw_text"] = metadata.get("raw_text", "")[:8_000]
    metadata["questions"] = metadata.get("questions", [])[:30]
    metadata["answers"] = metadata.get("answers", [])[:30]
    metadata["marks_hint"] = metadata.get("marks_hint") or -1
    metadata["year_hint"] = metadata.get("year_hint") or -1
    return metadata


def run_pipeline(
    file_paths: list[str],
    course_id: str,
    semantic_threshold: float = 0.45,
    max_chunk_tokens: int = 800,
) -> dict[str, Any]:
    load_environment("worker")
    paths = validate_file_paths(file_paths)
    target_index = get_course_index_name(course_id)
    target_namespace = os.getenv("QNA_CHUNKING_NAMESPACE", "qna-chunks")
    pipeline = load_ingestion_pipeline_module()

    chunks = []
    for path in paths:
        chunks.extend(
            pipeline.process_document(
                file_path=str(path),
                semantic_threshold=semantic_threshold,
                max_chunk_tokens=max_chunk_tokens,
            )
        )
    if not chunks:
        raise RuntimeError("QnA ingestion produced no chunks.")

    vectors = _embed_chunks_with_gemini([chunk["raw_text"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} vectors for {len(chunks)} chunks."
        )

    dimension = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    ensure_index(index_name=target_index, dimension=dimension)
    index = get_pinecone_client().Index(target_index)
    upsert_data = [
        {
            "id": chunk["chunk_id"],
            "values": vector,
            "metadata": {**_pinecone_metadata(chunk), "course_id": course_id},
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    index.upsert(vectors=upsert_data, namespace=target_namespace)

    return {
        "status": "completed",
        "file_count": len(paths),
        "files": [str(path) for path in paths],
        "chunk_count": len(chunks),
        "namespace": target_namespace,
        "pinecone_index": target_index,
        "chunks": chunks,
    }
