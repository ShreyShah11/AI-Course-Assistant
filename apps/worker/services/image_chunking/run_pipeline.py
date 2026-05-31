from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from apps.worker.core.config import load_environment
from apps.worker.core.pinecone import ensure_index, get_course_index_name, get_pinecone_client


PROJECT_ROOT = Path(__file__).resolve().parents[4]
IMAGE_PIPELINE_DIR = (
    PROJECT_ROOT
    / "apps"
    / "api"
    / "pipelines"
    / "chunking pipeline"
    / "image pipeline"
)
CHUNKING_PIPELINE_FILE = IMAGE_PIPELINE_DIR / "image_ingestion_pipeline.py"


def load_image_chunking_pipeline_module():
    if str(IMAGE_PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(IMAGE_PIPELINE_DIR))

    spec = importlib.util.spec_from_file_location(
        "image_ingestion_pipeline",
        CHUNKING_PIPELINE_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load image chunking pipeline from {CHUNKING_PIPELINE_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("image_ingestion_pipeline", module)
    spec.loader.exec_module(module)
    return module


def _get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")
    return genai.Client(api_key=api_key)


def _embed_texts_with_gemini(texts: list[str]) -> list[list[float]]:
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    dimension = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    client = _get_gemini_client()
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


def _pinecone_metadata(chunk) -> dict[str, Any]:
    metadata = chunk.metadata.to_dict()
    metadata["text"] = chunk.text[:8_000]
    metadata["page_numbers"] = [str(page) for page in metadata.get("page_numbers", [])]
    metadata["context_window"] = metadata.get("context_window", "")[:1_000]
    metadata["ocr_warnings"] = metadata.get("ocr_warnings", [])[:20]
    metadata["preprocessing_stages"] = metadata.get("preprocessing_stages", [])[:30]
    metadata["heading_path"] = metadata.get("heading_path", [])[:20]
    metadata["keywords"] = metadata.get("keywords", [])[:30]
    metadata["content_signals"] = metadata.get("content_signals", [])[:20]
    return metadata


def run_pipeline(
    file_path: str | Path,
    ocr_worker,
    course_id: str,
    dpi: int = 300,
    course_name: str = "",
    subject_area: str = "",
) -> dict[str, Any]:
    load_environment("worker")
    path = Path(file_path)
    target_index = get_course_index_name(course_id)
    target_namespace = os.getenv("IMAGE_CHUNKING_NAMESPACE", "image-chunks")

    raw_results = ocr_worker.process_file_results(path, dpi=dpi)
    pages = [ocr_worker._to_page_result(result) for result in raw_results]

    pipeline = load_image_chunking_pipeline_module()
    chunks = pipeline.chunk_document(
        ocr_results=raw_results,
        source_file=str(path),
        course_id=course_id,
        course_name=course_name,
        subject_area=subject_area,
    )
    if not chunks:
        raise RuntimeError("Image ingestion produced no chunks.")

    vectors = _embed_texts_with_gemini([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} vectors for {len(chunks)} chunks."
        )

    dimension = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    ensure_index(index_name=target_index, dimension=dimension)
    index = get_pinecone_client().Index(target_index)
    upsert_data = [
        {
            "id": chunk.metadata.chunk_id,
            "values": vector,
            "metadata": _pinecone_metadata(chunk),
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    index.upsert(vectors=upsert_data, namespace=target_namespace)

    response = ocr_worker.to_response(pages)
    response["source"] = {
        "file_path": str(path),
        "file_type": path.suffix.lower().lstrip("."),
        "dpi": dpi,
    }
    response["chunking"] = {
        "chunk_count": len(chunks),
        "namespace": target_namespace,
        "pinecone_index": target_index,
        "chunks": pipeline.chunks_to_dicts(chunks),
    }
    return response
