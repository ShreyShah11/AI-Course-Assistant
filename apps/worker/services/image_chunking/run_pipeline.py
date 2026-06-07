from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

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


def _load_image_pipeline_module(filename: str, module_name: str):
    if str(IMAGE_PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(IMAGE_PIPELINE_DIR))

    module_path = IMAGE_PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _add_gemini_page_metadata(response: dict[str, Any], page_results: list[Any]) -> None:
    by_page = {getattr(result, "page", index + 1): result for index, result in enumerate(page_results)}
    for page in response.get("pages", []):
        page_number = page.get("page_number")
        result = by_page.get(page_number)
        if result is None:
            continue
        page["metadata"]["gemini_summary"] = getattr(result, "gemini_summary", "")
        page["metadata"]["gemini_topic"] = getattr(result, "gemini_topic", "")
        page["metadata"]["gemini_keywords"] = getattr(result, "gemini_keywords", [])
        page["metadata"]["gemini_content_signals"] = getattr(result, "gemini_content_signals", [])


def _upsert_chunks(
    chunks: list[Any],
    vectors: list[list[float]],
    enrichments: dict[str, Any],
    target_index: str,
    target_namespace: str,
    gemini_processing,
) -> None:
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Embedding count mismatch: got {len(vectors)} vectors for {len(chunks)} chunks.")

    dimension = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    ensure_index(index_name=target_index, dimension=dimension)
    index = get_pinecone_client().Index(target_index)
    index.upsert(
        vectors=[
            {
                "id": chunk.metadata.chunk_id,
                "values": vector,
                "metadata": gemini_processing.pinecone_metadata(
                    chunk,
                    enrichments.get(chunk.metadata.chunk_id),
                ),
            }
            for chunk, vector in zip(chunks, vectors)
        ],
        namespace=target_namespace,
    )


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

    image_pipeline = _load_image_pipeline_module("image_ingestion_pipeline.py", "image_ingestion_pipeline")
    gemini_processing = _load_image_pipeline_module("gemini_processing.py", "image_gemini_processing")

    page_images = gemini_processing.image_bytes_for_pages(path, dpi=dpi)
    tesseract_results = ocr_worker.process_file_results(path, dpi=dpi)
    page_results = gemini_processing.enrich_page_results(tesseract_results, page_images)
    pages = [ocr_worker._to_page_result(result) for result in page_results]

    chunks = image_pipeline.chunk_document(
        ocr_results=page_results,
        source_file=str(path),
        course_id=course_id,
        course_name=course_name,
        subject_area=subject_area,
    )
    if not chunks:
        raise RuntimeError("Image ingestion produced no chunks.")

    vectors = gemini_processing.embed_texts([chunk.text for chunk in chunks])
    enrichments = gemini_processing.build_chunk_enrichment_map(chunks)
    _upsert_chunks(
        chunks=chunks,
        vectors=vectors,
        enrichments=enrichments,
        target_index=target_index,
        target_namespace=target_namespace,
        gemini_processing=gemini_processing,
    )

    response = ocr_worker.to_response(pages)
    _add_gemini_page_metadata(response, page_results)
    response["source"] = {
        "file_path": str(path),
        "file_type": path.suffix.lower().lstrip("."),
        "dpi": dpi,
    }
    response["chunking"] = {
        "chunk_count": len(chunks),
        "namespace": target_namespace,
        "pinecone_index": target_index,
        "chunks": image_pipeline.chunks_to_dicts(chunks),
    }
    return response
