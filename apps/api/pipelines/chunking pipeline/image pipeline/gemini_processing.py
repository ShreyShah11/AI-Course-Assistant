"""
Gemini processing for the image pipeline.

Pages with low Tesseract confidence are sent to Gemini with the image bytes.
Pages with good confidence are sent to Gemini with Tesseract text. Both paths
return the same Pydantic-validated metadata shape.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class GeminiPageMetadata(BaseModel):
    text: str = ""
    confidence: float = Field(0.75, ge=0.0, le=1.0)
    summary: str = ""
    topic: str = ""
    keywords: list[str] = Field(default_factory=list)
    content_signals: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GeminiChunkMetadata(BaseModel):
    summary: str = ""
    topic: str = ""
    keywords: list[str] = Field(default_factory=list)
    content_signals: list[str] = Field(default_factory=list)


def low_confidence_threshold() -> float:
    return float(os.getenv("IMAGE_LOW_CONFIDENCE_THRESHOLD", "80"))


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")
    return genai.Client(api_key=api_key)


def gemini_retry_count() -> int:
    return int(os.getenv("IMAGE_GEMINI_RETRIES", "3"))


def gemini_retry_sleep(attempt: int) -> float:
    base_sleep = float(os.getenv("IMAGE_GEMINI_RETRY_SLEEP_SECONDS", "2"))
    return base_sleep * attempt


def gemini_fallback_model() -> str:
    return os.getenv("IMAGE_GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")


def _is_closed_client_error(exc: Exception) -> bool:
    return "client has been closed" in str(exc).lower()


def _is_retryable_gemini_error(exc: Exception) -> bool:
    if _is_closed_client_error(exc):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in {429, 500, 502, 503, 504}


def _run_gemini_request(
    request_fn: Callable[[genai.Client, str], Any],
    model: str,
    *,
    fallback_model: str | None = None,
) -> Any:
    """
    Gemini's HTTP client can be closed by the SDK after transient failures.
    Create a fresh client on every retry so long PDF jobs do not die mid-page.
    """
    attempts = max(1, gemini_retry_count())
    last_exc: Exception | None = None
    models = [model]
    fallback = fallback_model or gemini_fallback_model()
    if fallback and fallback != model:
        models.append(fallback)

    for model_name in models:
        for attempt in range(1, attempts + 1):
            client = get_gemini_client()
            try:
                return request_fn(client, model_name)
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_gemini_error(exc):
                    raise
                if attempt < attempts:
                    time.sleep(gemini_retry_sleep(attempt))
                    continue
                break

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Gemini request failed before it was sent.")


def extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def image_mime_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(path.suffix.lower(), "image/png")


def image_bytes_for_pages(path: Path, dpi: int) -> dict[int, tuple[bytes, str]]:
    if path.suffix.lower() != ".pdf":
        return {1: (path.read_bytes(), image_mime_type(path))}

    import fitz

    pages: dict[int, tuple[bytes, str]] = {}
    doc = fitz.open(path)
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    for page_number, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        pages[page_number] = (pix.tobytes("png"), "image/png")
    doc.close()
    return pages


def analyze_low_confidence_image(
    image_bytes: bytes,
    mime_type: str,
    page_number: int,
) -> GeminiPageMetadata:
    model = os.getenv("IMAGE_GEMINI_OCR_MODEL", "gemini-2.5-flash")
    prompt = (
        "Read this academic page carefully. Return JSON with text, confidence, "
        "summary, topic, keywords, content_signals, and warnings. Preserve "
        "tables, equations, code, headings, and lists in plain text or markdown."
    )
    response = _run_gemini_request(
        lambda client, model_name: client.models.generate_content(
            model=model_name,
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiPageMetadata,
                temperature=0.1,
            ),
        ),
        model,
    )
    parsed = getattr(response, "parsed", None)
    metadata = parsed if isinstance(parsed, GeminiPageMetadata) else GeminiPageMetadata.model_validate(extract_json(response.text))
    metadata.warnings.append(f"Gemini image OCR used on page {page_number}.")
    return metadata


def analyze_good_confidence_text(text: str, page_number: int) -> GeminiPageMetadata:
    model = os.getenv("IMAGE_GEMINI_TEXT_MODEL", os.getenv("IMAGE_GEMINI_ENRICH_MODEL", "gemini-2.5-flash"))
    prompt = (
        "Analyze this academic OCR text. Return JSON with text, confidence, "
        "summary, topic, keywords, content_signals, and warnings. Keep text as "
        "the cleaned OCR text unless obvious OCR artifacts should be corrected.\n\n"
        f"PAGE {page_number} OCR TEXT:\n{text[:12000]}"
    )
    response = _run_gemini_request(
        lambda client, model_name: client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiPageMetadata,
                temperature=0.1,
            ),
        ),
        model,
    )
    parsed = getattr(response, "parsed", None)
    metadata = parsed if isinstance(parsed, GeminiPageMetadata) else GeminiPageMetadata.model_validate(extract_json(response.text))
    if not metadata.text.strip():
        metadata.text = text
    metadata.warnings.append(f"Gemini text metadata used on page {page_number}.")
    return metadata


def result_from_gemini_page(
    original_result: Any,
    metadata: GeminiPageMetadata,
    source: str,
) -> Any:
    confidence = round(metadata.confidence * 100, 2)
    page = getattr(original_result, "page", 1)
    text = metadata.text or getattr(original_result, "full_text", "")
    block = SimpleNamespace(
        block_id=1,
        block_type=SimpleNamespace(value="paragraph"),
        text=text,
        raw_text=text,
        confidence=confidence,
        page=page,
        bbox=(0, 0, 0, 0),
        language=None,
        is_code=False,
        line_count=max(1, text.count("\n") + 1),
    )
    return SimpleNamespace(
        page=page,
        blocks=[block] if text.strip() else [],
        full_text=text,
        code_blocks=[],
        mean_confidence=confidence,
        image_type=getattr(original_result, "image_type", source),
        skew_corrected=getattr(original_result, "skew_corrected", False),
        skew_angle=getattr(original_result, "skew_angle", 0.0),
        tesseract_config=getattr(original_result, "tesseract_config", ""),
        stages_applied=[
            *list(getattr(original_result, "stages_applied", []) or []),
            source,
        ],
        warnings=[
            *list(getattr(original_result, "warnings", []) or []),
            *metadata.warnings,
        ],
        gemini_summary=metadata.summary,
        gemini_topic=metadata.topic,
        gemini_keywords=metadata.keywords,
        gemini_content_signals=metadata.content_signals,
    )


def enrich_page_results(
    raw_results: list[Any],
    page_images: dict[int, tuple[bytes, str]],
) -> list[Any]:
    output = []
    threshold = low_confidence_threshold()
    for result in raw_results:
        page = getattr(result, "page", 1)
        text = getattr(result, "full_text", "")
        if getattr(result, "mean_confidence", 0.0) < threshold and page in page_images:
            image_bytes, mime_type = page_images[page]
            try:
                metadata = analyze_low_confidence_image(image_bytes, mime_type, page)
            except Exception as exc:
                metadata = GeminiPageMetadata(
                    text=text,
                    confidence=max(0.0, min(1.0, getattr(result, "mean_confidence", 0.0) / 100)),
                    warnings=[f"Gemini image OCR failed on page {page}: {exc}"],
                )
            output.append(result_from_gemini_page(result, metadata, "gemini_image_ocr"))
        else:
            try:
                metadata = analyze_good_confidence_text(text, page)
            except Exception as exc:
                metadata = GeminiPageMetadata(
                    text=text,
                    confidence=max(0.0, min(1.0, getattr(result, "mean_confidence", 0.0) / 100)),
                    warnings=[f"Gemini text metadata failed on page {page}: {exc}"],
                )
            output.append(result_from_gemini_page(result, metadata, "gemini_text_metadata"))
    return output


def enrich_chunk_with_gemini(text: str) -> GeminiChunkMetadata:
    model = os.getenv("IMAGE_GEMINI_ENRICH_MODEL", "gemini-2.5-flash")
    prompt = (
        "Summarize this academic OCR chunk and extract retrieval metadata. "
        "Return JSON with summary, topic, keywords, and content_signals.\n\n"
        f"CHUNK:\n{text[:6000]}"
    )
    response = _run_gemini_request(
        lambda client, model_name: client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiChunkMetadata,
                temperature=0.1,
            ),
        ),
        model,
    )
    parsed = getattr(response, "parsed", None)
    return parsed if isinstance(parsed, GeminiChunkMetadata) else GeminiChunkMetadata.model_validate(extract_json(response.text))


def build_chunk_enrichment_map(chunks: list[Any]) -> dict[str, GeminiChunkMetadata]:
    if os.getenv("ENABLE_IMAGE_SUMMARIES", "true").lower() not in {"1", "true", "yes"}:
        return {}

    enrichments: dict[str, GeminiChunkMetadata] = {}
    for chunk in chunks:
        try:
            enrichments[chunk.metadata.chunk_id] = enrich_chunk_with_gemini(chunk.text)
        except Exception as exc:
            enrichments[chunk.metadata.chunk_id] = GeminiChunkMetadata()
            chunk.metadata.ocr_warnings.append(f"Gemini chunk enrichment failed: {exc}")
    return enrichments


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    dimension = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    vectors: list[list[float]] = []

    for text in texts:
        response = _run_gemini_request(
            lambda client, model_name: client.models.embed_content(
                model=model_name,
                contents=text if text.strip() else " ",
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=dimension,
                ),
            ),
            model,
            fallback_model=model,
        )
        vectors.append(response.embeddings[0].values)
    return vectors


def pinecone_metadata(chunk: Any, enrichment: GeminiChunkMetadata | None = None) -> dict[str, Any]:
    metadata = chunk.metadata.to_dict()
    metadata["text"] = chunk.text[:8_000]
    metadata["page_numbers"] = [str(page) for page in metadata.get("page_numbers", [])]
    metadata["context_window"] = metadata.get("context_window", "")[:1_000]
    metadata["ocr_warnings"] = metadata.get("ocr_warnings", [])[:20]
    metadata["preprocessing_stages"] = metadata.get("preprocessing_stages", [])[:30]
    metadata["heading_path"] = metadata.get("heading_path", [])[:20]

    if enrichment:
        metadata["gemini_summary"] = enrichment.summary[:1_000]
        metadata["gemini_topic"] = enrichment.topic[:200]
        metadata["gemini_keywords"] = enrichment.keywords[:30]
        metadata["gemini_content_signals"] = enrichment.content_signals[:20]

    return metadata
