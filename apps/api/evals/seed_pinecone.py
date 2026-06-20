"""
seed_pinecone.py
================
Seeds all chunk fixtures from tests.py into a dedicated Pinecone index
so the live eval runner can retrieve against real vector data.

What it does
------------
1.  Reads all chunk fixtures from evals/Test suites/tests.py:
      QNA_CHUNKS, AUDIO_CHUNKS, DOCUMENT_CHUNKS,
      IMAGE_CHUNKS, HANDWRITTEN_CHUNKS, YOUTUBE_CHUNKS

2.  Embeds each chunk's text with Gemini (RETRIEVAL_DOCUMENT task type,
    same as the production ingestion workers)

3.  Upserts to the target Pinecone index under the correct namespace:
      qna         → qna-chunks
      audio       → audio-chunks
      documents   → document-chunks
      image       → image-chunks
      handwritten → handwritten-chunks
      youtube     → youtube-chunks

4.  Returns a seeding report (chunk counts per namespace, index name)

The index is created automatically if it does not exist.
Existing vectors with the same IDs are overwritten (idempotent).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EMBED_MODEL    = None   # resolved from env at runtime
EMBED_DIM      = None
EMBED_BATCH    = 20     # max items per Gemini embed batch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — load test suite
# ─────────────────────────────────────────────────────────────────────────────

def _load_suites():
    suite_file = Path(__file__).resolve().parent / "Test suites" / "tests.py"
    spec = importlib.util.spec_from_file_location("_eval_test_suites", suite_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction — consistent with what each pipeline sends to Pinecone
# ─────────────────────────────────────────────────────────────────────────────

def _text_for_chunk(chunk: dict, source: str) -> str:
    """Extract the primary text field that should be embedded."""
    if source == "qna":
        parts = []
        for q in chunk.get("questions", []):
            parts.append(f"Q: {q}")
        for a in chunk.get("answers", []):
            parts.append(f"A: {a}")
        return " ".join(parts) or chunk.get("raw_text", "") or " "

    if source == "image":
        # image chunks have nested metadata; text is top-level
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        summary = meta.get("gemini_summary", "")
        return (f"{summary}\n\n{text}" if summary else text) or " "

    if source == "audio":
        return chunk.get("text", "") or " "

    if source == "documents":
        raw = chunk.get("raw_text", "")
        tables = chunk.get("tables_text", "")
        return (f"{raw}\n\n{tables}" if tables else raw) or " "

    if source == "handwritten":
        return chunk.get("text_preview", "") or chunk.get("summary", "") or " "

    if source == "youtube":
        return chunk.get("raw_text", "") or chunk.get("summary", "") or " "

    return " "


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction — mirrors _pinecone_metadata() in each worker's run_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def _meta_for_chunk(chunk: dict, source: str, course_id: str) -> dict:
    """Build a flat Pinecone-safe metadata dict from a fixture chunk."""
    if source == "qna":
        return {
            "chunk_id":     chunk.get("chunk_id", ""),
            "source_file":  chunk.get("source_file", ""),
            "chunk_type":   chunk.get("chunk_type", ""),
            "topic":        chunk.get("topic_label", ""),
            "topic_cluster": chunk.get("topic_cluster", 0),
            "questions":    chunk.get("questions", [])[:10],
            "answers":      chunk.get("answers", [])[:10],
            "raw_text":     chunk.get("raw_text", "")[:8000],
            "marks_hint":   chunk.get("marks_hint") or -1,
            "year_hint":    chunk.get("year_hint") or -1,
            "difficulty_hint": chunk.get("difficulty_hint", ""),
            "has_sub_parts": chunk.get("has_sub_parts", False),
            "course_id":    course_id,
        }

    if source == "image":
        meta = chunk.get("metadata", {})
        return {
            "chunk_id":         meta.get("chunk_id", ""),
            "source_file":      meta.get("source_file", ""),
            "course_id":        course_id,
            "page_numbers":     str(meta.get("page_numbers", [])),
            "ocr_confidence":   meta.get("ocr_confidence", 0.0),
            "image_type":       meta.get("image_type", ""),
            "gemini_summary":   meta.get("gemini_summary", "")[:2000],
            "gemini_topic":     meta.get("gemini_topic", ""),
            "gemini_keywords":  meta.get("gemini_keywords", [])[:20],
            "has_code":         meta.get("has_code", False),
            "has_formula":      meta.get("has_formula", False),
            "chunk_type":       meta.get("chunk_type", ""),
            "section":          meta.get("section", ""),
        }

    if source == "audio":
        return {
            "chunk_id":      chunk.get("chunk_id", ""),
            "course_id":     course_id,
            "lecture_title": chunk.get("lecture_title", ""),
            "professor":     chunk.get("professor", ""),
            "strategy":      chunk.get("strategy", ""),
            "start_seconds": chunk.get("start_seconds", 0.0),
            "end_seconds":   chunk.get("end_seconds", 0.0),
            "start_label":   "",
            "end_label":     "",
            "week_number":   chunk.get("week_number") or -1,
            "concepts":      chunk.get("concepts", [])[:10],
            "keywords":      chunk.get("keywords", [])[:20],
            "avg_confidence": chunk.get("avg_confidence", 0.0),
            "source_file":   chunk.get("source_file", ""),
        }

    if source == "documents":
        return {
            "chunk_id":      chunk.get("chunk_id", ""),
            "source_file":   chunk.get("file_name", chunk.get("source", "")),
            "course_id":     course_id,
            "section_title": chunk.get("section_title", ""),
            "page_start":    chunk.get("page_start", 0),
            "page_end":      chunk.get("page_end", 0),
            "has_table":     chunk.get("has_table", False),
            "has_image":     chunk.get("has_image", False),
            "has_list":      chunk.get("has_list", False),
            "keywords":      chunk.get("keywords", ""),
            "chunk_index":   chunk.get("chunk_index", 0),
            "total_chunks":  chunk.get("total_chunks", 0),
            "ingested_at":   chunk.get("ingested_at", ""),
        }

    if source == "handwritten":
        return {
            "chunk_id":    chunk.get("source_file", "") + "_" + str(chunk.get("page_number", 0)),
            "source_file": chunk.get("source_file", ""),
            "course_id":   course_id,
            "topic":       chunk.get("topic", ""),
            "content_type": chunk.get("content_type", ""),
            "keywords":    chunk.get("keywords", [])[:20],
            "summary":     chunk.get("summary", "")[:2000],
            "has_diagrams": chunk.get("has_diagrams", False),
            "has_equations": chunk.get("has_equations", False),
            "ink_quality": chunk.get("ink_quality", ""),
            "model_used":  chunk.get("model_used", ""),
        }

    if source == "youtube":
        return {
            "chunk_id":    chunk.get("chunk_id", ""),
            "course_id":   course_id,
            "video_id":    chunk.get("video_id", ""),
            "video_title": chunk.get("video_title", ""),
            "author":      chunk.get("author", ""),
            "topic":       chunk.get("topic", ""),
            "chapter_title": chunk.get("chapter_title", ""),
            "start_label": chunk.get("start_label", ""),
            "end_label":   chunk.get("end_label", ""),
            "deep_link":   chunk.get("deep_link", ""),
            "summary":     chunk.get("summary", "")[:2000],
            "concept_tags": chunk.get("concept_tags", [])[:20],
            "source_file": chunk.get("video_id", ""),
        }

    return {}


def _chunk_id_for(chunk: dict, source: str) -> str:
    if source == "image":
        return chunk.get("metadata", {}).get("chunk_id", "")
    if source == "handwritten":
        sf = chunk.get("source_file", "hw")
        pg = chunk.get("page_number", 0)
        idx = chunk.get("chunk_index", 0)
        return f"hw_{sf}_{pg}_{idx}"
    return chunk.get("chunk_id", "")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def _embed_texts(texts: list[str]) -> list[list[float]]:
    from google import genai
    from google.genai import types as gtypes

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    dim   = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    client = genai.Client(api_key=api_key.strip())

    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        for text in batch:
            result = client.models.embed_content(
                model=model,
                contents=text.strip() or " ",
                config=gtypes.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=dim,
                ),
            )
            vectors.append(list(result.embeddings[0].values))
        # Small pause between batches to avoid rate limiting
        if i + EMBED_BATCH < len(texts):
            time.sleep(0.5)

    return vectors


# ─────────────────────────────────────────────────────────────────────────────
# Pinecone helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_index(index_name: str):
    from pinecone import Pinecone, ServerlessSpec

    api_key  = os.getenv("PINECONE_API_KEY", "")
    cloud    = os.getenv("PINECONE_CLOUD", "aws")
    region   = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    dim      = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key.strip())
    existing = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing:
        logger.info("Creating Pinecone index '%s'…", index_name)
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait for it to be ready
        for _ in range(30):
            time.sleep(2)
            status = pc.describe_index(index_name).status
            if getattr(status, "ready", False):
                break
        logger.info("Index '%s' is ready.", index_name)
    else:
        logger.info("Pinecone index '%s' already exists, reusing.", index_name)

    return pc.Index(index_name)


NAMESPACE_MAP = {
    "qna":         os.getenv("QNA_CHUNKING_NAMESPACE",         "qna-chunks"),
    "audio":       os.getenv("AUDIO_CHUNKING_NAMESPACE",       "audio-chunks"),
    "documents":   os.getenv("DOCUMENT_CHUNKING_NAMESPACE",    "document-chunks"),
    "image":       os.getenv("IMAGE_CHUNKING_NAMESPACE",       "image-chunks"),
    "handwritten": os.getenv("HANDWRITTEN_CHUNKING_NAMESPACE", "handwritten-chunks"),
    "youtube":     os.getenv("YOUTUBE_CHUNKING_NAMESPACE",     "youtube-chunks"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main seed function
# ─────────────────────────────────────────────────────────────────────────────

def seed_eval_index(
    course_id: str,
) -> dict:
    """
    Embed all fixture chunks from tests.py and upsert them into the
    Pinecone index that corresponds to `course_id`.

    Uses get_course_index_name(course_id) — the SAME naming function that
    the retrieval pipeline uses internally — so seeding and retrieval always
    target the exact same Pinecone index.

    Parameters
    ----------
    course_id : str
        Friendly course/eval name (e.g. "test01", "CS301").
        The actual Pinecone index name is derived automatically.

    Returns
    -------
    dict
        Seeding report including the resolved index name.
    """
    import sys
    from pathlib import Path as _Path

    # Make sure project root is importable for apps.worker.core.pinecone
    api_root = _Path(__file__).resolve().parents[1]
    project_root = api_root.parents[1]
    for p in (str(project_root), str(api_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from apps.worker.core.pinecone import get_course_index_name

    # Resolve to the real Pinecone index name — same as the retrieval pipeline
    real_index_name = get_course_index_name(course_id)
    logger.info(
        "course_id=%r  →  Pinecone index=%r",
        course_id, real_index_name,
    )

    suites = _load_suites()

    all_sources: dict[str, list[dict]] = {
        "qna":         suites.QNA_CHUNKS,
        "audio":       suites.AUDIO_CHUNKS,
        "documents":   suites.DOCUMENT_CHUNKS,
        "image":       suites.IMAGE_CHUNKS,
        "handwritten": suites.HANDWRITTEN_CHUNKS,
        "youtube":     suites.YOUTUBE_CHUNKS,
    }

    index = _get_or_create_index(real_index_name)
    report: dict[str, int] = {}
    total_upserted = 0

    for source, chunks in all_sources.items():
        if not chunks:
            report[source] = 0
            continue

        namespace = NAMESPACE_MAP.get(source, f"{source}-chunks")
        logger.info("Seeding %d %s chunks into namespace '%s'…", len(chunks), source, namespace)

        texts = [_text_for_chunk(c, source) for c in chunks]
        metas = [_meta_for_chunk(c, source, course_id) for c in chunks]
        ids   = [_chunk_id_for(c, source) for c in chunks]

        valid = [(cid, txt, meta) for cid, txt, meta in zip(ids, texts, metas) if cid]
        if not valid:
            logger.warning("No valid chunk IDs for source '%s', skipping.", source)
            report[source] = 0
            continue

        v_ids, v_texts, v_metas = zip(*valid)
        vectors = _embed_texts(list(v_texts))

        upsert_data = [
            {"id": cid, "values": vec, "metadata": meta}
            for cid, vec, meta in zip(v_ids, vectors, v_metas)
        ]

        index.upsert(vectors=upsert_data, namespace=namespace)
        report[source] = len(upsert_data)
        total_upserted += len(upsert_data)
        logger.info(
            "  OK  %d vectors → '%s' / '%s'",
            len(upsert_data), real_index_name, namespace,
        )

    return {
        "course_id":      course_id,
        "index_name":     real_index_name,   # the actual Pinecone index name
        "per_source":     report,
        "total_upserted": total_upserted,
    }
