from __future__ import annotations

import os
import re
from hashlib import sha256
from dataclasses import dataclass

from pinecone import Pinecone
from apps.worker.core.config import load_environment


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str
    index_name: str
    cloud: str
    region: str
    namespace: str


def _reserved_namespace_names() -> set[str]:
    return {
        "document-chunks",
        "image-chunks",
        "qna-chunks",
        "audio-chunks",
        os.getenv("DOCUMENT_CHUNKING_NAMESPACE", "document-chunks"),
        os.getenv("IMAGE_CHUNKING_NAMESPACE", "image-chunks"),
        os.getenv("QNA_CHUNKING_NAMESPACE", "qna-chunks"),
        os.getenv("AUDIO_CHUNKING_NAMESPACE", "audio-chunks"),
    }


def get_course_index_name(course_id: str) -> str:
    """Return a stable Pinecone-safe index name for one course."""
    normalized_course_id = course_id.strip()
    if not normalized_course_id:
        raise ValueError("course_id is required to select the Pinecone index.")

    slug = re.sub(r"[^a-z0-9]+", "-", normalized_course_id.lower()).strip("-")
    slug = slug or "course"
    digest = sha256(normalized_course_id.encode("utf-8")).hexdigest()[:8]
    prefix = os.getenv("PINECONE_COURSE_INDEX_PREFIX", "course").strip().lower()
    prefix = re.sub(r"[^a-z0-9]+", "-", prefix).strip("-") or "course"
    prefix = prefix[:20].rstrip("-") or "course"
    max_slug_length = 45 - len(prefix) - len(digest) - 2
    slug = slug[:max_slug_length].rstrip("-") or "course"
    return f"{prefix}-{slug}-{digest}"


def get_pinecone_settings() -> PineconeSettings:
    load_environment()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured.")

    return PineconeSettings(
        api_key=api_key,
        index_name=os.getenv("PINECONE_INDEX_NAME", "rag-index"),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
        namespace=os.getenv("PINECONE_NAMESPACE", "documents"),
    )


def get_pinecone_client() -> Pinecone:
    settings = get_pinecone_settings()
    return Pinecone(api_key=settings.api_key)


def ensure_index(
    index_name: str | None = None,
    dimension: int = 1536,
    metric: str = "cosine",
) -> None:
    from pinecone import ServerlessSpec

    settings = get_pinecone_settings()
    pc = get_pinecone_client()
    name = index_name or settings.index_name
    if name in _reserved_namespace_names():
        raise RuntimeError(
            f"Refusing to create Pinecone index '{name}'. "
            "That name is reserved for a namespace. Use course_id-based indexes instead."
        )
    existing = [idx.name for idx in pc.list_indexes()]

    if name in existing:
        return

    pc.create_index(
        name=name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
    )


def get_index(index_name: str | None = None):
    settings = get_pinecone_settings()
    ensure_index(index_name=index_name)
    return get_pinecone_client().Index(index_name or settings.index_name)
