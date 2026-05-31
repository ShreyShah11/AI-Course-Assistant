from __future__ import annotations

import importlib
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
AUDIO_PIPELINE_DIR = (
    PROJECT_ROOT
    / "apps"
    / "api"
    / "pipelines"
    / "chunking pipeline"
    / "audio pipeline"
)
PIPELINE_FILE = AUDIO_PIPELINE_DIR / "ingestion_pipeline"
TRANSCRIPT_MODULE = "generate_transcripts"

SUPPORTED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def validate_file_path(file_path: str) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported audio type. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    return path


def _add_audio_pipeline_to_import_path() -> None:
    pipeline_dir = str(AUDIO_PIPELINE_DIR)
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)


def load_transcript_module():
    load_environment("worker")
    _add_audio_pipeline_to_import_path()
    return importlib.import_module(TRANSCRIPT_MODULE)


def load_ingestion_pipeline_module():
    load_environment("worker")
    _add_audio_pipeline_to_import_path()
    load_transcript_module()
    loader = SourceFileLoader("audio_ingestion_pipeline", str(PIPELINE_FILE))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f"Cannot load audio pipeline from {PIPELINE_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
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


def _pinecone_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record["metadata"])
    metadata["segment_indices"] = [
        str(index) for index in metadata.get("segment_indices", [])
    ]
    metadata["concepts"] = metadata.get("concepts", [])[:50]
    metadata["keywords"] = metadata.get("keywords", [])[:50]
    metadata["text"] = record["text"][:8_000]
    return metadata


def run_pipeline(
    file_path: str,
    course_id: str,
    course_name: str = "",
    lecture_id: str = "",
    lecture_number: int = 0,
    week_number: int = 0,
    lecture_title: str = "",
    professor: str = "",
) -> dict[str, Any]:
    load_environment("worker")
    path = validate_file_path(file_path)
    target_index = get_course_index_name(course_id)
    target_namespace = os.getenv("AUDIO_CHUNKING_NAMESPACE", "audio-chunks")
    transcript_module = load_transcript_module()
    pipeline = load_ingestion_pipeline_module()

    transcript = transcript_module.generate_transcript(str(path))
    identity = pipeline.LectureIdentity(
        course_id=course_id,
        course_name=course_name,
        lecture_id=lecture_id or path.stem,
        lecture_number=lecture_number,
        week_number=week_number,
        lecture_title=lecture_title,
        professor=professor,
    )
    chunks_by_strategy = pipeline.run_pipeline(transcript, identity)
    chunks = pipeline.flatten_chunks(chunks_by_strategy)
    if not chunks:
        raise RuntimeError("Audio ingestion produced no chunks.")

    records = pipeline.to_store_records(chunks)
    vectors = _embed_chunks_with_gemini([record["text"] for record in records])
    if len(vectors) != len(records):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} vectors for {len(records)} chunks."
        )

    dimension = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    ensure_index(index_name=target_index, dimension=dimension)
    index = get_pinecone_client().Index(target_index)
    index.upsert(
        vectors=[
            {
                "id": record["id"],
                "values": vector,
                "metadata": _pinecone_metadata(record),
            }
            for record, vector in zip(records, vectors)
        ],
        namespace=target_namespace,
    )

    return {
        "status": "completed",
        "file_path": str(path),
        "lecture_id": identity.lecture_id,
        "chunk_count": len(chunks),
        "chunks_by_strategy": {
            strategy: len(strategy_chunks)
            for strategy, strategy_chunks in chunks_by_strategy.items()
        },
        "namespace": target_namespace,
        "pinecone_index": target_index,
        "transcript": transcript.model_dump(),
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
