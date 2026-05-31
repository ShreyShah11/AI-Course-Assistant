from __future__ import annotations

from typing import Any

from apps.worker.services.qna_chunking.run_pipeline import run_pipeline


def process_qna_chunking_request(
    file_paths: list[str],
    course_id: str,
    semantic_threshold: float = 0.45,
    max_chunk_tokens: int = 800,
) -> dict[str, Any]:
    return run_pipeline(
        file_paths=file_paths,
        course_id=course_id,
        semantic_threshold=semantic_threshold,
        max_chunk_tokens=max_chunk_tokens,
    )
