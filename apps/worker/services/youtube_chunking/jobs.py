from __future__ import annotations

from typing import Any

from apps.worker.services.youtube_chunking.run_pipeline import run_pipeline


def process_youtube_chunking_request(
    url: str,
    course_id: str,
    course_name: str = "",
    instructor: str = "",
    semester: str = "",
    subject: str = "",
    tags: list[str] | None = None,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    return run_pipeline(
        url=url,
        course_id=course_id,
        course_name=course_name,
        instructor=instructor,
        semester=semester,
        subject=subject,
        tags=tags,
        languages=languages,
    )
