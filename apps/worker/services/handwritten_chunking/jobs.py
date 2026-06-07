from __future__ import annotations

from typing import Any

from apps.worker.services.handwritten_chunking.run_pipeline import run_pipeline


def process_handwritten_chunking_request(
    file_path: str,
    course_id: str,
    course_name: str = "",
    subject: str = "",
    instructor: str = "",
    semester: str = "",
    university: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return run_pipeline(
        file_path=file_path,
        course_id=course_id,
        course_name=course_name,
        subject=subject,
        instructor=instructor,
        semester=semester,
        university=university,
        tags=tags,
    )
