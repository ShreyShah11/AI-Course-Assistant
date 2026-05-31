from __future__ import annotations

from typing import Any

from apps.worker.services.audio_chunking.run_pipeline import run_pipeline


def process_audio_chunking_request(
    file_path: str,
    course_id: str,
    course_name: str = "",
    lecture_id: str = "",
    lecture_number: int = 0,
    week_number: int = 0,
    lecture_title: str = "",
    professor: str = "",
) -> dict[str, Any]:
    return run_pipeline(
        file_path=file_path,
        course_id=course_id,
        course_name=course_name,
        lecture_id=lecture_id,
        lecture_number=lecture_number,
        week_number=week_number,
        lecture_title=lecture_title,
        professor=professor,
    )
