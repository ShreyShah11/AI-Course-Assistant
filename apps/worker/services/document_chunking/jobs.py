from __future__ import annotations

from typing import Any

from apps.worker.services.document_chunking.run_pipeline import run_pipeline


def process_document_chunking_request(
    file_paths: list[str],
    course_id: str,
    **_legacy_kwargs,
):
    print("Calling run_pipeline()...")

    try:
        result = run_pipeline(
            file_paths=file_paths,
            course_id=course_id,
        )
        print("run_pipeline returned:", result)
        return result

    except Exception as e:
        import traceback

        print("ERROR IN run_pipeline")
        print(type(e).__name__, str(e))
        traceback.print_exc()
        raise