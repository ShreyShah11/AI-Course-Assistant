from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.core.supabase import supabase

BUCKET = "materials"


def store_upload(course_id: UUID, upload: UploadFile) -> str:
    clean_name = Path(upload.filename or "material").name
    object_key = f"{course_id}/{uuid4().hex}-{clean_name}"

    upload.file.seek(0)

    supabase.storage.from_(BUCKET).upload(
        object_key,
        upload.file.read(),
        {"content-type": upload.content_type or "application/octet-stream"},
    )

    return object_key