from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.core.config import get_settings


def store_upload(course_id: UUID, upload: UploadFile) -> Path:
    course_dir = get_settings().upload_dir / str(course_id)
    course_dir.mkdir(parents=True, exist_ok=True)
    clean_name = Path(upload.filename or "material").name
    target = course_dir / f"{uuid4().hex}-{clean_name}"
    with target.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return target
